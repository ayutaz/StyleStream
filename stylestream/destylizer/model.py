"""Destylizer model: Conformer + FSQ + ASR Head for content extraction.

The Destylizer removes style information from speech, retaining only
linguistic content.  It operates on pre-extracted HuBERT layer-18 features
(not raw audio) and produces continuous content features *fc* that the
Stylizer consumes.

Architecture::

    HuBERT features (B, T, 768)
        -> LayerNorm
        -> ConformerEncoder (x6)
        -> fc (B, T, 768)           # content feature, used at inference
            -> FSQ [5, 3, 3]        # training-only bottleneck
            -> ASR Head              # training-only CTC loss
            -> logits (B, T, vocab)

At **training** time the full pipeline runs: Conformer -> FSQ -> ASR Head,
and the CTC loss drives the Conformer to learn content-preserving features.

At **inference** time only ``extract_content_features`` is called, which
returns the Conformer output *fc* (before FSQ) for the downstream Stylizer.

StyleStream spec
----------------
- 6 Conformer layers, hidden 768, FFN 3072, 12 heads, kernel 31
- FSQ levels [5, 3, 3], codebook size 45
- ASR head: 4 Transformer layers, CTC loss, vocab 30
- All at 50 Hz frame rate (hop 320, sr 16 kHz)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from stylestream.destylizer.conformer import ConformerEncoder
from stylestream.destylizer.fsq import FSQ
from stylestream.destylizer.asr_head import ASRHead

logger = logging.getLogger(__name__)

# Default HuBERT feature dimension.
_HUBERT_DIM: int = 768


class Destylizer(nn.Module):
    """Full Destylizer model: Conformer + FSQ + ASR Head.

    Takes pre-extracted HuBERT layer-18 features as input (not raw audio).
    HuBERT features are pre-computed and stored on disk by
    :class:`~stylestream.data.hubert_extractor.HuBERTExtractor`.

    Parameters
    ----------
    config : DestylizerConfig, optional
        Structured configuration with ``conformer``, ``fsq``, and
        ``asr_decoder`` sub-configs.  When *None*, keyword arguments
        are used to build the sub-components with their defaults.
    **kwargs
        Override individual sub-component settings when *config* is not
        provided.  Accepted keys:

        - ``num_layers``, ``hidden_size``, ``ffn_size``, ``num_heads``,
          ``kernel_size``, ``dropout``, ``causal`` -- forwarded to
          :class:`ConformerEncoder`.
        - ``fsq_levels``, ``fsq_hidden_size`` -- forwarded to :class:`FSQ`.
        - ``asr_loss_type``, ``asr_num_layers``, ``asr_ffn_size``,
          ``asr_num_heads``, ``vocab_size``, ``label_smoothing``,
          ``asr_dropout`` -- forwarded to :class:`ASRHead`.
    """

    def __init__(self, config=None, **kwargs: Any) -> None:
        super().__init__()

        if config is not None:
            self._init_from_config(config)
        else:
            self._init_from_kwargs(kwargs)

        logger.info(
            "Destylizer: conformer=%d layers, fsq=%s, asr_head vocab=%d",
            self.conformer.num_layers if hasattr(self.conformer, "num_layers") else "?",
            getattr(self.fsq, "_levels", "?"),
            self.asr_head.vocab_size if hasattr(self.asr_head, "vocab_size") else "?",
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_from_config(self, config) -> None:
        """Build sub-modules from a :class:`DestylizerConfig`."""
        c = config.conformer
        f = config.fsq
        a = config.asr_decoder
        hidden = c.hidden_size

        # Input normalisation (stabilises training with pre-extracted features)
        self.input_norm = nn.LayerNorm(hidden)

        # Core encoder
        self.conformer = ConformerEncoder(
            num_layers=c.num_layers,
            hidden_size=hidden,
            ffn_size=c.ffn_size,
            num_heads=c.num_heads,
            kernel_size=c.kernel_size,
            dropout=0.1,
            causal=False,
        )

        # FSQ bottleneck (training only)
        self.fsq = FSQ(levels=list(f.levels), hidden_size=hidden)

        # ASR decoder (training only)
        self.asr_head = ASRHead(
            loss_type=getattr(a, "loss_type", "ctc"),
            hidden_size=hidden,
            vocab_size=getattr(a, "vocab_size", 30),
            num_layers=a.num_layers,
            ffn_size=a.ffn_size,
            num_heads=getattr(a, "num_heads", c.num_heads),
            dropout=getattr(a, "dropout", 0.1),
            label_smoothing=getattr(a, "label_smoothing", 0.1),
        )

    def _init_from_kwargs(self, kw: dict[str, Any]) -> None:
        """Build sub-modules from flat keyword arguments."""
        hidden = kw.get("hidden_size", _HUBERT_DIM)

        # Input normalisation
        self.input_norm = nn.LayerNorm(hidden)

        # Conformer
        self.conformer = ConformerEncoder(
            num_layers=kw.get("num_layers", 6),
            hidden_size=hidden,
            ffn_size=kw.get("ffn_size", 3072),
            num_heads=kw.get("num_heads", 12),
            kernel_size=kw.get("kernel_size", 31),
            dropout=kw.get("dropout", 0.1),
            causal=kw.get("causal", False),
        )

        # FSQ
        self.fsq = FSQ(
            levels=kw.get("fsq_levels", [5, 3, 3]),
            hidden_size=kw.get("fsq_hidden_size", hidden),
        )

        # ASR Head
        self.asr_head = ASRHead(
            loss_type=kw.get("asr_loss_type", "ctc"),
            hidden_size=hidden,
            vocab_size=kw.get("vocab_size", 30),
            num_layers=kw.get("asr_num_layers", 4),
            ffn_size=kw.get("asr_ffn_size", 3072),
            num_heads=kw.get("asr_num_heads", 12),
            dropout=kw.get("asr_dropout", 0.1),
            label_smoothing=kw.get("label_smoothing", 0.1),
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: DestylizerConfig) -> "Destylizer":
        """Build a Destylizer from a :class:`DestylizerConfig`.

        Parameters
        ----------
        config : DestylizerConfig
            Full destylizer configuration.

        Returns
        -------
        Destylizer
            Initialised model (on CPU, with random weights).
        """
        return cls(config=config)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_time_last(x: torch.Tensor) -> torch.Tensor:
        """Normalise input to (B, T, D) layout.

        The :class:`~stylestream.data.destylizer_dataset.DestylizerDataset`
        stores features as ``(B, 768, T)`` (channels-first).  The Conformer
        expects ``(B, T, 768)``.  This helper auto-detects and transposes
        when necessary.

        Heuristic: if ``x.shape[1] == 768`` and ``x.shape[2] != 768``, the
        input is channels-first and needs transposing.  When ambiguous
        (e.g. ``T == 768``) we assume the caller has already arranged
        ``(B, T, 768)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3-D tensor (B, T, 768) or (B, 768, T), got shape {x.shape}"
            )
        B, d1, d2 = x.shape
        # Channels-first: (B, 768, T) where T != 768
        if d1 == _HUBERT_DIM and d2 != _HUBERT_DIM:
            return x.transpose(1, 2).contiguous()  # -> (B, T, 768)
        return x

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        hubert_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full forward pass for training.

        Parameters
        ----------
        hubert_features : Tensor
            Pre-extracted HuBERT layer-18 features.  Accepts either
            ``(B, T, 768)`` or ``(B, 768, T)`` layout (auto-detected).
        padding_mask : Tensor, optional
            ``(B, T)`` boolean mask where ``True`` marks padded positions.
        target_ids : Tensor, optional
            ``(B, S)`` character token IDs for ASR / CTC loss.  Not used
            inside ``forward`` directly (the trainer computes the loss),
            but is passed through the ASR head to produce logits.

        Returns
        -------
        dict
            ``'content_features'``
                ``(B, T, 768)`` -- continuous Conformer output *fc*
                (pre-FSQ), which is passed to the Stylizer at inference.
            ``'logits'``
                ``(B, T, vocab)`` -- ASR head output log-probabilities /
                logits for CTC loss computation.
            ``'fsq_info'``
                ``dict`` -- FSQ diagnostics: ``indices``,
                ``codebook_usage``, ``perplexity``, ``pre_quant``.
        """
        # --- Normalise layout ---
        x = self._ensure_time_last(hubert_features)  # (B, T, 768)

        # --- Input LayerNorm (stabilises HuBERT feature scale) ---
        x = self.input_norm(x)  # (B, T, 768)

        # --- Conformer encoder ---
        content_features = self.conformer(x, padding_mask=padding_mask)  # (B, T, 768)

        # --- FSQ bottleneck ---
        quantized, fsq_info = self.fsq(content_features)  # (B, T, 768), dict

        # --- ASR head ---
        logits = self.asr_head(
            encoder_output=quantized,
            target_ids=target_ids,
            encoder_padding_mask=padding_mask,
        )  # (B, T, vocab) or (B, S, vocab)

        return {
            "content_features": content_features,
            "logits": logits,
            "fsq_info": fsq_info,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_content_features(
        self,
        hubert_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract content features *fc* for inference (no FSQ, no ASR).

        This is the only method called at inference time.  It runs the
        input LayerNorm and Conformer encoder, returning the continuous
        pre-FSQ features that the Stylizer consumes.

        Parameters
        ----------
        hubert_features : Tensor
            ``(B, T, 768)`` or ``(B, 768, T)`` HuBERT features.
        padding_mask : Tensor, optional
            ``(B, T)`` boolean mask (``True`` = padded).

        Returns
        -------
        Tensor
            ``(B, T, 768)`` content features.
        """
        x = self._ensure_time_last(hubert_features)
        x = self.input_norm(x)
        content_features = self.conformer(x, padding_mask=padding_mask)
        return content_features

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Parameters
        ----------
        trainable_only : bool
            If *True* (default), count only parameters with
            ``requires_grad=True``.

        Returns
        -------
        int
            Total number of (trainable) parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n_params = self.num_parameters(trainable_only=True) / 1e6
        return (
            f"{self.__class__.__name__}(\n"
            f"  (input_norm): {self.input_norm}\n"
            f"  (conformer): {self.conformer}\n"
            f"  (fsq): {self.fsq}\n"
            f"  (asr_head): {self.asr_head}\n"
            f"  trainable_params={n_params:.2f}M\n"
            f")"
        )
