"""WavLM-TDNN style encoder for the StyleStream Stylizer.

Extracts a fixed-length style embedding from variable-length target audio.
The style embedding conditions DiT mel-spectrogram generation via adaLN-Zero:
``c = emb(t) + e`` where ``emb(t)`` is the timestep embedding and ``e`` is
the style embedding produced by this encoder.

Architecture::

    Target audio waveform (16kHz, variable length)
        -> Frozen WavLM-Base-Plus-SV (13 hidden states)
        -> Learned weighted sum of all 13 layer outputs (softmax-normalised)
        -> TDNN layers (4x Conv1d + ReLU + BatchNorm)
        -> Attentive Statistics Pooling (variable length -> fixed vector)
        -> Linear projection -> style embedding e in R^768

WavLM weights are **frozen**; only the layer aggregation weights, TDNN, and
pooling layers are trained.

StyleStream Stylizer spec:
    - WavLM model: ``microsoft/wavlm-base-plus-sv`` (speaker verification)
    - 13 hidden states (1 CNN + 12 Transformer layers)
    - TDNN: 768->512 (k=5,d=1), 512->512 (k=3,d=2), 512->512 (k=3,d=3),
      512->512 (k=1,d=1)
    - Attentive statistics pooling: 512 -> attention -> [mean, std] -> 768
    - Output: 768-dim style embedding

Reference:
    Desplanques, Thienpondt & Demuynck.  "ECAPA-TDNN: Emphasized Channel
    Attention, Propagation and Aggregation in TDNN Based Speaker Verification."
    Interspeech 2020.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from stylestream.config import StyleEncoderConfig
    from stylestream.data.manifest import Utterance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TDNN block
# ---------------------------------------------------------------------------


class TDNNBlock(nn.Module):
    """Single TDNN block: Conv1d + ReLU + BatchNorm.

    Applies a 1-D convolution with dilation and "same" padding so that the
    temporal dimension is preserved.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size. Default 5.
    dilation : int
        Convolution dilation factor. Default 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        # "Same" padding to preserve temporal length.
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, C, T)
            Input in channels-first layout (Conv1d native format).

        Returns
        -------
        Tensor (B, out_channels, T)
            Output in channels-first layout.
        """
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in={self.conv.in_channels}, "
            f"out={self.conv.out_channels}, "
            f"k={self.conv.kernel_size[0]}, "
            f"d={self.conv.dilation[0]})"
        )


# ---------------------------------------------------------------------------
# Attentive Statistics Pooling
# ---------------------------------------------------------------------------


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling for variable-length sequences.

    Computes attention-weighted mean and standard deviation over the time
    axis, then projects the concatenated statistics to the output dimension.

    Parameters
    ----------
    input_size : int
        Input feature dimension (e.g. 512 from TDNN output).
    attention_size : int
        Hidden size for the attention MLP. Default 128.
    output_size : int
        Output embedding dimension. Default 768.
    """

    def __init__(
        self,
        input_size: int,
        attention_size: int = 128,
        output_size: int = 768,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.attention_size = attention_size
        self.output_size = output_size

        # Attention MLP: Linear -> Tanh -> Linear -> softmax over time.
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )

        # Projection from [mean, std] to output embedding.
        self.projection = nn.Linear(input_size * 2, output_size)
        self.bn = nn.BatchNorm1d(output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize attention and projection weights."""
        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attentive statistics pooling.

        Parameters
        ----------
        x : Tensor (B, T, input_size)
            Input sequence features.
        lengths : Tensor (B,), optional
            Actual sequence lengths for masking padded positions.
            Values should be in range ``[1, T]``.  If *None*, all
            positions are assumed valid (no padding).

        Returns
        -------
        Tensor (B, output_size)
            Pooled style embedding.
        """
        B, T, C = x.shape

        # Compute raw attention scores: (B, T, 1)
        attn_scores = self.attention(x)  # (B, T, 1)

        # Mask padded positions before softmax.
        if lengths is not None:
            # Build mask: (B, T, 1), True for valid positions.
            indices = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
            mask = indices < lengths.unsqueeze(1)  # (B, T)
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        # Softmax over time dimension -> attention weights (B, T, 1).
        alpha = F.softmax(attn_scores, dim=1)  # (B, T, 1)

        # Weighted mean: (B, C)
        mu = (alpha * x).sum(dim=1)  # (B, C)

        # Weighted standard deviation: (B, C)
        # sigma = sqrt( sum(alpha * (x - mu)^2) + eps )
        diff = x - mu.unsqueeze(1)  # (B, T, C)
        sigma = torch.sqrt((alpha * diff.square()).sum(dim=1) + 1e-6)  # (B, C)

        # Concatenate and project: (B, 2*C) -> (B, output_size)
        stats = torch.cat([mu, sigma], dim=-1)  # (B, 2*C)
        out = self.projection(stats)  # (B, output_size)
        out = self.bn(out)  # (B, output_size)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_size={self.input_size}, "
            f"attention_size={self.attention_size}, "
            f"output_size={self.output_size})"
        )


# ---------------------------------------------------------------------------
# Style Encoder
# ---------------------------------------------------------------------------


class StyleEncoder(nn.Module):
    """WavLM-TDNN style encoder for Stylizer conditioning.

    Extracts a fixed-length style embedding from variable-length raw audio.
    The frozen WavLM backbone produces hidden states from all 13 layers,
    which are aggregated via a learned weighted sum.  The aggregated features
    pass through a TDNN stack and attentive statistics pooling to yield a
    single 768-dim style vector.

    Only the layer aggregation weights, TDNN parameters, and pooling layers
    are trained; WavLM weights remain frozen.

    Parameters
    ----------
    wavlm_model_id : str
        HuggingFace model ID for WavLM. Default ``"microsoft/wavlm-base-plus-sv"``.
    hidden_size : int
        WavLM hidden dimension. Default 768.
    num_wavlm_layers : int
        Number of WavLM hidden states to aggregate. Default 13.
    tdnn_channels : int
        TDNN intermediate channels. Default 512.
    output_size : int
        Style embedding dimension. Default 768.
    freeze_wavlm : bool
        Whether to freeze WavLM weights. Default True.
    """

    def __init__(
        self,
        wavlm_model_id: str = "microsoft/wavlm-base-plus-sv",
        hidden_size: int = 768,
        num_wavlm_layers: int = 13,
        tdnn_channels: int = 512,
        output_size: int = 768,
        freeze_wavlm: bool = True,
    ) -> None:
        super().__init__()

        self.wavlm_model_id = wavlm_model_id
        self.hidden_size = hidden_size
        self.num_wavlm_layers = num_wavlm_layers
        self.tdnn_channels = tdnn_channels
        self.output_size = output_size
        self.freeze_wavlm = freeze_wavlm

        # --- WavLM backbone ---
        self.wavlm = self._load_wavlm(wavlm_model_id, freeze_wavlm)

        # --- Learned layer aggregation ---
        # Initialised to zeros so that softmax gives uniform weights (1/13).
        self.layer_weights = nn.Parameter(torch.zeros(num_wavlm_layers))

        # --- TDNN stack ---
        self.tdnn = nn.Sequential(
            TDNNBlock(hidden_size, tdnn_channels, kernel_size=5, dilation=1),
            TDNNBlock(tdnn_channels, tdnn_channels, kernel_size=3, dilation=2),
            TDNNBlock(tdnn_channels, tdnn_channels, kernel_size=3, dilation=3),
            TDNNBlock(tdnn_channels, tdnn_channels, kernel_size=1, dilation=1),
        )

        # --- Attentive statistics pooling ---
        self.pooling = AttentiveStatisticsPooling(
            input_size=tdnn_channels,
            attention_size=256,
            output_size=output_size,
        )

        logger.info(
            "StyleEncoder: wavlm=%s (frozen=%s), %d layers, "
            "tdnn_channels=%d, output=%d",
            wavlm_model_id,
            freeze_wavlm,
            num_wavlm_layers,
            tdnn_channels,
            output_size,
        )

    # ------------------------------------------------------------------
    # WavLM loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_wavlm(model_id: str, freeze: bool) -> nn.Module:
        """Load WavLM model from HuggingFace Hub.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier.
        freeze : bool
            If True, all WavLM parameters are frozen.

        Returns
        -------
        nn.Module
            The loaded WavLM model.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed or the WavLM model
            class cannot be imported.
        """
        try:
            from transformers import WavLMModel
        except ImportError as exc:
            raise ImportError(
                "The `transformers` library is required for the style encoder. "
                "Install it with: pip install transformers\n"
                "Or: uv sync --extra train"
            ) from exc

        logger.info("Loading WavLM from '%s'...", model_id)
        wavlm = WavLMModel.from_pretrained(model_id)

        if freeze:
            for param in wavlm.parameters():
                param.requires_grad_(False)
            logger.info("WavLM weights frozen.")

        return wavlm

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: StyleEncoderConfig) -> StyleEncoder:
        """Build a StyleEncoder from a :class:`StyleEncoderConfig`.

        Parameters
        ----------
        config : StyleEncoderConfig
            Style encoder configuration from ``stylestream.config``.

        Returns
        -------
        StyleEncoder
            Initialised encoder (on CPU, with random trainable weights
            and pretrained frozen WavLM weights).
        """
        return cls(
            wavlm_model_id=config.model_id,
            hidden_size=config.hidden_size,
            num_wavlm_layers=config.num_layers,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract style embedding from audio waveform.

        Parameters
        ----------
        waveform : Tensor (B, num_samples)
            Raw audio at 16kHz.

        Returns
        -------
        Tensor (B, output_size)
            Style embedding vector.
        """
        # --- WavLM feature extraction (frozen) ---
        hidden_states = self._extract_wavlm_features(waveform)

        # --- Learned layer aggregation ---
        aggregated = self._aggregate_layers(hidden_states)  # (B, T_wavlm, hidden_size)

        # --- TDNN ---
        # Single transpose into channels-first for the entire TDNN stack,
        # then back to channels-last for pooling (reduces 8 transposes to 2).
        tdnn_in = aggregated.transpose(1, 2)  # (B, hidden_size, T_wavlm)
        tdnn_out = self.tdnn(tdnn_in)         # (B, tdnn_channels, T_wavlm)
        tdnn_out = tdnn_out.transpose(1, 2)   # (B, T_wavlm, tdnn_channels)

        # --- Attentive statistics pooling ---
        embedding = self.pooling(tdnn_out)  # (B, output_size)

        return embedding

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_wavlm_features(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run frozen WavLM and return all hidden states.

        Uses ``inference_mode`` for the frozen WavLM forward pass (~3-5%
        speedup over ``no_grad``).  The returned tensors are cloned outside
        the inference context so they are compatible with downstream autograd
        (e.g. the learnable layer-weight aggregation).

        Parameters
        ----------
        waveform : Tensor (B, num_samples)
            Raw 16kHz audio.

        Returns
        -------
        tuple of Tensor
            ``num_wavlm_layers`` tensors, each ``(B, T_wavlm, hidden_size)``.
        """
        with torch.inference_mode():
            outputs = self.wavlm(waveform, output_hidden_states=True)
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors
            # including the CNN feature extractor output (index 0) and all
            # Transformer layer outputs (indices 1..12).
            hidden_states = outputs.hidden_states  # tuple of 13 tensors

        # Clone outside inference_mode so tensors can participate in autograd
        # (needed for learnable layer_weights aggregation downstream).
        return tuple(h.clone() for h in hidden_states)

    def _aggregate_layers(
        self, hidden_states: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Aggregate WavLM hidden states via learned weighted sum.

        Parameters
        ----------
        hidden_states : tuple of Tensor
            Each tensor is ``(B, T, hidden_size)``.

        Returns
        -------
        Tensor (B, T, hidden_size)
            Weighted sum of all layer outputs.
        """
        weights = F.softmax(self.layer_weights, dim=0)  # (num_layers,)

        # Weighted sum over layers.
        aggregated = torch.zeros_like(hidden_states[0])
        for w, h in zip(weights, hidden_states):
            aggregated = aggregated + w * h

        return aggregated

    # ------------------------------------------------------------------
    # Embedding pre-caching
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def extract_and_cache_embeddings(
        self,
        manifest: list[Utterance],
        output_dir: str | Path,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        """Pre-compute style embeddings for all utterances and save to disk.

        Each embedding is saved as a ``.pt`` file named ``{utt.stem}.pt``
        containing a float16 tensor of shape ``(output_size,)``.  Existing
        files are skipped so the method is idempotent / resumable.

        The model is cast to FP16 on GPU for faster extraction and lower
        memory usage.  Original device and dtype are restored afterwards
        so that training (which may need float32) is not affected.

        Parameters
        ----------
        manifest :
            List of :class:`Utterance` objects whose audio will be processed.
        output_dir :
            Directory in which to write the cached ``.pt`` files.
        batch_size :
            Number of utterances to process per forward pass.
        device :
            Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        """
        from stylestream.utils.audio import load_audio, pad_or_trim

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save original state so we can restore after extraction
        original_device = next(self.parameters()).device
        original_dtype = next(self.parameters()).dtype

        # Cast to FP16 for faster extraction on GPU
        self.to(device)
        if device != "cpu":
            self.half()
        self.eval()

        # 5 seconds at 16 kHz, matching _STYLE_SAMPLES in stylizer_dataset.py
        style_samples = 80_000

        total = len(manifest)
        processed = 0
        skipped = 0

        for start in range(0, total, batch_size):
            batch_utts = manifest[start : start + batch_size]

            # Skip utterances that already have cached embeddings
            to_process = []
            for utt in batch_utts:
                out_path = output_dir / f"{utt.stem}.pt"
                if out_path.exists():
                    skipped += 1
                else:
                    to_process.append(utt)

            if not to_process:
                continue

            # Load and pad/trim waveforms
            waveforms = []
            for utt in to_process:
                wav = load_audio(utt.audio_path, sr=16_000)
                wav = pad_or_trim(wav, style_samples)
                waveforms.append(wav)

            # Stack into batch: (B, style_samples)
            # Match model dtype (FP16 on GPU, FP32 on CPU)
            model_dtype = next(self.parameters()).dtype
            batch_wav = torch.stack(waveforms, dim=0).to(device=device, dtype=model_dtype)

            # Forward through style encoder
            embeddings = self.forward(batch_wav)  # (B, output_size)

            # Save each embedding individually as float16
            for utt, emb in zip(to_process, embeddings):
                out_path = output_dir / f"{utt.stem}.pt"
                torch.save(emb.cpu().half(), out_path)

            processed += len(to_process)
            if (start // batch_size + 1) % 10 == 0 or start + batch_size >= total:
                logger.info(
                    "Style embedding caching: %d/%d processed, %d skipped",
                    processed,
                    total,
                    skipped,
                )

        # Restore original device and dtype (e.g. float32 for training)
        self.to(device=original_device, dtype=original_dtype)
        logger.info(
            "Style embedding caching complete: %d processed, %d skipped, "
            "saved to %s",
            processed,
            skipped,
            output_dir,
        )

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
        n_trainable = self.num_parameters(trainable_only=True) / 1e6
        n_total = self.num_parameters(trainable_only=False) / 1e6
        return (
            f"{self.__class__.__name__}(\n"
            f"  (wavlm): {self.wavlm_model_id} "
            f"(frozen={self.freeze_wavlm})\n"
            f"  (layer_weights): Parameter({self.num_wavlm_layers})\n"
            f"  (tdnn): {self.tdnn}\n"
            f"  (pooling): {self.pooling}\n"
            f"  trainable_params={n_trainable:.2f}M, "
            f"total_params={n_total:.2f}M\n"
            f")"
        )
