"""Streaming Destylizer for StyleStream streaming inference.

Streaming variant of the Destylizer that replaces the frozen HuBERT
feature extractor with a causal, unfrozen StreamingHuBERT, and uses
causal Conformer blocks for chunk-by-chunk processing.

Trained via MSE distillation: the offline Destylizer serves as
teacher, and this streaming version (student) is trained to match
the teacher's content feature output.

StyleStream spec:
    - StreamingHuBERT: layer 18, unfrozen, chunked causal attention
    - HuBERT projection: Linear(1024, 768) -- new layer for streaming
    - Conformer: 6 layers, causal=True, ALiBi, kernel 31
    - Chunk size: 30 frames (600ms @ 50Hz)
    - Distillation: L_distill = MSE(fc_streaming, fc_offline.detach())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from stylestream.destylizer.conformer import ConformerEncoder
from stylestream.streaming.hubert_causal import StreamingHuBERT

logger = logging.getLogger(__name__)

# HuBERT-Large hidden dimension (1024 for facebook/hubert-large-ls960-ft).
_HUBERT_LARGE_DIM: int = 1024

# Conformer / content feature dimension.
_CONFORMER_DIM: int = 768


class StreamingDestylizer(nn.Module):
    """Streaming Destylizer with causal HuBERT and Conformer.

    Streaming variant of the Destylizer that uses:
    - StreamingHuBERT with chunked causal attention and causal CNN
    - ConformerEncoder with causal=True for causal convolutions
    - Chunked causal attention masks for Conformer self-attention

    For MSE distillation training, both the streaming (student) and
    offline (teacher) Destylizers extract content features, and the
    MSE loss between them drives the student to match the teacher.

    Parameters
    ----------
    chunk_size : int
        Chunk size in frames (default 30 = 600ms @ 50Hz).
    hidden_size : int
        Conformer hidden size (default 768).
    num_layers : int
        Number of Conformer layers (default 6).
    ffn_size : int
        Conformer FFN size (default 3072).
    num_heads : int
        Conformer attention heads (default 12).
    kernel_size : int
        Conformer depthwise conv kernel (default 31).
    hubert_model_id : str
        HuBERT model ID (default "facebook/hubert-large-ls960-ft").
    hubert_layer : int
        HuBERT layer to extract (default 18).
    max_cache_frames : int
        KV cache max frames (default 250).
    """

    def __init__(
        self,
        chunk_size: int = 30,
        hidden_size: int = _CONFORMER_DIM,
        num_layers: int = 6,
        ffn_size: int = 3072,
        num_heads: int = 12,
        kernel_size: int = 31,
        hubert_model_id: str = "facebook/hubert-large-ls960-ft",
        hubert_layer: int = 18,
        max_cache_frames: int = 250,
    ) -> None:
        super().__init__()

        self.chunk_size = chunk_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --- StreamingHuBERT (unfrozen by default) ---
        self.hubert = StreamingHuBERT(
            model_id=hubert_model_id,
            layer=hubert_layer,
            chunk_size=chunk_size,
            max_cache_frames=max_cache_frames,
            frozen=False,
        )

        # --- Projection: HuBERT 1024-dim -> Conformer 768-dim ---
        # This is a NEW layer not present in the offline Destylizer,
        # since the offline model receives pre-extracted 768-dim features
        # (via a separate extraction pipeline) whereas the streaming
        # model takes raw HuBERT-Large output at 1024-dim.
        self.hubert_proj = nn.Linear(_HUBERT_LARGE_DIM, hidden_size)

        # --- Input LayerNorm (matches offline Destylizer) ---
        self.input_norm = nn.LayerNorm(hidden_size)

        # --- Causal Conformer encoder ---
        self.conformer = ConformerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dropout=0.1,
            causal=True,
        )

        logger.info(
            "StreamingDestylizer: hubert=%s layer=%d, proj=%d->%d, "
            "conformer=%d layers (causal), chunk_size=%d",
            hubert_model_id,
            hubert_layer,
            _HUBERT_LARGE_DIM,
            hidden_size,
            num_layers,
            chunk_size,
        )

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract content features from raw audio.

        Full forward pass used during distillation training.  The
        returned ``content_features`` are compared against the offline
        Destylizer's output via MSE loss.

        Parameters
        ----------
        waveform : Tensor
            Raw audio at 16kHz, shape ``(B, T_samples)``.
        padding_mask : Tensor or None
            ``(B, T_frames)`` boolean padding mask where ``True``
            marks padded positions.  Frame-level mask aligned with
            the Conformer output (50 Hz).

        Returns
        -------
        dict
            ``'content_features'`` : Tensor
                ``(B, T, 768)`` -- pre-FSQ continuous content features
                suitable for distillation loss or downstream Stylizer.
        """
        # StreamingHuBERT: raw audio -> causal features
        hubert_features = self.hubert(waveform)  # (B, T, 1024)

        # Project HuBERT features to Conformer dimension
        x = self.hubert_proj(hubert_features)  # (B, T, 768)

        # Input LayerNorm (stabilises feature scale)
        x = self.input_norm(x)

        # Causal Conformer encoder
        content_features = self.conformer(x, padding_mask=padding_mask)  # (B, T, 768)

        return {"content_features": content_features}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_content_features(
        self,
        waveform: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract content features for inference.

        Convenience wrapper around :meth:`forward` that disables
        gradient computation and returns only the content feature
        tensor (not wrapped in a dict).

        Parameters
        ----------
        waveform : Tensor
            Raw audio at 16kHz, shape ``(B, T_samples)``.
        padding_mask : Tensor or None
            ``(B, T_frames)`` boolean padding mask.

        Returns
        -------
        Tensor
            ``(B, T, 768)`` content features.
        """
        return self.forward(waveform, padding_mask)["content_features"]

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_from_offline(
        self,
        offline_checkpoint_path: str | Path,
    ) -> None:
        """Load weights from an offline Destylizer checkpoint.

        Transfers the ``input_norm`` and ``conformer`` weights from
        a trained offline Destylizer.  The ``hubert_proj`` layer is
        left randomly initialised because the offline model does not
        have a corresponding projection (it receives pre-extracted
        768-dim features).  HuBERT weights are handled internally by
        :class:`StreamingHuBERT` (loaded from the pretrained model).

        Weight mapping:

        +-----------------------+----------------------------+
        | Streaming key         | Offline key                |
        +=======================+============================+
        | ``input_norm.*``      | ``input_norm.*``           |
        +-----------------------+----------------------------+
        | ``conformer.*``       | ``conformer.*``            |
        +-----------------------+----------------------------+
        | ``hubert_proj.*``     | *(not present -- random)*  |
        +-----------------------+----------------------------+
        | ``hubert.*``          | *(loaded by StreamingHuBERT)* |
        +-----------------------+----------------------------+

        Parameters
        ----------
        offline_checkpoint_path : str or Path
            Path to the offline Destylizer checkpoint.  Accepts:
            - A directory containing ``model.safetensors``
            - A single ``.safetensors`` file
            - A single ``.pt`` / ``.bin`` state dict file
        """
        offline_checkpoint_path = Path(offline_checkpoint_path)
        state_dict = self._load_state_dict(offline_checkpoint_path)

        # Filter to only the keys we want to transfer:
        # input_norm.* and conformer.*
        transferred: dict[str, torch.Tensor] = {}
        skipped: list[str] = []

        for key, value in state_dict.items():
            if key.startswith("input_norm.") or key.startswith("conformer."):
                transferred[key] = value
            else:
                skipped.append(key)

        if not transferred:
            logger.warning(
                "No matching weights found in offline checkpoint %s. "
                "The streaming model's input_norm and conformer remain "
                "randomly initialised.",
                offline_checkpoint_path,
            )
            return

        # Load the filtered state dict (strict=False to skip missing keys
        # like hubert.* and hubert_proj.*)
        missing, unexpected = self.load_state_dict(transferred, strict=False)

        logger.info(
            "Loaded %d tensors from offline checkpoint %s "
            "(skipped offline keys: %s, missing streaming keys: %s)",
            len(transferred),
            offline_checkpoint_path,
            skipped if skipped else "none",
            [k for k in missing if not k.startswith(("hubert.", "hubert_proj."))]
            or "none",
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> StreamingDestylizer:
        """Build a StreamingDestylizer from a configuration object.

        Supports :class:`~stylestream.config.DestylizerConfig` (with
        ``conformer`` and ``hubert`` sub-configs) as well as
        :class:`~stylestream.config.StreamingConfig` for the
        ``chunk_size`` parameter.

        Parameters
        ----------
        config : DestylizerConfig or dict-like
            Configuration with ``conformer``, ``hubert``, and
            optionally ``streaming`` sub-configs.

        Returns
        -------
        StreamingDestylizer
            Initialised model (on CPU, with random weights except
            HuBERT pretrained weights loaded by StreamingHuBERT).
        """
        conformer = config.conformer
        hubert = config.hubert

        # Chunk size: try streaming sub-config, then fall back to default.
        chunk_size = 30
        if hasattr(config, "streaming"):
            # streaming.chunk_size_ms -> frames: ms / 20 (at 50 Hz)
            chunk_size_ms = getattr(config.streaming, "chunk_size_ms", 600)
            chunk_size = chunk_size_ms // 20
        elif hasattr(config, "chunk_size"):
            chunk_size = config.chunk_size

        return cls(
            chunk_size=chunk_size,
            hidden_size=conformer.hidden_size,
            num_layers=conformer.num_layers,
            ffn_size=conformer.ffn_size,
            num_heads=conformer.num_heads,
            kernel_size=conformer.kernel_size,
            hubert_model_id=hubert.model_id,
            hubert_layer=hubert.layer,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        """Load model weights from a checkpoint path.

        Supports:
        - Directory with ``model.safetensors`` (CheckpointManager format)
        - Single ``.safetensors`` file
        - Single ``.pt`` / ``.bin`` file (torch.load)
        """
        # Case 1: directory containing model.safetensors
        if checkpoint_path.is_dir():
            safetensors_path = checkpoint_path / "model.safetensors"
            if safetensors_path.exists():
                from safetensors.torch import load_file

                return load_file(str(safetensors_path))

            # Fallback: look for a .pt file in the directory
            pt_files = list(checkpoint_path.glob("*.pt"))
            if pt_files:
                state = torch.load(
                    pt_files[0], map_location="cpu", weights_only=True,
                )
                if isinstance(state, dict) and "model" in state:
                    return state["model"]
                return state

            raise FileNotFoundError(
                f"No model weights found in checkpoint directory: "
                f"{checkpoint_path}"
            )

        # Case 2: single safetensors file
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(checkpoint_path))

        # Case 3: single .pt / .bin file
        state = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True,
        )
        if isinstance(state, dict) and "model" in state:
            return state["model"]
        return state

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_total = self.num_parameters(trainable_only=False) / 1e6
        n_train = self.num_parameters(trainable_only=True) / 1e6
        return (
            f"{self.__class__.__name__}(\n"
            f"  (hubert): {self.hubert}\n"
            f"  (hubert_proj): {self.hubert_proj}\n"
            f"  (input_norm): {self.input_norm}\n"
            f"  (conformer): {self.conformer}\n"
            f"  chunk_size={self.chunk_size}\n"
            f"  total_params={n_total:.2f}M\n"
            f"  trainable_params={n_train:.2f}M\n"
            f")"
        )
