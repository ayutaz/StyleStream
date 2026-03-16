"""End-to-end content feature extraction API for StyleStream inference.

Wraps HuBERT (frozen) + trained Destylizer (Conformer only) into a single
callable that goes from raw audio to content features *fc*.  The FSQ and
ASR head are discarded -- only the Conformer output is used.

These content features are consumed by the Stylizer for voice conversion.

Usage::

    from stylestream.destylizer.feature_extractor import ContentFeatureExtractor

    extractor = ContentFeatureExtractor.from_checkpoint(
        "outputs/destylizer/checkpoints/best",
        device="cuda",
    )

    # Single file  ->  (768, T) @ 50 Hz
    fc = extractor.extract("audio.wav")

    # Batch  ->  list of (768, T_i)
    features = extractor.extract_batch(["a.wav", "b.wav"])

    # From pre-computed HuBERT features  ->  (B, T, 768)
    fc_batch = extractor.extract_from_hubert_features(hubert_feats)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from stylestream.config import DestylizerConfig
from stylestream.destylizer.model import Destylizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE: int = 16_000
_HOP_LENGTH: int = 320  # HuBERT CNN encoder stride -> 50 Hz at 16 kHz
_HUBERT_DIM: int = 768
_CHUNK_OVERLAP_SAMPLES: int = _HOP_LENGTH  # 320 samples = 1 frame at 50 Hz


def _expected_frames(num_samples: int) -> int:
    """Number of HuBERT output frames for *num_samples* at 16 kHz."""
    return math.ceil(num_samples / _HOP_LENGTH)


# ---------------------------------------------------------------------------
# ContentFeatureExtractor
# ---------------------------------------------------------------------------

class ContentFeatureExtractor:
    """End-to-end content feature extraction from raw audio.

    Wraps HuBERT (frozen) + trained Destylizer (Conformer only) for inference.
    This is the API that the Stylizer uses to get content features.

    Parameters
    ----------
    destylizer : Destylizer
        Trained Destylizer model (only Conformer weights are used).
    device : str
        Device for inference.
    hubert_layer : int
        HuBERT layer to extract (default 18).
    max_audio_sec : float
        Maximum audio length in seconds before chunking (default 30.0).
    use_fp16 : bool
        Run Destylizer forward passes in float16 for speed (default False).
    """

    def __init__(
        self,
        destylizer: nn.Module,
        device: str = "cuda",
        hubert_layer: int = 18,
        max_audio_sec: float = 30.0,
        use_fp16: bool = False,
    ) -> None:
        self._device = device
        self._hubert_layer = hubert_layer
        self._max_audio_samples = int(max_audio_sec * _SAMPLE_RATE)
        self._use_fp16 = use_fp16

        # Place Destylizer on device in eval mode; freeze all parameters.
        self._destylizer = destylizer.to(device).eval()
        for param in self._destylizer.parameters():
            param.requires_grad = False

        if use_fp16:
            self._destylizer = self._destylizer.half()

        # HuBERT is loaded lazily on first use to avoid heavy model loading
        # when not needed (e.g. when using extract_from_hubert_features).
        self._hubert_model: Any | None = None
        self._hubert_extract_fn: Any | None = None

    # ------------------------------------------------------------------
    # Construction from checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cuda",
        config: DestylizerConfig | dict | None = None,
        use_fp16: bool = False,
    ) -> "ContentFeatureExtractor":
        """Load from a training checkpoint.

        Supports two checkpoint formats:

        1. **CheckpointManager directory** -- a directory containing
           ``model.safetensors`` (and optionally ``config.yaml``).  The
           model state dict is loaded via ``safetensors``.
        2. **Plain state dict file** -- a single ``.pt`` / ``.bin`` file
           that ``torch.load`` can read into a ``dict[str, Tensor]``.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint directory or state-dict file.
        device : str
            Device for inference.
        config : DestylizerConfig or dict, optional
            Model configuration.  When *None*, attempts to load
            ``config.yaml`` from the checkpoint directory, or falls
            back to default :class:`DestylizerConfig`.
        use_fp16 : bool
            Run Destylizer in float16 (default False).

        Returns
        -------
        ContentFeatureExtractor
            Ready-to-use extractor.
        """
        checkpoint_path = Path(checkpoint_path)

        # ---- Resolve config ----
        destylizer_config = cls._resolve_config(checkpoint_path, config)

        # ---- Build model (CPU first, then move) ----
        destylizer = Destylizer(config=destylizer_config)

        # ---- Load weights ----
        state_dict = cls._load_state_dict(checkpoint_path)
        destylizer.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded Destylizer weights from %s (%d tensors)",
            checkpoint_path,
            len(state_dict),
        )

        return cls(
            destylizer=destylizer,
            device=device,
            hubert_layer=destylizer_config.hubert.layer,
            use_fp16=use_fp16,
        )

    # ------------------------------------------------------------------
    # Public extraction API
    # ------------------------------------------------------------------

    def extract(self, audio_path: str | Path) -> torch.Tensor:
        """Extract content features from a single audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to an audio file (WAV, FLAC, etc.).

        Returns
        -------
        torch.Tensor
            ``(768, T)`` float32 tensor on CPU, where T = ceil(samples / 320).
        """
        from stylestream.utils.audio import load_audio

        self._ensure_hubert_loaded()
        assert self._hubert_extract_fn is not None

        waveform = load_audio(str(audio_path), sr=_SAMPLE_RATE)  # (samples,)
        total_samples = waveform.shape[0]

        if total_samples == 0:
            logger.warning("Empty audio file: %s", audio_path)
            return torch.zeros(_HUBERT_DIM, 0)

        # Short audio -- single forward pass.
        if total_samples <= self._max_audio_samples:
            hubert_feat = self._hubert_single(waveform)  # (768, T)
            content = self._destylizer_forward(hubert_feat.unsqueeze(0))  # (1, T, 768)
            return content.squeeze(0).transpose(0, 1).cpu().float()  # (768, T)

        # Long audio -- chunk with overlap.
        return self._extract_chunked(waveform)

    def extract_batch(self, audio_paths: list[str | Path]) -> list[torch.Tensor]:
        """Extract content features from multiple audio files.

        Each file is processed independently.  Long files are automatically
        chunked.

        Parameters
        ----------
        audio_paths : list
            List of audio file paths.

        Returns
        -------
        list[torch.Tensor]
            One ``(768, T_i)`` float32 CPU tensor per input file.
        """
        return [self.extract(p) for p in audio_paths]

    def extract_from_hubert_features(
        self,
        hubert_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract content features from pre-computed HuBERT features.

        Useful when HuBERT features are already available (e.g., cached on
        disk from the training pipeline).

        Parameters
        ----------
        hubert_features : torch.Tensor
            ``(B, T, 768)`` or ``(B, 768, T)`` pre-extracted HuBERT features.
        padding_mask : torch.Tensor, optional
            ``(B, T)`` boolean mask where ``True`` marks padded positions.

        Returns
        -------
        torch.Tensor
            ``(B, T, 768)`` content features on the same device as the
            Destylizer model.
        """
        return self._destylizer_forward(hubert_features, padding_mask)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Content feature dimension (768)."""
        return _HUBERT_DIM

    @property
    def frame_rate(self) -> int:
        """Feature frame rate in Hz (50)."""
        return _SAMPLE_RATE // _HOP_LENGTH  # 50

    @property
    def device(self) -> str:
        """Device the models are placed on."""
        return self._device

    # ------------------------------------------------------------------
    # Internal: HuBERT management
    # ------------------------------------------------------------------

    def _ensure_hubert_loaded(self) -> None:
        """Lazily load HuBERT on first use."""
        if self._hubert_model is not None:
            return
        from stylestream.utils.hub import load_hubert

        logger.info(
            "Loading HuBERT-Large (layer %d) on %s ...",
            self._hubert_layer,
            self._device,
        )
        self._hubert_model, self._hubert_extract_fn = load_hubert(
            self._device, self._hubert_layer
        )

    def _hubert_single(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run HuBERT on a single 1-D waveform.

        Parameters
        ----------
        waveform : (samples,) CPU tensor.

        Returns
        -------
        (768, T) CPU float32 tensor.
        """
        assert self._hubert_extract_fn is not None
        batch = waveform.unsqueeze(0)  # (1, samples)
        with torch.no_grad():
            feat = self._hubert_extract_fn(batch)  # (1, 768, T)
        return feat.squeeze(0).cpu()  # (768, T)

    # ------------------------------------------------------------------
    # Internal: Destylizer forward
    # ------------------------------------------------------------------

    def _destylizer_forward(
        self,
        hubert_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the Destylizer Conformer encoder.

        Handles device transfer and optional fp16 casting.

        Parameters
        ----------
        hubert_features : (B, T, 768) or (B, 768, T)
        padding_mask : (B, T), optional

        Returns
        -------
        (B, T, 768) content features on the Destylizer device.
        """
        x = hubert_features.to(self._device)
        mask = padding_mask.to(self._device) if padding_mask is not None else None

        if self._use_fp16:
            x = x.half()

        with torch.no_grad():
            content = self._destylizer.extract_content_features(x, padding_mask=mask)

        return content.float()

    # ------------------------------------------------------------------
    # Internal: chunked extraction for long audio
    # ------------------------------------------------------------------

    def _extract_chunked(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from a long waveform by chunking with overlap.

        Reuses the same overlap/trim logic as
        :class:`~stylestream.data.hubert_extractor.HuBERTExtractor`.

        Parameters
        ----------
        waveform : (samples,) 1-D CPU tensor.

        Returns
        -------
        (768, T) float32 CPU tensor.
        """
        assert self._hubert_extract_fn is not None

        total_samples = waveform.shape[0]
        chunks = self._split_with_overlap(waveform)
        features: list[torch.Tensor] = []

        for idx, chunk in enumerate(chunks):
            # HuBERT forward
            hubert_feat = self._hubert_single(chunk)  # (768, T_chunk)

            # Destylizer forward -- (1, T, 768)
            content = self._destylizer_forward(hubert_feat.unsqueeze(0))
            # -> (768, T_chunk) on CPU
            feat = content.squeeze(0).transpose(0, 1).cpu().float()

            # Trim overlap frames from interior chunks.
            overlap_frames = _expected_frames(_CHUNK_OVERLAP_SAMPLES)
            if idx > 0:
                feat = feat[:, overlap_frames:]
            if idx < len(chunks) - 1:
                feat = feat[:, :-overlap_frames]

            features.append(feat)

        concatenated = torch.cat(features, dim=1)  # (768, T_total)

        # Trim or pad to the expected frame count.
        expected_t = _expected_frames(total_samples)
        current_t = concatenated.shape[1]
        if current_t > expected_t:
            concatenated = concatenated[:, :expected_t]
        elif current_t < expected_t:
            pad = torch.zeros(
                _HUBERT_DIM,
                expected_t - current_t,
                dtype=concatenated.dtype,
            )
            concatenated = torch.cat([concatenated, pad], dim=1)

        return concatenated

    def _split_with_overlap(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """Split *waveform* into chunks of at most *_max_audio_samples*.

        Adjacent chunks overlap by ``_CHUNK_OVERLAP_SAMPLES`` so that
        HuBERT's CNN feature extractor has valid context at boundaries.

        Parameters
        ----------
        waveform : (samples,) 1-D tensor.

        Returns
        -------
        list[torch.Tensor]
            Each chunk is a 1-D tensor.
        """
        total = waveform.shape[0]
        step = self._max_audio_samples - _CHUNK_OVERLAP_SAMPLES
        chunks: list[torch.Tensor] = []

        start = 0
        while start < total:
            end = min(start + self._max_audio_samples, total)
            chunks.append(waveform[start:end])

            next_start = start + step

            # If the remaining tail is shorter than the overlap region,
            # absorb it into the last chunk.
            if next_start < total and (total - next_start) < _CHUNK_OVERLAP_SAMPLES:
                chunks[-1] = waveform[start:total]
                break

            start = next_start

        return chunks

    # ------------------------------------------------------------------
    # Internal: checkpoint / config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(
        checkpoint_path: Path,
        config: DestylizerConfig | dict | None,
    ) -> DestylizerConfig:
        """Resolve a DestylizerConfig from the caller's argument or the checkpoint."""
        if isinstance(config, DestylizerConfig):
            return config

        if isinstance(config, dict):
            return DestylizerConfig(**config)

        # Try to load config.yaml from the checkpoint (or its parent).
        for candidate in [checkpoint_path / "config.yaml",
                          checkpoint_path.parent / "config.yaml",
                          checkpoint_path.parent.parent / "config.yaml"]:
            if candidate.exists():
                try:
                    from omegaconf import OmegaConf
                    raw = OmegaConf.load(candidate)
                    # The config may be a full ExperimentConfig or just
                    # a DestylizerConfig section.
                    if hasattr(raw, "destylizer"):
                        return DestylizerConfig(**OmegaConf.to_container(raw.destylizer, resolve=True))
                    if hasattr(raw, "conformer"):
                        return DestylizerConfig(**OmegaConf.to_container(raw, resolve=True))
                except Exception as exc:
                    logger.warning(
                        "Could not parse config from %s: %s. Falling back to defaults.",
                        candidate,
                        exc,
                    )

        logger.info("No config found; using default DestylizerConfig.")
        return DestylizerConfig()

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
                    pt_files[0], map_location="cpu", weights_only=True
                )
                if isinstance(state, dict) and "model" in state:
                    return state["model"]
                return state

            raise FileNotFoundError(
                f"No model weights found in checkpoint directory: {checkpoint_path}"
            )

        # Case 2: single safetensors file
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(checkpoint_path))

        # Case 3: single .pt / .bin file
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            return state["model"]
        return state

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self._destylizer.parameters()) / 1e6
        fp = "fp16" if self._use_fp16 else "fp32"
        hubert_status = "loaded" if self._hubert_model is not None else "lazy"
        return (
            f"ContentFeatureExtractor(\n"
            f"  device={self._device!r},\n"
            f"  precision={fp},\n"
            f"  hubert_layer={self._hubert_layer},\n"
            f"  hubert_status={hubert_status},\n"
            f"  destylizer_params={n_params:.2f}M,\n"
            f"  max_audio_sec={self._max_audio_samples / _SAMPLE_RATE:.1f},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  frame_rate={self.frame_rate}Hz,\n"
            f")"
        )
