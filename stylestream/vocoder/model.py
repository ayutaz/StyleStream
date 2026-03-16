"""Causal Vocos vocoder model for StyleStream.

Full model integrating the ConvNeXt backbone and ISTFT head for mel-to-waveform
synthesis. Supports warm starting from official Vocos checkpoints and causal
mode for streaming inference.

StyleStream spec:
    - 8 ConvNeXt blocks, hidden 512, intermediate 1536
    - Causal convolutions throughout
    - ISTFT head: n_fft=1024, hop=320
    - Input: 100-bin mel @ 50 Hz
    - Output: 16 kHz waveform
    - Warm start from charactr/vocos-mel-24khz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from stylestream.vocoder.backbone import VocosBackbone
from stylestream.vocoder.istft_head import ISTFTHead

logger = logging.getLogger(__name__)

# Default architecture constants matching the YAML and paper spec.
_DEFAULT_N_MELS: int = 100
_DEFAULT_HIDDEN_SIZE: int = 512
_DEFAULT_INTERMEDIATE_SIZE: int = 1536
_DEFAULT_NUM_LAYERS: int = 8
_DEFAULT_N_FFT: int = 1024
_DEFAULT_HOP_LENGTH: int = 320
_DEFAULT_KERNEL_SIZE: int = 7


class CausalVocos(nn.Module):
    """Causal Vocos vocoder: mel spectrogram -> waveform.

    ISTFT-based neural vocoder with ConvNeXt backbone and causal
    convolutions for streaming inference. Based on the Vocos architecture
    with all convolutions replaced by causal variants.

    Parameters
    ----------
    n_mels : int
        Number of mel bins (100).
    hidden_size : int
        Hidden dimension (512).
    intermediate_size : int
        ConvNeXt intermediate dimension (1536).
    num_layers : int
        Number of ConvNeXt blocks (8).
    n_fft : int
        FFT size for ISTFT head (1024).
    hop_length : int
        Hop length for ISTFT (320).
    kernel_size : int
        Kernel size for ConvNeXt depthwise conv (default 7).
    causal : bool
        Use causal convolutions (default True).
    """

    def __init__(
        self,
        n_mels: int = _DEFAULT_N_MELS,
        hidden_size: int = _DEFAULT_HIDDEN_SIZE,
        intermediate_size: int = _DEFAULT_INTERMEDIATE_SIZE,
        num_layers: int = _DEFAULT_NUM_LAYERS,
        n_fft: int = _DEFAULT_N_FFT,
        hop_length: int = _DEFAULT_HOP_LENGTH,
        kernel_size: int = _DEFAULT_KERNEL_SIZE,
        causal: bool = True,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.kernel_size = kernel_size
        self.causal = causal

        # ConvNeXt backbone: mel -> hidden features
        self.backbone = VocosBackbone(
            n_mels=n_mels,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            causal=causal,
        )

        # ISTFT head: hidden features -> waveform
        self.head = ISTFTHead(
            hidden_size=hidden_size,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        logger.info(
            "CausalVocos: %d ConvNeXt blocks, hidden=%d, intermediate=%d, "
            "causal=%s, n_fft=%d, hop=%d, n_mels=%d",
            num_layers,
            hidden_size,
            intermediate_size,
            causal,
            n_fft,
            hop_length,
            n_mels,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Parameters
        ----------
        mel : Tensor
            Mel spectrogram of shape ``(B, n_mels, T)`` (channels-first).

        Returns
        -------
        Tensor
            Waveform of shape ``(B, T_samples)`` where
            ``T_samples ~ T * hop_length``.
        """
        features = self.backbone(mel)  # (B, hidden_size, T)
        waveform = self.head(features)  # (B, T_samples)
        return waveform

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> CausalVocos:
        """Build from config object.

        Accepts either a :class:`~stylestream.config.VocoderConfig` or a
        full experiment config with ``.model`` and ``.mel`` sub-configs
        (as loaded from the YAML).  Falls back to sensible defaults for
        any missing attributes.

        Parameters
        ----------
        config
            A config object.  Supported layouts:

            1. **YAML-style config** with ``config.model`` (containing
               ``hidden_size``, ``num_layers``, ``intermediate_size``,
               ``causal``) and ``config.mel`` (containing ``n_mels``,
               ``n_fft``, ``hop_length``).
            2. **Flat config** with direct attributes matching the
               constructor parameters.

        Returns
        -------
        CausalVocos
            Initialised model (on CPU, with random weights).
        """
        # YAML-style: config.model + config.mel
        if hasattr(config, "model") and hasattr(config, "mel"):
            model_cfg = config.model
            mel_cfg = config.mel
            return cls(
                n_mels=getattr(mel_cfg, "n_mels", _DEFAULT_N_MELS),
                hidden_size=getattr(model_cfg, "hidden_size", _DEFAULT_HIDDEN_SIZE),
                intermediate_size=getattr(
                    model_cfg, "intermediate_size", _DEFAULT_INTERMEDIATE_SIZE
                ),
                num_layers=getattr(model_cfg, "num_layers", _DEFAULT_NUM_LAYERS),
                n_fft=getattr(mel_cfg, "n_fft", _DEFAULT_N_FFT),
                hop_length=getattr(mel_cfg, "hop_length", _DEFAULT_HOP_LENGTH),
                kernel_size=getattr(model_cfg, "kernel_size", _DEFAULT_KERNEL_SIZE),
                causal=getattr(model_cfg, "causal", True),
            )

        # Flat config (e.g. VocoderConfig or simple namespace)
        return cls(
            n_mels=getattr(config, "n_mels", _DEFAULT_N_MELS),
            hidden_size=getattr(config, "hidden_size", _DEFAULT_HIDDEN_SIZE),
            intermediate_size=getattr(
                config, "intermediate_size", _DEFAULT_INTERMEDIATE_SIZE
            ),
            num_layers=getattr(config, "num_layers", _DEFAULT_NUM_LAYERS),
            n_fft=getattr(config, "n_fft", _DEFAULT_N_FFT),
            hop_length=getattr(config, "hop_length", _DEFAULT_HOP_LENGTH),
            kernel_size=getattr(config, "kernel_size", _DEFAULT_KERNEL_SIZE),
            causal=getattr(config, "causal", True),
        )

    # ------------------------------------------------------------------
    # Warm start from official Vocos checkpoint
    # ------------------------------------------------------------------

    def load_warm_start(
        self, checkpoint_path_or_id: str
    ) -> dict[str, list[str]]:
        """Load weights from official Vocos checkpoint for warm starting.

        The official Vocos (``charactr/vocos-mel-24khz``) has different mel
        specs (80 bins, hop 256, 24 kHz) so:

        - ConvNeXt backbone layers: weights transfer if hidden_size matches.
        - Input embedding: shape mismatch (80 vs 100 mels) -- skip/reinitialize.
        - ISTFT head: may differ -- skip/reinitialize.

        Parameters
        ----------
        checkpoint_path_or_id : str
            Path to a local checkpoint file (``.pt`` / ``.safetensors``) or
            a HuggingFace model ID (e.g. ``"charactr/vocos-mel-24khz"``).

        Returns
        -------
        dict
            ``'loaded'``: list of parameter names that were loaded.
            ``'skipped'``: list of parameter names that were skipped
            due to shape mismatch or missing keys.
        """
        state_dict = self._load_external_state_dict(checkpoint_path_or_id)

        own_state = self.state_dict()
        loaded: list[str] = []
        skipped: list[str] = []

        for name, param in own_state.items():
            if name in state_dict:
                external_param = state_dict[name]
                if external_param.shape == param.shape:
                    own_state[name].copy_(external_param)
                    loaded.append(name)
                else:
                    skipped.append(name)
                    logger.debug(
                        "Skipped %s: shape mismatch (own=%s, checkpoint=%s)",
                        name,
                        param.shape,
                        external_param.shape,
                    )
            else:
                skipped.append(name)
                logger.debug("Skipped %s: not found in checkpoint", name)

        # Apply the updated state dict back to the model.
        self.load_state_dict(own_state, strict=False)

        logger.info(
            "Warm start: loaded %d / %d parameters (%d skipped)",
            len(loaded),
            len(loaded) + len(skipped),
            len(skipped),
        )

        return {"loaded": loaded, "skipped": skipped}

    @staticmethod
    def _load_external_state_dict(
        checkpoint_path_or_id: str,
    ) -> dict[str, torch.Tensor]:
        """Load a state dict from a local file or HuggingFace Hub.

        Supports ``.pt``, ``.pth``, ``.bin`` (via ``torch.load``) and
        ``.safetensors`` (via ``safetensors.torch.load_file``).  If the
        path does not point to an existing local file, it is treated as
        a HuggingFace model ID and downloaded via ``huggingface_hub``.

        Parameters
        ----------
        checkpoint_path_or_id : str
            Local file path or HuggingFace model ID.

        Returns
        -------
        dict[str, Tensor]
            Flat state dict mapping parameter names to tensors.
        """
        path = Path(checkpoint_path_or_id)

        # --- Local file ---
        if path.is_file():
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file

                return load_file(str(path))
            # .pt / .pth / .bin
            data = torch.load(str(path), map_location="cpu", weights_only=False)
            # Handle checkpoints that wrap the state dict in a container.
            if isinstance(data, dict) and "state_dict" in data:
                return data["state_dict"]
            if isinstance(data, dict) and "model" in data:
                return data["model"]
            return data

        # --- HuggingFace Hub ---
        logger.info(
            "Checkpoint not found locally; attempting HuggingFace Hub "
            "download for '%s' ...",
            checkpoint_path_or_id,
        )
        try:
            from huggingface_hub import hf_hub_download

            # Try safetensors first, then pytorch_model.bin.
            for filename in ("model.safetensors", "pytorch_model.bin"):
                try:
                    local_path = hf_hub_download(
                        repo_id=checkpoint_path_or_id, filename=filename
                    )
                    return CausalVocos._load_external_state_dict(local_path)
                except Exception:
                    continue

            # If neither file was found, raise.
            raise FileNotFoundError(
                f"Could not find model weights in HuggingFace repo "
                f"'{checkpoint_path_or_id}'"
            )
        except ImportError:
            raise FileNotFoundError(
                f"Checkpoint not found at '{checkpoint_path_or_id}' and "
                f"huggingface_hub is not installed for remote download."
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
            f"  (backbone): {self.num_layers} ConvNeXt blocks, "
            f"hidden={self.hidden_size}, "
            f"intermediate={self.intermediate_size}\n"
            f"  (head): ISTFT n_fft={self.n_fft}, hop={self.hop_length}\n"
            f"  n_mels={self.n_mels}, causal={self.causal}\n"
            f"  trainable_params={n_trainable:.2f}M, "
            f"total_params={n_total:.2f}M\n"
            f")"
        )
