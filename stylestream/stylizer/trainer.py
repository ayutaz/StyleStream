"""Stylizer trainer: DiT + Conditional Flow Matching on mel inpainting.

Extends :class:`BaseTrainer` to train the Stylizer pipeline (DiT +
WavLM-TDNN style encoder + CFM) on pre-extracted mel spectrograms and
content features.  Training uses the spectrogram inpainting objective
with classifier-free guidance dropout.

Training spec (paper Section 10.8):
    - 400k steps, batch 64, AdamW, cosine annealing with 2k warmup
    - Peak LR 1e-4, betas (0.9, 0.999), weight decay 0.01
    - Gradient clip 1.0, bf16 mixed precision
    - Dataset: Emilia-EN (~50k hours)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stylestream.data.manifest import Manifest
from stylestream.data.stylizer_dataset import build_stylizer_dataloader
from stylestream.stylizer.model import Stylizer
from stylestream.training.trainer import BaseTrainer


class StylizerTrainer(BaseTrainer):
    """Trainer for the StyleStream Stylizer (DiT + CFM).

    Extends :class:`BaseTrainer` with Stylizer-specific model
    construction, data loading, loss computation, and validation.

    The trainer expects the following config structure (OmegaConf)::

        config.training.steps = 400000
        config.training.batch_size = 64
        config.training.peak_lr = 1e-4
        config.training.warmup_steps = 2000
        config.training.gradient_clip = 1.0
        config.training.mixed_precision = "bf16"
        config.training.log_interval = 100
        config.training.save_interval = 10000
        config.training.val_interval = 5000

        config.data.manifest_path = "data/manifests/emilia.csv"
        config.data.mel_dir = "data/processed/mel"
        config.data.content_features_dir = "data/processed/content_features"
        config.data.val_manifest_path = None

        # Stylizer sub-configs: dit, style_encoder, cfm, cfg
        config.dit.num_layers = 16
        config.style_encoder.model_id = "microsoft/wavlm-base-plus-sv"
        config.cfm.nfe = 16
        config.cfg.strength = 2.0

    Parameters
    ----------
    config :
        OmegaConf ``DictConfig`` with at least ``training``, ``data``,
        ``dit``, ``style_encoder``, ``cfm``, and ``cfg`` sub-configs.
    """

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self) -> nn.Module:
        """Build the Stylizer model from config.

        Returns
        -------
        nn.Module
            Stylizer model (DiT + WavLM-TDNN style encoder + CFM).
        """
        model = Stylizer(config=self.config)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            "Stylizer model built: %s total params, %s trainable",
            f"{n_params:,}",
            f"{n_trainable:,}",
        )
        return model

    # ------------------------------------------------------------------
    # Optimizer (override for paper-specific betas)
    # ------------------------------------------------------------------

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build AdamW with paper-specified betas and weight decay.

        Paper defaults: betas=(0.9, 0.999), weight_decay=0.01.
        These can be overridden via ``config.training.betas`` and
        ``config.training.weight_decay``.

        Parameters
        ----------
        model : nn.Module
            The model whose parameters will be optimised.

        Returns
        -------
        torch.optim.Optimizer
            Configured AdamW optimiser.
        """
        betas_cfg = getattr(self.config.training, "betas", [0.9, 0.999])
        betas = tuple(betas_cfg) if not isinstance(betas_cfg, tuple) else betas_cfg
        weight_decay = getattr(self.config.training, "weight_decay", 0.01)

        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.peak_lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def build_dataloader(self) -> DataLoader:
        """Build the training DataLoader from manifest + pre-extracted features.

        Reads ``config.data.manifest_path``, ``config.data.mel_dir``,
        and ``config.data.content_features_dir`` to construct a
        :func:`build_stylizer_dataloader`.

        Returns
        -------
        DataLoader
            Yields dicts with ``mel``, ``content_features``, ``mask``,
            ``context_mel``, ``style_waveform``, and CFG drop flags.
        """
        manifest_path = self.config.data.manifest_path
        mel_dir = self.config.data.mel_dir
        content_features_dir = getattr(
            self.config.data, "content_features_dir", None
        )
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)

        manifest = Manifest.load(manifest_path)
        self.logger.info(
            "Training manifest loaded: %d utterances from %s",
            len(manifest),
            manifest_path,
        )

        return build_stylizer_dataloader(
            manifest=manifest,
            mel_dir=mel_dir,
            content_features_dir=content_features_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def build_val_dataloader(self) -> DataLoader | None:
        """Build the validation DataLoader if ``val_manifest_path`` is set.

        Returns
        -------
        DataLoader or None
            *None* if no validation manifest is configured.
        """
        val_path = getattr(self.config.data, "val_manifest_path", None)
        if val_path is None:
            self.logger.info("No val_manifest_path configured, skipping validation.")
            return None

        mel_dir = self.config.data.mel_dir
        content_features_dir = getattr(
            self.config.data, "content_features_dir", None
        )
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)

        manifest = Manifest.load(val_path)
        self.logger.info(
            "Validation manifest loaded: %d utterances from %s",
            len(manifest),
            val_path,
        )

        self._val_dataloader = build_stylizer_dataloader(
            manifest=manifest,
            mel_dir=mel_dir,
            content_features_dir=content_features_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return self._val_dataloader

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """Compute the CFM loss for a single batch.

        Steps:
            1. Transpose mel/content/context from ``(B, D, T)`` to
               ``(B, T, D)`` (dataset returns channels-first, model
               expects time-first).
            2. Forward through the Stylizer model.
            3. Return loss dict.

        Parameters
        ----------
        batch : dict
            From :class:`StylizerCollator` with keys ``mel``,
            ``content_features``, ``mask``, ``context_mel``,
            ``style_waveform``, ``cfg_drop_content``,
            ``cfg_drop_context``, ``cfg_drop_style``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``loss`` : scalar CFM loss (masked MSE of velocity).
        """
        # Transpose from channels-first to time-first
        mel = batch["mel"].transpose(1, 2)                  # (B,100,T) -> (B,T,100)
        content = batch["content_features"].transpose(1, 2)  # (B,768,T) -> (B,T,768)
        mask = batch["mask"]                                  # (B, T) already correct
        style_waveform = batch["style_waveform"]              # (B, samples)

        # Forward through the Stylizer model
        # Note: context_mel is derived from mel and mask internally by the model
        result = self.model(
            mel=mel,
            content_features=content,
            mask=mask,
            style_waveform=style_waveform,
            cfg_drop_content=batch["cfg_drop_content"],
            cfg_drop_context=batch["cfg_drop_context"],
            cfg_drop_style=batch["cfg_drop_style"],
        )

        return {"loss": result["loss"]}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, float]:
        """Run validation and return average loss.

        Iterates over the full validation dataloader (built by
        :meth:`build_val_dataloader`) and aggregates metrics.

        Returns
        -------
        dict[str, float]
            ``loss`` -- average CFM loss over the validation set.
            Empty dict if no validation dataloader is available.
        """
        val_dl = getattr(self, "_val_dataloader", None)
        if val_dl is None:
            return {}

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_dl:
                # Move batch to device (Accelerate handles this for the
                # training dataloader, but val_dl may not be prepared)
                batch = _move_batch(batch, self.accelerator.device)

                loss_dict = self.compute_loss(batch)
                total_loss += loss_dict["loss"].item()
                n_batches += 1

        if n_batches == 0:
            return {}

        return {"loss": total_loss / n_batches}

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Log model summary and config at the start of training."""
        self.logger.info("Config: %s", _config_summary(self.config))


# ======================================================================
# Helpers
# ======================================================================


def _move_batch(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in *batch* to *device*."""
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _config_summary(config) -> str:
    """Return a concise string summary of the config for logging."""
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_yaml(config, resolve=True)
    except Exception:
        return str(config)
