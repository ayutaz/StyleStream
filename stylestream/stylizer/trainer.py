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

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stylestream.data.manifest import Manifest
from stylestream.data.stylizer_dataset import StylizerDataset, build_stylizer_dataloader
from stylestream.stylizer.model import Stylizer
from stylestream.training.trainer import BaseTrainer, ProgressiveSchedule


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
        """Build optimizer (AdamW or Lion) with paper-specified hyperparameters.

        Paper defaults: betas=(0.9, 0.999), weight_decay=0.01 for AdamW.
        Lion defaults: betas=(0.9, 0.99), weight_decay=0.01.
        These can be overridden via ``config.training.betas`` and
        ``config.training.weight_decay``.

        Parameters
        ----------
        model : nn.Module
            The model whose parameters will be optimised.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        from stylestream.training.trainer import Lion

        optimizer_name = getattr(self.config.training, "optimizer", "adamw").lower()
        weight_decay = getattr(self.config.training, "weight_decay", 0.01)

        if optimizer_name == "lion":
            betas_cfg = getattr(self.config.training, "betas", [0.9, 0.99])
            betas = tuple(betas_cfg) if not isinstance(betas_cfg, tuple) else betas_cfg
            return Lion(
                model.parameters(),
                lr=self.config.training.peak_lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        else:
            betas_cfg = getattr(self.config.training, "betas", [0.9, 0.999])
            betas = tuple(betas_cfg) if not isinstance(betas_cfg, tuple) else betas_cfg
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

        A reference to the underlying :class:`StylizerDataset` is stored as
        ``self._train_dataset`` so that progressive training can update its
        parameters mid-training.

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
        style_embeddings_dir = getattr(
            self.config.data, "style_embeddings_dir", None
        ) or None  # treat empty string as None
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)

        manifest = Manifest.load(manifest_path)
        self.logger.info(
            "Training manifest loaded: %d utterances from %s",
            len(manifest),
            manifest_path,
        )
        if style_embeddings_dir:
            self.logger.info(
                "Using pre-cached style embeddings from %s", style_embeddings_dir
            )

        dataloader = build_stylizer_dataloader(
            manifest=manifest,
            mel_dir=mel_dir,
            content_features_dir=content_features_dir,
            style_embeddings_dir=style_embeddings_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Store reference to the dataset for progressive training updates
        self._train_dataset: StylizerDataset = dataloader.dataset  # type: ignore[assignment]

        return dataloader

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
        style_embeddings_dir = getattr(
            self.config.data, "style_embeddings_dir", None
        ) or None  # treat empty string as None
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
            style_embeddings_dir=style_embeddings_dir,
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

        # Style: use pre-cached embedding if available, otherwise raw waveform
        style_waveform = batch.get("style_waveform")          # (B, samples) or None
        style_embedding = batch.get("style_embedding")        # (B, emb_dim) or None

        # Forward through the Stylizer model
        # Note: context_mel is derived from mel and mask internally by the model
        result = self.model(
            mel=mel,
            content_features=content,
            mask=mask,
            style_waveform=style_waveform,
            style_embedding=style_embedding,
            cfg_drop_content=batch["cfg_drop_content"],
            cfg_drop_context=batch["cfg_drop_context"],
            cfg_drop_style=batch["cfg_drop_style"],
        )

        return {"loss": result["loss"]}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, float]:
        """Run validation and return average loss (single forward pass only).

        Iterates over the full validation dataloader (built by
        :meth:`build_val_dataloader`) and aggregates metrics.

        The CFM training loss is computed at a single random timestep
        (one DiT forward pass per batch) -- it does NOT require multi-step
        ODE integration.  Full ODE sampling (NFE steps) is only performed
        when ``config.cfm.val_nfe > 0``, which is useful for periodically
        checking sample quality but is expensive (``val_nfe`` forward
        passes per sample).

        Returns
        -------
        dict[str, float]
            ``loss`` -- average CFM loss over the validation set.
            ``val_sample_mse`` -- reconstruction MSE from ODE sampling
            (only present when ``config.cfm.val_nfe > 0``).
            Empty dict if no validation dataloader is available.
        """
        val_dl = getattr(self, "_val_dataloader", None)
        if val_dl is None:
            return {}

        # Read val_nfe: 0 means loss-only (no ODE sampling), >0 enables
        # reduced-step sampling for quality monitoring.
        cfm_cfg = getattr(self.config, "cfm", None)
        val_nfe = getattr(cfm_cfg, "val_nfe", 0) if cfm_cfg is not None else 0

        total_loss = 0.0
        n_batches = 0

        with torch.inference_mode():
            for batch in val_dl:
                # Move batch to device (Accelerate handles this for the
                # training dataloader, but val_dl may not be prepared)
                batch = _move_batch(batch, self.accelerator.device)

                # Single-step CFM loss (1 forward pass, no ODE integration)
                loss_dict = self.compute_loss(batch)
                total_loss += loss_dict["loss"].item()
                n_batches += 1

        if n_batches == 0:
            return {}

        metrics: dict[str, float] = {"loss": total_loss / n_batches}

        # Optional: run ODE sampling with reduced NFE for quality monitoring.
        # Skipped by default (val_nfe=0) since it is expensive.
        if val_nfe > 0:
            self.logger.info(
                "Validation sampling with val_nfe=%d (reduced from training nfe=%d)",
                val_nfe,
                getattr(self.model, "nfe", 16),
            )
            original_nfe = self.model.nfe
            self.model.nfe = val_nfe
            try:
                with torch.inference_mode():
                    sample_batch = next(iter(val_dl))
                    sample_batch = _move_batch(sample_batch, self.accelerator.device)

                    mel = sample_batch["mel"].transpose(1, 2)
                    content = sample_batch["content_features"].transpose(1, 2)
                    mask = sample_batch["mask"]
                    style_waveform = sample_batch.get("style_waveform")

                    if style_waveform is not None:
                        context_mel = mel * (1.0 - mask.unsqueeze(-1))
                        sampled = self.model.sample(
                            content_features=content,
                            style_waveform=style_waveform,
                            context_mel=context_mel,
                            mask=mask,
                            nfe=val_nfe,
                        )
                        # Compute reconstruction MSE on the sampled output
                        mask_expanded = mask.unsqueeze(-1)
                        sample_mse = (
                            (((sampled - mel) * mask_expanded) ** 2).sum()
                            / (mask_expanded.sum() * mel.shape[-1] + 1e-8)
                        ).item()
                        metrics["val_sample_mse"] = sample_mse
            finally:
                self.model.nfe = original_nfe

        return metrics

    # ------------------------------------------------------------------
    # Hooks (progressive training integration)
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Log config and initialise progressive schedule if enabled."""
        self.logger.info("Config: %s", _config_summary(self.config))

        # Initialise progressive training schedule
        progressive = getattr(
            getattr(self.config, "stylizer", self.config),
            "progressive",
            False,
        )
        self._progressive_schedule: ProgressiveSchedule | None = None
        self._progressive_stage_idx: int = -1

        if progressive:
            total_steps = self.config.training.steps
            self._progressive_schedule = ProgressiveSchedule(total_steps)
            self.logger.info(
                "Progressive training enabled with %d stages over %d steps",
                len(self._progressive_schedule.stages),
                total_steps,
            )
            # Apply the first stage immediately
            self._maybe_update_progressive_stage()

    def on_step_end(self, metrics: dict[str, Any]) -> None:
        """Check for progressive stage transitions after each step."""
        if self._progressive_schedule is not None:
            self._maybe_update_progressive_stage()

    def _maybe_update_progressive_stage(self) -> None:
        """Update dataset parameters if the progressive stage has changed.

        Compares the current stage index against the last applied stage.
        If a new stage is active, calls
        :meth:`StylizerDataset.update_progressive_params` to adjust
        segment length and mask ratios.
        """
        assert self._progressive_schedule is not None
        stage = self._progressive_schedule.get_stage(self.global_step)

        # Find the index of the active stage
        stage_idx = 0
        for i, s in enumerate(self._progressive_schedule.stages):
            if self.global_step >= s.start_step:
                stage_idx = i

        if stage_idx == self._progressive_stage_idx:
            return  # No change

        self._progressive_stage_idx = stage_idx
        self.logger.info(
            "Progressive training: transitioning to stage %d at step %d "
            "(segment=%.1fs, mask=[%.2f, %.2f])",
            stage_idx,
            self.global_step,
            stage.segment_length,
            stage.mask_ratio_min,
            stage.mask_ratio_max,
        )

        # Update the dataset parameters
        dataset = getattr(self, "_train_dataset", None)
        if dataset is not None and hasattr(dataset, "update_progressive_params"):
            dataset.update_progressive_params(
                segment_length=stage.segment_length,
                mask_ratio_min=stage.mask_ratio_min,
                mask_ratio_max=stage.mask_ratio_max,
            )


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
