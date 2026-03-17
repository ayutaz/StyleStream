"""Destylizer trainer: ASR-supervised content feature extraction.

Extends :class:`BaseTrainer` to train the Destylizer pipeline
(Conformer + FSQ + ASR decoder) on pre-extracted HuBERT layer-18
features.  Training uses CTC or sequence-to-sequence cross-entropy
loss on character-level targets to ensure content preservation.

Training spec (paper Section 10.8):
    - 100k steps, batch 32, AdamW, cosine annealing with 4k warmup
    - Peak LR 1e-4, gradient clip 1.0, bf16 mixed precision
    - Dataset: LMG (~1,300 hours)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stylestream.data.destylizer_dataset import build_destylizer_dataloader
from stylestream.data.manifest import Manifest
from stylestream.data.text import CharTokenizer
from stylestream.destylizer.model import Destylizer
from stylestream.training.trainer import BaseTrainer


class DestylizerTrainer(BaseTrainer):
    """Trainer for the StyleStream Destylizer component.

    Extends :class:`BaseTrainer` with Destylizer-specific model
    construction, data loading, loss computation, and validation.

    The trainer expects the following config structure (OmegaConf)::

        config.training.steps = 100000
        config.training.batch_size = 32
        config.training.peak_lr = 1e-4
        config.training.warmup_steps = 4000
        config.training.gradient_clip = 1.0
        config.training.mixed_precision = "bf16"
        config.training.log_interval = 100
        config.training.save_interval = 10000
        config.training.val_interval = 5000

        config.data.manifest_path = "data/manifests/lmg.csv"
        config.data.features_dir = "data/processed"
        config.data.val_manifest_path = None  # optional

        # Destylizer sub-configs: conformer, fsq, asr_decoder
        config.conformer.num_layers = 6
        config.fsq.levels = [5, 3, 3]
        config.asr_decoder.vocab_size = 30

    Parameters
    ----------
    config :
        OmegaConf ``DictConfig`` with at least ``training``, ``data``,
        ``conformer``, ``fsq``, and ``asr_decoder`` sub-configs.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.tokenizer = CharTokenizer()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self) -> nn.Module:
        """Build the Destylizer model from config.

        Returns
        -------
        nn.Module
            Destylizer model (Conformer + FSQ + ASR decoder).
        """
        model = Destylizer(config=self.config)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            "Destylizer model built: %s total params, %s trainable",
            f"{n_params:,}",
            f"{n_trainable:,}",
        )
        return model

    # ------------------------------------------------------------------
    # Optimizer (override for paper-specific betas)
    # ------------------------------------------------------------------

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build optimizer (AdamW or Lion) with paper-specified hyperparameters."""
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
            betas_cfg = getattr(self.config.training, "betas", [0.9, 0.98])
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

        Reads ``config.data.manifest_path`` and ``config.data.features_dir``
        to construct a :func:`build_destylizer_dataloader`.

        Returns
        -------
        DataLoader
            Yields dicts with ``hubert_features``, ``token_ids``,
            ``feature_lengths``, ``token_lengths``, ``feature_padding_mask``.
        """
        manifest_path = self.config.data.manifest_path
        features_dir = self.config.data.features_dir
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)
        max_frames = getattr(self.config.data, "max_frames", 3000)

        manifest = Manifest.load(manifest_path)
        self.logger.info(
            "Training manifest loaded: %d utterances from %s",
            len(manifest),
            manifest_path,
        )

        return build_destylizer_dataloader(
            manifest=manifest,
            features_dir=features_dir,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            max_frames=max_frames,
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

        features_dir = self.config.data.features_dir
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)
        max_frames = getattr(self.config.data, "max_frames", 3000)

        manifest = Manifest.load(val_path)
        self.logger.info(
            "Validation manifest loaded: %d utterances from %s",
            len(manifest),
            val_path,
        )

        self._val_dataloader = build_destylizer_dataloader(
            manifest=manifest,
            features_dir=features_dir,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            max_frames=max_frames,
        )
        return self._val_dataloader

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """Compute the Destylizer training loss for a single batch.

        Steps:
            1. Transpose HuBERT features from ``(B, 768, T)`` to ``(B, T, 768)``.
            2. Forward through the Destylizer model.
            3. Compute ASR loss via ASR head.
            4. Return loss dict with auxiliary FSQ metrics.

        Parameters
        ----------
        batch : dict
            From :class:`DestylizerCollator` with keys ``hubert_features``,
            ``token_ids``, ``feature_lengths``, ``token_lengths``,
            ``feature_padding_mask``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``loss`` : scalar ASR loss.
            ``codebook_usage`` : FSQ codebook utilization (0--1).
            ``perplexity`` : exp(entropy) of codebook distribution.
        """
        # Unpack batch
        hubert_features = batch["hubert_features"]    # (B, 768, T)
        token_ids = batch["token_ids"]                # (B, S)
        feature_lengths = batch["feature_lengths"]    # (B,)
        token_lengths = batch["token_lengths"]        # (B,)
        padding_mask = batch["feature_padding_mask"]  # (B, T)

        # Transpose to (B, T, 768) as expected by the Conformer
        hubert_features = hubert_features.transpose(1, 2)

        # Forward through the Destylizer
        model_out = self.model(
            hubert_features=hubert_features,
            padding_mask=padding_mask,
            target_ids=token_ids,
        )

        logits = model_out["logits"]       # (B, T, vocab_size)
        fsq_info = model_out["fsq_info"]   # dict with codebook diagnostics

        # Compute ASR loss
        unwrapped = self.accelerator.unwrap_model(self.model)
        asr_loss = unwrapped.asr_head.compute_loss(
            logits=logits,
            targets=token_ids,
            encoder_lengths=feature_lengths,
            target_lengths=token_lengths,
        )

        # Pack loss dict with auxiliary metrics
        return {
            "loss": asr_loss,
            "codebook_usage": torch.tensor(fsq_info["codebook_usage"]),
            "perplexity": torch.tensor(fsq_info["perplexity"]),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, float]:
        """Run validation and return average loss and FSQ metrics.

        Iterates over the full validation dataloader (built by
        :meth:`build_val_dataloader`) and aggregates metrics.

        Returns
        -------
        dict[str, float]
            ``loss``, ``codebook_usage``, ``perplexity``.
            Empty dict if no validation dataloader is available.
        """
        val_dl = getattr(self, "_val_dataloader", None)
        if val_dl is None:
            return {}

        total_loss = 0.0
        total_usage = 0.0
        total_perplexity = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_dl:
                # Move batch to device (Accelerate handles this for the
                # training dataloader, but val_dl may not be prepared)
                batch = _move_batch(batch, self.accelerator.device)

                loss_dict = self.compute_loss(batch)
                total_loss += loss_dict["loss"].item()
                total_usage += loss_dict["codebook_usage"].item()
                total_perplexity += loss_dict["perplexity"].item()
                n_batches += 1

        if n_batches == 0:
            return {}

        return {
            "loss": total_loss / n_batches,
            "codebook_usage": total_usage / n_batches,
            "perplexity": total_perplexity / n_batches,
        }

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Log model summary and config at the start of training."""
        self.logger.info("Config: %s", _config_summary(self.config))
        self.logger.info(
            "Tokenizer: vocab_size=%d, blank_id=%d",
            self.tokenizer.vocab_size,
            self.tokenizer.blank_id,
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
