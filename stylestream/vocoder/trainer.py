"""Vocoder trainer: GAN-based training for Causal Vocos.

Extends BaseTrainer with GAN-specific training logic for the Causal Vocos
vocoder. Handles alternating generator/discriminator updates, multiple
optimizers, and combined loss computation.

Training spec (paper):
    - 100k steps, batch 64, 2 GPUs
    - AdamW, peak LR 2e-4, cosine annealing with 1k warmup
    - Losses: mel reconstruction (45.0) + GAN (1.0) + feature matching (2.0)
    - Warm start from official Vocos checkpoint
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stylestream.data.manifest import Manifest
from stylestream.data.vocoder_dataset import build_vocoder_dataloader
from stylestream.training.trainer import BaseTrainer, _cycle
from stylestream.utils.logging import log_metrics
from stylestream.vocoder.discriminator import MultiScaleDiscriminator
from stylestream.vocoder.losses import VocoderLoss
from stylestream.vocoder.model import CausalVocos


class VocoderTrainer(BaseTrainer):
    """GAN trainer for the Causal Vocos vocoder.

    Extends BaseTrainer with GAN-specific logic:
    - Two models: generator (CausalVocos) + discriminator (MultiScaleDiscriminator)
    - Two optimizers with independent learning rates
    - Alternating generator/discriminator update steps
    - Combined losses: mel reconstruction + adversarial + feature matching

    Config structure (OmegaConf)::

        config.training.steps = 100000
        config.training.batch_size = 64
        config.training.peak_lr = 2e-4
        config.training.warmup_steps = 1000
        config.training.gradient_clip = 1.0
        config.training.mixed_precision = "bf16"

        config.model.hidden_size = 512
        config.model.num_layers = 8
        config.model.causal = true
        config.model.init_checkpoint = "charactr/vocos-mel-24khz"

        config.discriminator.scales = [1, 2, 4]
        config.discriminator.channels = 64

        config.loss.reconstruction = 45.0
        config.loss.gan_generator = 1.0
        config.loss.feature_matching = 2.0

        config.data.manifest_path = "data/manifests/libritts.csv"
        config.data.audio_dir = "data/processed/audio"
        config.data.mel_dir = "data/processed/mel"

    Parameters
    ----------
    config :
        OmegaConf ``DictConfig`` with at least ``training``, ``model``,
        ``discriminator``, ``loss``, and ``data`` sub-configs.
    """

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self) -> nn.Module:
        """Build the CausalVocos generator from config.

        Optionally loads warm-start weights from an official Vocos
        checkpoint when ``config.model.init_checkpoint`` is set.

        Returns
        -------
        nn.Module
            CausalVocos generator model.
        """
        generator = CausalVocos.from_config(self.config)

        # Warm start from official Vocos checkpoint
        init_ckpt = getattr(self.config.model, "init_checkpoint", None)
        if init_ckpt:
            warm_info = generator.load_warm_start(init_ckpt)
            self.logger.info(
                "Warm start: loaded %d params, skipped %d params",
                len(warm_info["loaded"]),
                len(warm_info["skipped"]),
            )

        n_params = sum(p.numel() for p in generator.parameters())
        n_trainable = sum(
            p.numel() for p in generator.parameters() if p.requires_grad
        )
        self.logger.info(
            "Generator (CausalVocos) built: %s total params, %s trainable",
            f"{n_params:,}",
            f"{n_trainable:,}",
        )
        return generator

    def build_discriminator(self) -> nn.Module:
        """Build the MultiScaleDiscriminator from config.

        Returns
        -------
        nn.Module
            Multi-scale discriminator.
        """
        discriminator = MultiScaleDiscriminator.from_config(self.config)
        n_params = sum(p.numel() for p in discriminator.parameters())
        self.logger.info(
            "Discriminator (MSD) built: %s params",
            f"{n_params:,}",
        )
        return discriminator

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def build_dataloader(self) -> DataLoader:
        """Build the vocoder training DataLoader from manifest.

        Reads ``config.data.manifest_path``, ``config.data.audio_dir``,
        and optionally ``config.data.mel_dir`` to construct a
        :func:`build_vocoder_dataloader`.

        Returns
        -------
        DataLoader
            Yields dicts with ``mel`` ``(B, 100, 100)`` and
            ``waveform`` ``(B, 32000)``.
        """
        manifest_path = self.config.data.manifest_path
        audio_dir = self.config.data.audio_dir
        mel_dir = getattr(self.config.data, "mel_dir", None)
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)

        manifest = Manifest.load(manifest_path)
        self.logger.info(
            "Training manifest loaded: %d utterances from %s",
            len(manifest),
            manifest_path,
        )

        return build_vocoder_dataloader(
            manifest=manifest,
            audio_dir=audio_dir,
            mel_dir=mel_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
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
            self.logger.info(
                "No val_manifest_path configured, skipping validation."
            )
            return None

        audio_dir = self.config.data.audio_dir
        mel_dir = getattr(self.config.data, "mel_dir", None)
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)

        manifest = Manifest.load(val_path)
        self.logger.info(
            "Validation manifest loaded: %d utterances from %s",
            len(manifest),
            val_path,
        )

        self._val_dataloader = build_vocoder_dataloader(
            manifest=manifest,
            audio_dir=audio_dir,
            mel_dir=mel_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return self._val_dataloader

    # ------------------------------------------------------------------
    # Optimizers (generator + discriminator)
    # ------------------------------------------------------------------

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build optimizer (AdamW or Lion) for the generator.

        Respects ``config.training.optimizer`` to select between AdamW
        and Lion.  Lion uses betas=(0.9, 0.99) by default; AdamW uses
        betas=(0.9, 0.999).

        Parameters
        ----------
        model : nn.Module
            The generator model.

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

    def build_discriminator_optimizer(
        self, discriminator: nn.Module
    ) -> torch.optim.Optimizer:
        """Build a separate AdamW optimizer for the discriminator.

        The discriminator always uses AdamW regardless of the generator's
        optimizer setting.  AdamW provides more stable GAN training for
        the discriminator due to its adaptive second-moment estimates,
        whereas Lion's sign-based updates can cause oscillation in the
        adversarial min-max game.

        Uses the same learning rate as the generator unless overridden
        via ``config.training.discriminator_lr``.

        Parameters
        ----------
        discriminator : nn.Module
            The discriminator model.

        Returns
        -------
        torch.optim.Optimizer
            Configured AdamW optimizer for the discriminator.
        """
        lr = getattr(
            self.config.training, "discriminator_lr",
            self.config.training.peak_lr,
        )
        betas_cfg = getattr(self.config.training, "betas", [0.9, 0.999])
        betas = tuple(betas_cfg) if not isinstance(betas_cfg, tuple) else betas_cfg
        weight_decay = getattr(self.config.training, "weight_decay", 0.01)

        optimizer_name = getattr(self.config.training, "optimizer", "adamw").lower()
        if optimizer_name == "lion":
            self.logger.info(
                "Generator uses Lion, but discriminator uses AdamW "
                "for GAN training stability."
            )

        return torch.optim.AdamW(
            discriminator.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Loss (BaseTrainer compatibility)
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """Compute generator-only loss for a single batch.

        This method is provided for BaseTrainer interface compatibility
        (e.g. validation). The actual GAN training loop uses the full
        alternating G/D update in :meth:`train`.

        Parameters
        ----------
        batch : dict
            Dict with ``mel`` ``(B, 100, 100)`` and
            ``waveform`` ``(B, 32000)``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``loss`` : scalar mel reconstruction loss.
        """
        mel = batch["mel"]            # (B, 100, 100)
        waveform = batch["waveform"]  # (B, 32000)

        fake_waveform = self.model(mel)

        # Align lengths
        min_len = min(fake_waveform.shape[-1], waveform.shape[-1])
        fake_waveform = fake_waveform[..., :min_len]
        real_waveform = waveform[..., :min_len]

        # Compute only reconstruction loss for validation
        loss_fn = self._get_loss_fn()
        g_loss_dict = loss_fn.generator_loss(
            fake_waveform, real_waveform,
            disc_fake_outputs=None,
            disc_real_features=None,
            disc_fake_features=None,
        )

        return {"loss": g_loss_dict["loss"]}

    # ------------------------------------------------------------------
    # Main GAN training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: str | Path | None = None) -> None:
        """Execute the full GAN training loop with alternating G/D updates.

        Overrides :meth:`BaseTrainer.train` to handle the two-model,
        two-optimizer GAN training procedure.

        Parameters
        ----------
        resume_from :
            Path to a checkpoint directory. If provided, generator,
            discriminator, both optimizers, both schedulers, and
            ``global_step`` are restored before training resumes.
        """
        total_steps: int = self.config.training.steps
        log_interval: int = getattr(self.config.training, "log_interval", 100)
        save_interval: int = getattr(self.config.training, "save_interval", 10000)
        val_interval: int = getattr(self.config.training, "val_interval", 2000)
        grad_clip: float = getattr(self.config.training, "gradient_clip", 1.0)

        # 1. Build components -------------------------------------------
        generator = self.build_model()
        discriminator = self.build_discriminator()
        dataloader = self.build_dataloader()
        loss_fn = self._build_loss_fn()

        g_optimizer = self.build_optimizer(generator)
        d_optimizer = self.build_discriminator_optimizer(discriminator)
        g_scheduler = self.build_scheduler(g_optimizer)
        d_scheduler = self.build_scheduler(d_optimizer)

        # 2. Prepare with Accelerate ------------------------------------
        (
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            dataloader,
            g_scheduler,
            d_scheduler,
        ) = self.accelerator.prepare(
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            dataloader,
            g_scheduler,
            d_scheduler,
        )

        # Store references for checkpointing and validation
        self.model = generator
        self.discriminator = discriminator
        self.optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.scheduler = g_scheduler
        self.d_scheduler = d_scheduler
        self._loss_fn = loss_fn

        # 2b. Optional torch.compile ------------------------------------
        # Compile generator and discriminator separately.  We use
        # mode="default" rather than "reduce-overhead" because the
        # latter relies on CUDA graphs, which are incompatible with
        # the alternating gradient-enable/disable pattern of GAN
        # training (D frozen during G step and vice versa).
        if getattr(self.config.training, "compile_model", False) and hasattr(
            torch, "compile"
        ):
            self.logger.info(
                "Compiling generator and discriminator with "
                "torch.compile(mode='default')"
            )
            self.model = torch.compile(self.model, mode="default")
            self.discriminator = torch.compile(
                self.discriminator, mode="default"
            )

        # 3. Optionally resume from checkpoint --------------------------
        if resume_from is not None:
            self.load_checkpoint(Path(resume_from))

        # 4. Hooks -------------------------------------------------------
        self.on_train_start()
        self.logger.info(
            "Starting GAN training from step %d / %d",
            self.global_step,
            total_steps,
        )

        # 5. GAN training loop ------------------------------------------
        # Use self.model / self.discriminator which may have been wrapped
        # by torch.compile in step 2b above.
        generator = self.model
        discriminator = self.discriminator
        generator.train()
        discriminator.train()
        data_iter = _cycle(dataloader)
        step_t0 = time.monotonic()

        while self.global_step < total_steps:
            batch = next(data_iter)
            mel = batch["mel"]            # (B, 100, 100)
            waveform = batch["waveform"]  # (B, 32000)

            # --- Generator forward -------------------------------------
            fake_waveform = generator(mel)

            # Trim to match lengths (ISTFT may produce slightly different
            # length than the target waveform)
            min_len = min(fake_waveform.shape[-1], waveform.shape[-1])
            fake_waveform = fake_waveform[..., :min_len]
            real_waveform = waveform[..., :min_len]

            # --- Discriminator step ------------------------------------
            # Forward discriminator on real and fake (fake detached for D)
            real_logits, real_features = discriminator(
                real_waveform.unsqueeze(1)
            )
            fake_logits_d, _ = discriminator(
                fake_waveform.detach().unsqueeze(1)
            )

            d_loss_dict = loss_fn.discriminator_loss(
                real_logits, fake_logits_d
            )
            d_loss = d_loss_dict["loss"]

            d_optimizer.zero_grad()
            self.accelerator.backward(d_loss)
            if grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    discriminator.parameters(), grad_clip
                )
            d_optimizer.step()
            d_scheduler.step()

            # Cache detached real features immediately after D backward.
            # This frees the discriminator's computation graph for real
            # audio while keeping the features for G-step feature matching.
            cached_real_features = [
                [f.detach() for f in scale_features]
                for scale_features in real_features
            ]
            del real_features  # free D-step graph references

            # --- Generator step ----------------------------------------
            # Forward discriminator on fake (without detach for G grads)
            fake_logits_g, fake_features = discriminator(
                fake_waveform.unsqueeze(1)
            )

            g_loss_dict = loss_fn.generator_loss(
                fake_waveform,
                real_waveform,
                fake_logits_g,
                cached_real_features,
                fake_features,
            )
            g_loss = g_loss_dict["loss"]

            g_optimizer.zero_grad()
            self.accelerator.backward(g_loss)
            if grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    generator.parameters(), grad_clip
                )
            g_optimizer.step()
            g_scheduler.step()

            self.global_step += 1

            # --- Logging -----------------------------------------------
            if self.global_step % log_interval == 0:
                elapsed = time.monotonic() - step_t0
                steps_per_sec = log_interval / elapsed if elapsed > 0 else 0.0

                metrics: dict[str, Any] = {
                    "g_loss": g_loss.detach().item(),
                    "d_loss": d_loss.detach().item(),
                    "lr": g_scheduler.get_last_lr()[0],
                    "steps_per_sec": steps_per_sec,
                }
                # Merge auxiliary metrics from both loss dicts
                for prefix, loss_dict in [("g", g_loss_dict), ("d", d_loss_dict)]:
                    for k, v in loss_dict.items():
                        if k != "loss" and isinstance(v, torch.Tensor):
                            metrics[f"{prefix}_{k}"] = v.detach().item()

                log_metrics(
                    self.logger, self.global_step, metrics, prefix="train"
                )
                self._log_wandb(metrics, step=self.global_step)
                self.on_step_end(metrics)
                step_t0 = time.monotonic()

            # --- Validation --------------------------------------------
            if val_interval > 0 and self.global_step % val_interval == 0:
                generator.eval()
                discriminator.eval()
                val_metrics = self.validate()
                generator.train()
                discriminator.train()

                if val_metrics:
                    log_metrics(
                        self.logger,
                        self.global_step,
                        val_metrics,
                        prefix="val",
                    )
                    self._log_wandb(
                        {f"val/{k}": v for k, v in val_metrics.items()},
                        step=self.global_step,
                    )
                    val_loss = val_metrics.get("loss", float("inf"))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save("best")

            # --- Checkpointing -----------------------------------------
            if save_interval > 0 and self.global_step % save_interval == 0:
                self._save(f"step_{self.global_step}")

        # Final save
        self._save("final")
        self.logger.info("GAN training complete at step %d.", self.global_step)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, float]:
        """Run validation and return average generator loss.

        Iterates over the validation dataloader and computes the
        mel reconstruction loss (no adversarial / feature matching).

        Returns
        -------
        dict[str, float]
            ``loss`` -- average reconstruction loss.
            Empty dict if no validation dataloader is available.
        """
        val_dl = getattr(self, "_val_dataloader", None)
        if val_dl is None:
            return {}

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_dl:
                batch = _move_batch(batch, self.accelerator.device)
                loss_dict = self.compute_loss(batch)
                total_loss += loss_dict["loss"].item()
                n_batches += 1

        if n_batches == 0:
            return {}

        return {"loss": total_loss / n_batches}

    # ------------------------------------------------------------------
    # Checkpoint save / load (handles both G and D)
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save generator, discriminator, both optimizers, and training state.

        Parameters
        ----------
        path : str or Path
            Directory in which to save all checkpoint files.
        """
        path = Path(path)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            path.mkdir(parents=True, exist_ok=True)

            unwrapped_g = self.accelerator.unwrap_model(self.model)
            unwrapped_d = self.accelerator.unwrap_model(self.discriminator)

            torch.save(
                {
                    "generator": unwrapped_g.state_dict(),
                    "discriminator": unwrapped_d.state_dict(),
                    "g_optimizer": self.optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "g_scheduler": self.scheduler.state_dict(),
                    "d_scheduler": self.d_scheduler.state_dict(),
                    "global_step": self.global_step,
                    "best_val_loss": self.best_val_loss,
                },
                path / "trainer_state.pt",
            )
            self.logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore generator, discriminator, optimizers, and training state.

        Parameters
        ----------
        path : str or Path
            Directory containing a ``trainer_state.pt`` file.
        """
        path = Path(path)
        if not path.exists():
            self.logger.warning(
                "Checkpoint path %s does not exist, skipping.", path
            )
            return

        state_file = path / "trainer_state.pt"
        if not state_file.exists():
            self.logger.warning(
                "No trainer_state.pt in %s, skipping.", path
            )
            return

        state = torch.load(
            state_file,
            map_location=self.accelerator.device,
            weights_only=False,
        )

        # Restore generator
        unwrapped_g = self.accelerator.unwrap_model(self.model)
        unwrapped_g.load_state_dict(state["generator"])

        # Restore discriminator
        unwrapped_d = self.accelerator.unwrap_model(self.discriminator)
        unwrapped_d.load_state_dict(state["discriminator"])

        # Restore optimizers
        self.optimizer.load_state_dict(state["g_optimizer"])
        self.d_optimizer.load_state_dict(state["d_optimizer"])

        # Restore schedulers
        self.scheduler.load_state_dict(state["g_scheduler"])
        self.d_scheduler.load_state_dict(state["d_scheduler"])

        # Restore training state
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))

        self.logger.info(
            "Resumed from %s at step %d (best_val_loss=%.4f)",
            path,
            self.global_step,
            self.best_val_loss,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Log model summary and config at the start of training."""
        self.logger.info("Config: %s", _config_summary(self.config))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_loss_fn(self) -> VocoderLoss:
        """Build the VocoderLoss module from config.

        Returns
        -------
        VocoderLoss
            Combined loss module for generator and discriminator.
        """
        loss_cfg = self.config.loss
        mel_cfg = getattr(self.config, "mel", None)

        n_mels = getattr(mel_cfg, "n_mels", 100) if mel_cfg else 100
        hop_length = getattr(mel_cfg, "hop_length", 320) if mel_cfg else 320
        sample_rate = getattr(mel_cfg, "sample_rate", 16000) if mel_cfg else 16000

        return VocoderLoss(
            reconstruction_weight=getattr(loss_cfg, "reconstruction", 45.0),
            gan_generator_weight=getattr(loss_cfg, "gan_generator", 1.0),
            gan_discriminator_weight=getattr(
                loss_cfg, "gan_discriminator", 1.0
            ),
            feature_matching_weight=getattr(
                loss_cfg, "feature_matching", 2.0
            ),
            n_mels=n_mels,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )

    def _get_loss_fn(self) -> VocoderLoss:
        """Return the cached loss function, building it if needed.

        Returns
        -------
        VocoderLoss
            The loss module.
        """
        if not hasattr(self, "_loss_fn") or self._loss_fn is None:
            self._loss_fn = self._build_loss_fn()
        return self._loss_fn


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
