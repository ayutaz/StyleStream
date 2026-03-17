"""Base trainer for all StyleStream components.

Step-based training (not epoch-based) following the paper's methodology.
Uses HuggingFace Accelerate for distributed training, mixed precision,
and gradient accumulation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

from stylestream.utils.logging import log_metrics, setup_logger


# ======================================================================
# Progressive training schedule
# ======================================================================


@dataclass
class ProgressiveStage:
    """A stage in progressive training.

    Parameters
    ----------
    start_step :
        Global step at which this stage becomes active.
    segment_length :
        Training segment length in seconds.
    mask_ratio_min :
        Lower bound for uniform mask ratio sampling.
    mask_ratio_max :
        Upper bound for uniform mask ratio sampling.
    """

    start_step: int
    segment_length: float  # seconds
    mask_ratio_min: float
    mask_ratio_max: float


class ProgressiveSchedule:
    """Gradually increases training difficulty over the course of training.

    The schedule is defined as a list of :class:`ProgressiveStage` instances,
    each specifying the segment length and mask ratio range to use from a
    given step onward.  The active stage is the last stage whose
    ``start_step`` is <= the current global step.

    When ``stages`` is not provided, a default 3-stage schedule is used:

    * **Stage 1** (0 -- 25 % of training): 3 s segments, mask ratio 0.5--0.7
    * **Stage 2** (25 % -- 50 %): 4.5 s segments, mask ratio 0.6--0.9
    * **Stage 3** (50 % -- end): 6 s segments, mask ratio 0.7--1.0

    Parameters
    ----------
    total_steps :
        Total number of training steps (used to compute default stage
        boundaries).
    stages :
        Explicit list of stages.  If ``None``, the default 3-stage schedule
        is constructed from *total_steps*.
    """

    def __init__(
        self,
        total_steps: int,
        stages: list[ProgressiveStage] | None = None,
    ) -> None:
        if stages is None:
            s1 = int(total_steps * 0.25)
            s2 = int(total_steps * 0.5)
            self.stages = [
                ProgressiveStage(0, 3.0, 0.5, 0.7),     # Easy: 3s, low mask
                ProgressiveStage(s1, 4.5, 0.6, 0.9),    # Medium: 4.5s, med mask
                ProgressiveStage(s2, 6.0, 0.7, 1.0),    # Hard: 6s, full mask
            ]
        else:
            self.stages = stages

    def get_stage(self, current_step: int) -> ProgressiveStage:
        """Return the active stage for *current_step*.

        The active stage is the last one whose ``start_step`` is
        <= *current_step*.

        Parameters
        ----------
        current_step :
            The current global training step.

        Returns
        -------
        ProgressiveStage
            The stage that should be applied at this step.
        """
        active = self.stages[0]
        for stage in self.stages:
            if current_step >= stage.start_step:
                active = stage
        return active


def _cycle(dataloader: DataLoader):
    """Yield batches from *dataloader* forever, cycling through epochs."""
    while True:
        yield from dataloader


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float = float("inf")
        self.should_stop = False

    def check(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class BaseTrainer(ABC):
    """Abstract base trainer for StyleStream component training.

    All StyleStream components (Destylizer, Stylizer, Vocoder) share the same
    high-level training loop: step-based iteration with cosine-annealing LR,
    gradient clipping, periodic checkpointing, and optional W&B logging.

    Subclasses must implement:
        - :meth:`build_model`
        - :meth:`build_dataloader`
        - :meth:`compute_loss`

    Parameters
    ----------
    config :
        OmegaConf ``DictConfig`` (or any dot-accessible namespace) with at
        least a ``training`` sub-config containing the fields listed in
        :class:`stylestream.config.TrainingConfig`.
    """

    def __init__(self, config) -> None:
        self.config = config

        # --- CUDA optimizations --------------------------------------------
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        # --- Accelerator ---------------------------------------------------
        mixed_precision = getattr(config.training, "mixed_precision", "bf16")
        grad_accum = getattr(config.training, "gradient_accumulation_steps", 1)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum,
        )

        # --- Logger --------------------------------------------------------
        log_dir = getattr(config, "log_dir", None)
        self.logger = setup_logger(
            name=self.__class__.__name__,
            level=getattr(config, "log_level", "INFO"),
            log_dir=log_dir,
            rank=self.accelerator.process_index,
        )

        # --- Tracking state ------------------------------------------------
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")

        # --- Early stopping ------------------------------------------------
        if getattr(config.training, "early_stopping", False):
            patience = getattr(config.training, "early_stopping_patience", 10)
            min_delta = getattr(config.training, "early_stopping_min_delta", 1e-4)
            self.early_stopping: EarlyStopping | None = EarlyStopping(
                patience=patience, min_delta=min_delta,
            )
            self.logger.info(
                "Early stopping enabled (patience=%d, min_delta=%.1e)",
                patience, min_delta,
            )
        else:
            self.early_stopping = None

        # --- W&B -----------------------------------------------------------
        self._wandb_run = None
        if self.accelerator.is_main_process:
            self._init_wandb()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model to be trained."""

    @abstractmethod
    def build_dataloader(self) -> DataLoader:
        """Build and return the training dataloader."""

    @abstractmethod
    def compute_loss(self, batch) -> dict[str, torch.Tensor]:
        """Compute loss for a single batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Must contain a ``"loss"`` key with the scalar loss to back-prop.
            Additional keys are logged as auxiliary metrics.
        """

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def build_val_dataloader(self) -> DataLoader | None:
        """Optionally build a validation dataloader. Return *None* to skip."""
        return None

    def validate(self) -> dict[str, float]:
        """Run validation and return a metrics dict.

        Called every ``config.training.val_interval`` steps.  The default
        implementation is a no-op.  Subclasses should override this to
        evaluate on a held-out set.
        """
        return {}

    def on_train_start(self) -> None:
        """Hook called once before the training loop begins."""

    def on_step_end(self, metrics: dict[str, Any]) -> None:
        """Hook called after every optimiser step with the logged metrics."""

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build the AdamW optimiser (paper default)."""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.peak_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

    def build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build a cosine-annealing scheduler with linear warmup."""
        from stylestream.training.scheduler import CosineAnnealingWarmup

        return CosineAnnealingWarmup(
            optimizer,
            warmup_steps=self.config.training.warmup_steps,
            total_steps=self.config.training.steps,
            peak_lr=self.config.training.peak_lr,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: str | Path | None = None) -> None:
        """Execute the full step-based training loop.

        Parameters
        ----------
        resume_from :
            Path to a checkpoint directory.  If provided, model / optimiser /
            scheduler state and ``global_step`` are restored before training
            resumes.
        """
        total_steps: int = self.config.training.steps
        log_interval: int = getattr(self.config.training, "log_interval", 100)
        save_interval: int = getattr(self.config.training, "save_interval", 5000)
        val_interval: int = getattr(self.config.training, "val_interval", 1000)
        grad_clip: float = getattr(self.config.training, "gradient_clip", 1.0)

        # 1. Build components -----------------------------------------------
        model = self.build_model()
        dataloader = self.build_dataloader()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)

        # 2. Prepare with Accelerate ----------------------------------------
        model, optimizer, dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # 2b. Optional torch.compile ----------------------------------------
        if getattr(self.config.training, "compile_model", False) and hasattr(torch, "compile"):
            self.logger.info("Compiling model with torch.compile(mode='reduce-overhead')")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # 3. Optionally resume from checkpoint ------------------------------
        if resume_from is not None:
            self.load_checkpoint(Path(resume_from))

        # 4. Hooks -----------------------------------------------------------
        self.on_train_start()
        self.logger.info(
            "Starting training from step %d / %d", self.global_step, total_steps
        )

        # 5. Step-based loop -------------------------------------------------
        model.train()
        data_iter = _cycle(dataloader)
        step_t0 = time.monotonic()

        while self.global_step < total_steps:
            batch = next(data_iter)

            with self.accelerator.accumulate(model):
                loss_dict = self.compute_loss(batch)
                loss = loss_dict["loss"]
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    if grad_clip > 0:
                        self.accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Only count a step when gradients have been synchronised (i.e.
            # after all gradient-accumulation micro-steps).
            if not self.accelerator.sync_gradients:
                continue

            self.global_step += 1

            # --- Logging ---------------------------------------------------
            if self.global_step % log_interval == 0:
                elapsed = time.monotonic() - step_t0
                steps_per_sec = log_interval / elapsed if elapsed > 0 else 0.0

                metrics: dict[str, Any] = {
                    "loss": loss.detach().item(),
                    "lr": scheduler.get_last_lr()[0],
                    "steps_per_sec": steps_per_sec,
                }
                # Merge auxiliary metrics from compute_loss
                for k, v in loss_dict.items():
                    if k != "loss" and isinstance(v, torch.Tensor):
                        metrics[k] = v.detach().item()

                log_metrics(self.logger, self.global_step, metrics, prefix="train")
                self._log_wandb(metrics, step=self.global_step)
                self.on_step_end(metrics)
                step_t0 = time.monotonic()

            # --- Validation ------------------------------------------------
            if val_interval > 0 and self.global_step % val_interval == 0:
                model.eval()
                val_metrics = self.validate()
                model.train()

                if val_metrics:
                    log_metrics(
                        self.logger, self.global_step, val_metrics, prefix="val"
                    )
                    self._log_wandb(
                        {f"val/{k}": v for k, v in val_metrics.items()},
                        step=self.global_step,
                    )
                    val_loss = val_metrics.get("loss", float("inf"))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save("best")

                    # Early stopping check
                    if (
                        self.early_stopping is not None
                        and self.early_stopping.check(val_loss)
                    ):
                        self.logger.info(
                            "Early stopping triggered at step %d "
                            "(no improvement for %d validations, best_val_loss=%.4f)",
                            self.global_step,
                            self.early_stopping.patience,
                            self.early_stopping.best_loss,
                        )
                        break

            # --- Checkpointing ---------------------------------------------
            if save_interval > 0 and self.global_step % save_interval == 0:
                self._save(f"step_{self.global_step}")

        # Final save
        self._save("final")
        self.logger.info("Training complete at step %d.", self.global_step)

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _get_checkpoint_dir(self) -> Path:
        """Return the base directory for checkpoints."""
        base = getattr(self.config, "output_dir", "outputs")
        name = getattr(self.config, "name", "experiment")
        return Path(base) / name / "checkpoints"

    def _save(self, tag: str) -> None:
        """Save a checkpoint with the given *tag* (e.g. ``step_10000``)."""
        ckpt_dir = self._get_checkpoint_dir() / tag
        self.save_checkpoint(ckpt_dir)

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist model, optimiser, scheduler, and trainer state.

        Uses :class:`stylestream.utils.checkpoint.CheckpointManager` when
        available; otherwise falls back to Accelerate's built-in saver.
        """
        path = Path(path)
        self.accelerator.wait_for_everyone()

        try:
            from stylestream.utils.checkpoint import CheckpointManager

            mgr = CheckpointManager(path)
            mgr.save(
                accelerator=self.accelerator,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                global_step=self.global_step,
                best_val_loss=self.best_val_loss,
                config=self.config,
            )
        except ImportError:
            # Fallback: use Accelerate's native save
            if self.accelerator.is_main_process:
                path.mkdir(parents=True, exist_ok=True)
                unwrapped = self.accelerator.unwrap_model(self.model)
                torch.save(
                    {
                        "model": unwrapped.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "global_step": self.global_step,
                        "best_val_loss": self.best_val_loss,
                    },
                    path / "trainer_state.pt",
                )
                self.logger.info("Checkpoint saved to %s (fallback)", path)

        if self.accelerator.is_main_process:
            self.logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore training state from a checkpoint directory."""
        path = Path(path)
        if not path.exists():
            self.logger.warning("Checkpoint path %s does not exist, skipping.", path)
            return

        try:
            from stylestream.utils.checkpoint import CheckpointManager

            mgr = CheckpointManager(path)
            state = mgr.load(
                accelerator=self.accelerator,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            self.global_step = state.get("global_step", 0)
            self.best_val_loss = state.get("best_val_loss", float("inf"))
        except ImportError:
            # Fallback: plain torch load
            state_file = path / "trainer_state.pt"
            if not state_file.exists():
                self.logger.warning("No trainer_state.pt in %s, skipping.", path)
                return
            state = torch.load(
                state_file,
                map_location=self.accelerator.device,
                weights_only=False,
            )
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state.get("global_step", 0)
            self.best_val_loss = state.get("best_val_loss", float("inf"))

        self.logger.info(
            "Resumed from %s at step %d (best_val_loss=%.4f)",
            path,
            self.global_step,
            self.best_val_loss,
        )

    # ------------------------------------------------------------------
    # W&B integration
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialise a W&B run if the library is installed and configured."""
        wandb_cfg = getattr(self.config, "wandb", None)
        if wandb_cfg is None:
            return
        enabled = getattr(wandb_cfg, "enabled", False)
        if not enabled:
            return
        try:
            import wandb

            project = getattr(wandb_cfg, "project", "stylestream")
            name = getattr(self.config, "name", None)
            self._wandb_run = wandb.init(
                project=project,
                name=name,
                config=_config_to_dict(self.config),
                resume="allow",
            )
            self.logger.info("W&B run initialised: %s", self._wandb_run.url)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to initialise W&B: %s", exc)

    def _log_wandb(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to W&B if a run is active."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Clean up resources (e.g. close W&B run)."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None


# ======================================================================
# Helpers
# ======================================================================

def _config_to_dict(cfg) -> dict:
    """Best-effort conversion of an OmegaConf DictConfig to a plain dict."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return {"config": str(cfg)}
