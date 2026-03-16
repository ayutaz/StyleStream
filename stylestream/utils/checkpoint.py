"""Checkpoint management for StyleStream training.

Handles saving and loading of model weights (safetensors), optimizer state,
scheduler state, and training metadata.  Supports automatic cleanup of old
checkpoints and maintains a separate ``best/`` checkpoint.

Directory structure::

    {output_dir}/{experiment_name}/{timestamp}/
        config.yaml
        checkpoints/
            step_10000/
                model.safetensors
                optimizer.pt
                scheduler.pt
                training_state.pt
            step_20000/
                ...
            best/
                ...
        logs/
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# Filenames used inside each checkpoint directory
_MODEL_FILE = "model.safetensors"
_OPTIMIZER_FILE = "optimizer.pt"
_SCHEDULER_FILE = "scheduler.pt"
_TRAINING_STATE_FILE = "training_state.pt"

# Regex to extract the step number from a checkpoint directory name
_STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def _unwrap_model(model: Any) -> torch.nn.Module:
    """Unwrap an accelerator-wrapped model to get the raw ``nn.Module``.

    Supports ``accelerate.utils.extract_model_from_parallel`` when available,
    and also handles ``torch.nn.DataParallel`` / ``DistributedDataParallel``
    directly.
    """
    # accelerate wrapper
    try:
        from accelerate.utils import extract_model_from_parallel

        return extract_model_from_parallel(model)
    except ImportError:
        pass

    # PyTorch DDP / DP
    if hasattr(model, "module"):
        return model.module

    return model


def _timestamp() -> str:
    """Return a filesystem-safe timestamp string (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Parameters
    ----------
    output_dir:
        Base output directory (e.g. ``"outputs"``).
    experiment_name:
        Human-readable experiment name.  Used as a sub-directory under
        *output_dir*.
    max_keep:
        Maximum number of **step** checkpoints to retain.  The ``best/``
        checkpoint is always kept separately and does not count toward this
        limit.  Oldest checkpoints are deleted first.
    """

    def __init__(
        self,
        output_dir: str | Path,
        experiment_name: str,
        max_keep: int = 5,
    ) -> None:
        self.max_keep = max_keep
        self.experiment_name = experiment_name

        # Build experiment directory with a timestamp
        timestamp = _timestamp()
        self.experiment_dir = Path(output_dir) / experiment_name / timestamp
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"

        # Create directory tree
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Checkpoint directory: %s", self.checkpoint_dir)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        extra_state: dict | None = None,
        is_best: bool = False,
    ) -> Path:
        """Save a full training checkpoint.

        Parameters
        ----------
        step:
            Current global training step.
        model:
            The model (may be accelerator-wrapped).
        optimizer:
            The optimizer.
        scheduler:
            The learning rate scheduler (must support ``.state_dict()``).
        extra_state:
            Optional dict of additional state (e.g. best_val_loss, epoch).
        is_best:
            If *True*, also save a copy under ``checkpoints/best/``.

        Returns
        -------
        Path
            Path to the saved checkpoint directory.
        """
        step_dir = self.checkpoint_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # 1. Model weights -> safetensors
        raw_model = _unwrap_model(model)
        state_dict = raw_model.state_dict()
        # safetensors requires all tensors to be contiguous and on CPU
        cpu_state = {k: v.contiguous().cpu() for k, v in state_dict.items()}
        save_file(cpu_state, step_dir / _MODEL_FILE)

        # 2. Optimizer state -> torch.save (contains non-tensor objects)
        torch.save(optimizer.state_dict(), step_dir / _OPTIMIZER_FILE)

        # 3. Scheduler state -> torch.save
        torch.save(scheduler.state_dict(), step_dir / _SCHEDULER_FILE)

        # 4. Training state (global step, metrics, etc.)
        training_state: dict[str, Any] = {"global_step": step}
        if extra_state is not None:
            training_state.update(extra_state)
        torch.save(training_state, step_dir / _TRAINING_STATE_FILE)

        logger.info("Saved checkpoint at step %d -> %s", step, step_dir)

        # 5. Optionally copy to best/
        if is_best:
            best_dir = self.checkpoint_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(step_dir, best_dir)
            logger.info("Updated best checkpoint -> %s", best_dir)

        # 6. Cleanup old step checkpoints
        self._cleanup_old()

        return step_dir

    def save_config(self, config: Any) -> Path:
        """Save an OmegaConf or dict config as ``config.yaml`` in the experiment dir.

        Parameters
        ----------
        config:
            An OmegaConf ``DictConfig`` / ``ListConfig``, a dataclass, or a
            plain ``dict``.

        Returns
        -------
        Path
            Path to the saved config file.
        """
        config_path = self.experiment_dir / "config.yaml"

        try:
            from omegaconf import OmegaConf

            # If it's already an OmegaConf container, save directly.
            # Otherwise, convert from a structured config / dict first.
            if not OmegaConf.is_config(config):
                config = OmegaConf.structured(config)
            OmegaConf.save(config, config_path)
        except ImportError:
            # Fallback: dump with PyYAML if omegaconf is unavailable.
            import yaml

            if hasattr(config, "__dataclass_fields__"):
                from dataclasses import asdict

                config = asdict(config)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(dict(config), f, default_flow_style=False, sort_keys=False)

        logger.info("Saved config -> %s", config_path)
        return config_path

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load a checkpoint.

        Parameters
        ----------
        path:
            Path to a specific checkpoint directory (e.g. ``step_10000/``).
            If *None*, the latest checkpoint is loaded.

        Returns
        -------
        dict
            Dictionary with keys ``'model'``, ``'optimizer'``, ``'scheduler'``,
            ``'extra_state'``.  The ``'model'`` value is a ``dict[str, Tensor]``
            (from safetensors); the others are plain dicts suitable for
            ``.load_state_dict()``.

        Raises
        ------
        FileNotFoundError
            If no checkpoint is found.
        """
        if path is None:
            return self.load_latest()

        ckpt_dir = Path(path)
        return self._load_from_dir(ckpt_dir)

    def load_latest(self) -> dict[str, Any]:
        """Find and load the most recent step checkpoint.

        Returns
        -------
        dict
            Same structure as :meth:`load`.

        Raises
        ------
        FileNotFoundError
            If no step checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in {self.checkpoint_dir}"
            )
        latest = checkpoints[-1]  # sorted by step, last is highest
        logger.info("Loading latest checkpoint: %s", latest)
        return self._load_from_dir(latest)

    def load_best(self) -> dict[str, Any]:
        """Load the best checkpoint.

        Returns
        -------
        dict
            Same structure as :meth:`load`.

        Raises
        ------
        FileNotFoundError
            If no best checkpoint exists.
        """
        best_dir = self.checkpoint_dir / "best"
        if not best_dir.exists():
            raise FileNotFoundError(
                f"No best checkpoint found at {best_dir}"
            )
        logger.info("Loading best checkpoint: %s", best_dir)
        return self._load_from_dir(best_dir)

    def list_checkpoints(self) -> list[Path]:
        """List all ``step_*`` checkpoint directories sorted by step number.

        Returns
        -------
        list[Path]
            Sorted list of checkpoint directories (ascending by step).
        """
        if not self.checkpoint_dir.exists():
            return []

        step_dirs: list[tuple[int, Path]] = []
        for child in self.checkpoint_dir.iterdir():
            if not child.is_dir():
                continue
            match = _STEP_DIR_RE.match(child.name)
            if match:
                step_dirs.append((int(match.group(1)), child))

        step_dirs.sort(key=lambda pair: pair[0])
        return [path for _, path in step_dirs]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_dir(self, ckpt_dir: Path) -> dict[str, Any]:
        """Load all checkpoint components from a directory.

        Raises
        ------
        FileNotFoundError
            If the directory or required files do not exist.
        """
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        model_path = ckpt_dir / _MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Model weights from safetensors (returns dict[str, Tensor])
        model_state = load_file(model_path)

        # Optimizer
        optimizer_path = ckpt_dir / _OPTIMIZER_FILE
        optimizer_state = (
            torch.load(optimizer_path, map_location="cpu", weights_only=False)
            if optimizer_path.exists()
            else {}
        )

        # Scheduler
        scheduler_path = ckpt_dir / _SCHEDULER_FILE
        scheduler_state = (
            torch.load(scheduler_path, map_location="cpu", weights_only=False)
            if scheduler_path.exists()
            else {}
        )

        # Training state
        state_path = ckpt_dir / _TRAINING_STATE_FILE
        training_state = (
            torch.load(state_path, map_location="cpu", weights_only=False)
            if state_path.exists()
            else {}
        )

        logger.info(
            "Loaded checkpoint from %s (step %s)",
            ckpt_dir,
            training_state.get("global_step", "unknown"),
        )

        return {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "extra_state": training_state,
        }

    def _cleanup_old(self) -> None:
        """Remove old step checkpoints beyond :attr:`max_keep`.

        The ``best/`` checkpoint is never removed by this method.
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.max_keep:
            return

        to_remove = checkpoints[: len(checkpoints) - self.max_keep]
        for ckpt_path in to_remove:
            logger.info("Removing old checkpoint: %s", ckpt_path)
            shutil.rmtree(ckpt_path)
