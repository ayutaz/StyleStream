"""Learning rate scheduler for StyleStream training.

Implements cosine annealing with linear warmup, used by all StyleStream
components:
    - Destylizer: peak_lr=1e-4, warmup=4000, total=100000
    - Stylizer:   peak_lr=1e-4, warmup=2000, total=400000

Schedule:
    1. Steps [0, warmup_steps): linear ramp from 0 to peak_lr
    2. Steps [warmup_steps, total_steps]: cosine decay from peak_lr to min_lr
    3. Steps > total_steps: constant at min_lr
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_lr_at_step(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Compute the learning rate at a given step (for testing / debugging).

    Parameters
    ----------
    step:
        Current training step (0-indexed).
    warmup_steps:
        Number of warmup steps during which the lr ramps linearly.
    total_steps:
        Total number of training steps (warmup + cosine decay).
    peak_lr:
        Maximum learning rate reached at the end of warmup.
    min_lr:
        Minimum learning rate at the end of cosine decay.

    Returns
    -------
    float
        Learning rate at *step*.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if warmup_steps > total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be <= total_steps ({total_steps})"
        )

    # Phase 1: linear warmup
    if step < warmup_steps:
        if warmup_steps == 0:
            return peak_lr
        return peak_lr * (step / warmup_steps)

    # Phase 3: past total_steps -- clamp at min_lr
    if step >= total_steps:
        return min_lr

    # Phase 2: cosine decay
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps  # 0.0 -> 1.0
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine_decay


class CosineAnnealingWarmup:
    """Cosine annealing with linear warmup scheduler.

    Wraps :class:`torch.optim.lr_scheduler.LambdaLR` so that all standard
    PyTorch scheduler methods (``.step()``, ``.state_dict()``,
    ``.load_state_dict()``) work transparently.

    Parameters
    ----------
    optimizer:
        The optimizer whose learning rate will be scheduled.
    warmup_steps:
        Number of steps for the linear warmup phase.
    total_steps:
        Total number of training steps (warmup + cosine decay).
    peak_lr:
        Maximum learning rate reached at the end of warmup.
    min_lr:
        Minimum learning rate at the end of cosine decay.  Default 0.

    Examples
    --------
    >>> import torch
    >>> model = torch.nn.Linear(10, 10)
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    >>> scheduler = CosineAnnealingWarmup(
    ...     optimizer, warmup_steps=4000, total_steps=100000, peak_lr=1e-4
    ... )
    >>> for step in range(100000):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if warmup_steps > total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be <= total_steps ({total_steps})"
            )

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        # LambdaLR multiplies the *base_lr* (i.e. the lr set on each param
        # group) by the value returned from the lambda.  We normalise so that
        # the lambda returns 1.0 at peak and min_lr/peak_lr at the floor.
        #
        # If peak_lr is 0 the schedule is degenerate -- always 0.
        def _lr_lambda(current_step: int) -> float:
            if peak_lr == 0.0:
                return 0.0
            return get_lr_at_step(
                current_step, warmup_steps, total_steps, peak_lr, min_lr
            ) / peak_lr

        self._scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # ------------------------------------------------------------------
    # Delegate standard scheduler interface to the wrapped LambdaLR
    # ------------------------------------------------------------------

    def step(self, epoch: int | None = None) -> None:
        """Advance the scheduler by one step."""
        self._scheduler.step(epoch)

    def state_dict(self) -> dict:
        """Return the scheduler state for checkpointing."""
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the scheduler state from a checkpoint."""
        self._scheduler.load_state_dict(state_dict)

    def get_last_lr(self) -> list[float]:
        """Return the last computed learning rate for each param group."""
        return self._scheduler.get_last_lr()

    @property
    def last_epoch(self) -> int:
        """Current step count (LambdaLR uses ``last_epoch`` internally)."""
        return self._scheduler.last_epoch

    @property
    def optimizer(self) -> Optimizer:
        """The wrapped optimizer."""
        return self._scheduler.optimizer

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"warmup_steps={self.warmup_steps}, "
            f"total_steps={self.total_steps}, "
            f"peak_lr={self.peak_lr}, "
            f"min_lr={self.min_lr})"
        )
