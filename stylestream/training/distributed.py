"""Distributed training utilities for StyleStream.

Thin wrappers around HuggingFace Accelerate to keep the rest of the
codebase decoupled from the specific distributed-training backend.
"""

from __future__ import annotations

from typing import Any

from accelerate import Accelerator


def get_accelerator(**kwargs: Any) -> Accelerator:
    """Create and return a configured :class:`Accelerator` instance.

    All keyword arguments are forwarded to the ``Accelerator`` constructor.
    Common keys include ``mixed_precision`` (``"no"``, ``"fp16"``,
    ``"bf16"``), ``gradient_accumulation_steps``, and ``log_with``.

    Returns
    -------
    Accelerator
        Ready-to-use accelerator.
    """
    return Accelerator(**kwargs)


def is_main_process(accelerator: Accelerator) -> bool:
    """Return *True* if the current process is the main (rank-0) process.

    This is the recommended guard for operations that should only happen
    once across all workers -- e.g. logging to W&B, saving a single
    checkpoint copy, or printing progress bars.
    """
    return accelerator.is_main_process


def wait_for_everyone(accelerator: Accelerator) -> None:
    """Block until all processes have reached this point.

    Wraps :meth:`Accelerator.wait_for_everyone` to serve as a clean
    synchronisation barrier (e.g. before loading a freshly-saved
    checkpoint or after a validation pass).
    """
    accelerator.wait_for_everyone()


def print_rank0(accelerator: Accelerator, msg: str) -> None:
    """Print *msg* only on the main (rank-0) process.

    For logging that should go through Python's :mod:`logging` module,
    prefer :func:`stylestream.utils.logging.setup_logger` with
    ``rank=accelerator.process_index`` instead.
    """
    if accelerator.is_main_process:
        print(msg)  # noqa: T201


def gather_scalar(accelerator: Accelerator, value: float) -> float:
    """Gather a scalar metric across all processes and return the mean.

    Useful for aggregating per-process loss values before logging.

    Parameters
    ----------
    accelerator :
        The active accelerator.
    value :
        A Python float to broadcast and average.

    Returns
    -------
    float
        The mean of *value* across all processes.
    """
    import torch

    tensor = torch.tensor([value], device=accelerator.device)
    gathered = accelerator.gather(tensor)
    return gathered.mean().item()


def get_world_size(accelerator: Accelerator) -> int:
    """Return the total number of processes in the current group."""
    return accelerator.num_processes


def get_rank(accelerator: Accelerator) -> int:
    """Return the rank (process index) of the current process."""
    return accelerator.process_index
