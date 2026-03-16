"""Logging utilities for StyleStream."""

import logging
from pathlib import Path


_LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: str | None = None,
    rank: int = 0,
) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name.
        level: Log level string (e.g. "INFO", "DEBUG").
        log_dir: If provided, a file handler writes to
            ``{log_dir}/{name}_rank{rank}.log``.
        rank: Distributed-training rank.  Only rank 0 gets a console handler.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers when called multiple times.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler — rank 0 only.
    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logger.addHandler(console)

    # File handler — all ranks.
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{name}_rank{rank}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return an existing logger by *name*.

    This is a thin wrapper around :func:`logging.getLogger` for convenience.
    """
    return logging.getLogger(name)


def log_metrics(
    logger: logging.Logger,
    step: int,
    metrics: dict[str, float],
    prefix: str = "train",
) -> None:
    """Log a dictionary of metrics in a compact, readable format.

    Example output::

        [train] step=1000 | loss=0.234 | lr=0.0001
    """
    parts = [f"step={step}"]
    for key, value in metrics.items():
        parts.append(f"{key}={value:.4g}")
    logger.info("[%s] %s", prefix, " | ".join(parts))
