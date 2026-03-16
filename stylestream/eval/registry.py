"""Registry of all available evaluation metrics.

Provides lazy-loaded access to evaluator classes so that heavy dependencies
(Whisper, Resemblyzer, emotion2vec, etc.) are only imported when the
corresponding evaluator is actually requested.

Usage::

    from stylestream.eval.registry import get_evaluator, available_metrics

    # List what is available
    print(available_metrics())
    # ['a_sim', 'cer', 'e_sim', 'mos', 's_sim', 'speaker_sim', 'utmos', 'wer']

    # Get a specific evaluator (not yet loaded)
    evaluator = get_evaluator("wer", device="cuda")
    evaluator.load()
    result = evaluator.evaluate_pair(converted, source_text="hello world")
    evaluator.unload()

    # Or use the context-manager protocol
    with get_evaluator("s_sim", device="cuda") as evaluator:
        result = evaluator.evaluate_pair(converted, target)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stylestream.eval.base import BaseEvaluator


# ---------------------------------------------------------------------------
# Registry: metric name -> (module_path, class_name)
# ---------------------------------------------------------------------------

_EVALUATOR_CLASSES: dict[str, tuple[str, str]] = {
    # Content preservation
    "wer": ("stylestream.eval.whisper_evaluator", "WhisperEvaluator"),
    "cer": ("stylestream.eval.whisper_evaluator", "WhisperEvaluator"),
    # Speaker / timbre similarity
    "s_sim": ("stylestream.eval.resemblyzer_evaluator", "ResemblyzerEvaluator"),
    "speaker_sim": ("stylestream.eval.resemblyzer_evaluator", "ResemblyzerEvaluator"),
    # Accent similarity
    "a_sim": ("stylestream.eval.accent_evaluator", "AccentEvaluator"),
    # Emotion similarity
    "e_sim": ("stylestream.eval.emotion_evaluator", "EmotionEvaluator"),
    # Speech quality
    "utmos": ("stylestream.eval.utmos_evaluator", "UTMOSEvaluator"),
    "mos": ("stylestream.eval.utmos_evaluator", "UTMOSEvaluator"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_evaluator(metric: str, **kwargs: object) -> BaseEvaluator:
    """Get an evaluator instance by metric name.

    The returned evaluator is **not** loaded yet -- call
    :meth:`~BaseEvaluator.load` (or use the context-manager protocol)
    to initialise the underlying model.

    Parameters
    ----------
    metric : str
        Metric identifier.  One of :func:`available_metrics`.
    **kwargs
        Forwarded to the evaluator constructor (e.g. ``device="cuda"``).

    Returns
    -------
    BaseEvaluator
        Evaluator instance (not yet loaded).

    Raises
    ------
    ValueError
        If *metric* is not in the registry.
    """
    if metric not in _EVALUATOR_CLASSES:
        available = sorted(set(_EVALUATOR_CLASSES.keys()))
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {available}"
        )

    module_path, class_name = _EVALUATOR_CLASSES[metric]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    # Special handling: CER uses WhisperEvaluator with use_cer=True
    if metric == "cer":
        kwargs.setdefault("use_cer", True)

    return cls(**kwargs)


def get_all_evaluators(**kwargs: object) -> dict[str, BaseEvaluator]:
    """Get one instance per unique evaluator class.

    Deduplicates by ``(module_path, class_name)`` so that, for example,
    ``"wer"`` and ``"cer"`` do not instantiate the same class twice --
    only ``"wer"`` is returned (``"cer"`` would be a duplicate).

    Parameters
    ----------
    **kwargs
        Forwarded to each evaluator constructor.

    Returns
    -------
    dict[str, BaseEvaluator]
        Mapping of canonical metric name to evaluator instance.
    """
    seen: set[str] = set()
    evaluators: dict[str, BaseEvaluator] = {}
    for metric, (mod_path, cls_name) in _EVALUATOR_CLASSES.items():
        key = f"{mod_path}.{cls_name}"
        if key in seen:
            continue
        seen.add(key)
        evaluators[metric] = get_evaluator(metric, **kwargs)
    return evaluators


def available_metrics() -> list[str]:
    """Return a sorted list of all registered metric names.

    Returns
    -------
    list[str]
        Sorted metric identifiers (includes aliases like ``"mos"``
        and ``"speaker_sim"``).
    """
    return sorted(set(_EVALUATOR_CLASSES.keys()))
