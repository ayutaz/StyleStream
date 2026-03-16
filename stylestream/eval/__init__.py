"""Evaluation pipeline for StyleStream voice style conversion.

Exports
-------
BaseEvaluator
    Abstract base class for all metric evaluators.
SimilarityEvaluator
    Base for embedding similarity metrics (S-SIM, A-SIM, E-SIM).
EvalResult
    Single metric result dataclass.
PairResult
    Per-pair result dataclass.
WhisperEvaluator
    WER/CER computation using Whisper-large-v3.
ResemblyzerEvaluator
    Speaker similarity (S-SIM) using Resemblyzer.
AccentEvaluator
    Accent similarity (A-SIM) using ECAPA-TDNN.
EmotionEvaluator
    Emotion similarity (E-SIM) using emotion2vec.
UTMOSEvaluator
    MOS prediction using UTMOS.
EvalDataset
    StyleStream-Test dataset (3000 pairs).
BatchInference
    Batch conversion pipeline.
MetricsAggregator
    Results aggregation and reporting.
MetricStats, PairMetrics, compute_stats
    Aggregation data structures and helpers.
StyleProbing
    Linear probing for style leakage analysis.
get_evaluator, available_metrics
    Evaluator registry.
"""

from __future__ import annotations

from stylestream.eval.aggregator import (
    MetricStats,
    MetricsAggregator,
    PairMetrics,
    compute_stats,
)
from stylestream.eval.base import (
    BaseEvaluator,
    EvalResult,
    PairResult,
    SimilarityEvaluator,
)
from stylestream.eval.batch_inference import BatchInference
from stylestream.eval.registry import (
    available_metrics,
    get_all_evaluators,
    get_evaluator,
)

__all__ = [
    "BaseEvaluator",
    "SimilarityEvaluator",
    "EvalResult",
    "PairResult",
    "BatchInference",
    "MetricsAggregator",
    "MetricStats",
    "PairMetrics",
    "compute_stats",
    "get_evaluator",
    "get_all_evaluators",
    "available_metrics",
]
