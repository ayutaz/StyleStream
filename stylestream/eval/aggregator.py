"""Metrics aggregation and reporting for StyleStream evaluation.

Aggregates per-pair evaluation results into summary statistics
by category (emotion/accent), source dataset, and overall.

Outputs:
    - Detailed CSV with per-pair results
    - Summary JSON with mean, std, 95% CI per metric/category
    - Comparison table against paper baselines
    - Markdown/LaTeX formatted tables
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricStats:
    """Statistics for a single metric."""

    name: str
    mean: float
    std: float
    ci_95_low: float
    ci_95_high: float
    count: int
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci_95": [round(self.ci_95_low, 4), round(self.ci_95_high, 4)],
            "count": self.count,
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "median": round(self.median, 4),
        }


def compute_stats(values: list[float], name: str = "") -> MetricStats:
    """Compute summary statistics for a list of values.

    Parameters
    ----------
    values : list[float]
        Raw metric values.
    name : str
        Metric name (stored in the returned :class:`MetricStats`).

    Returns
    -------
    MetricStats
        Aggregated statistics including mean, std, 95% CI, min, max, median.
    """
    if not values:
        return MetricStats(
            name=name, mean=0, std=0, ci_95_low=0, ci_95_high=0, count=0
        )

    n = len(values)
    mean = sum(values) / n

    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    # 95% CI: mean +/- 1.96 * std / sqrt(n)
    se = std / math.sqrt(n) if n > 0 else 0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    sorted_vals = sorted(values)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    return MetricStats(
        name=name,
        mean=mean,
        std=std,
        ci_95_low=ci_low,
        ci_95_high=ci_high,
        count=n,
        min=min(values),
        max=max(values),
        median=median,
    )


@dataclass
class PairMetrics:
    """All metrics for a single evaluation pair.

    Parameters
    ----------
    source_id : str
        Identifier for the source utterance.
    target_id : str
        Identifier for the target utterance.
    target_category : str
        ``"emotion"`` or ``"accent"``.
    target_label : str
        Specific label (e.g., ``"happy"``, ``"british"``).
    source_dataset : str
        Source dataset name (e.g., ``"esd"``, ``"globe"``, ``"libritts"``).
    metrics : dict[str, float]
        Mapping of metric name to value.
    metadata : dict[str, Any]
        Arbitrary extra information.
    """

    source_id: str
    target_id: str
    target_category: str = ""
    target_label: str = ""
    source_dataset: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_pair_result(
        pair_result: Any,
        target_category: str = "",
        target_label: str = "",
        source_dataset: str = "",
    ) -> PairMetrics:
        """Create from a :class:`~stylestream.eval.base.PairResult`.

        Parameters
        ----------
        pair_result : PairResult
            Result from the base evaluation module.
        target_category : str
            Category for this pair.
        target_label : str
            Label for this pair.
        source_dataset : str
            Source dataset name.

        Returns
        -------
        PairMetrics
        """
        return PairMetrics(
            source_id=pair_result.source_id,
            target_id=pair_result.target_id,
            target_category=target_category,
            target_label=target_label,
            source_dataset=source_dataset,
            metrics=dict(pair_result.metrics),
            metadata=dict(pair_result.metadata),
        )


class MetricsAggregator:
    """Aggregates evaluation results and produces reports.

    Parameters
    ----------
    paper_baselines : dict or None
        Paper baseline values for comparison, keyed by system name
        (e.g. ``"stylestream_offline"``).  Each value is a dict mapping
        metric name to float.
    """

    def __init__(self, paper_baselines: dict | None = None) -> None:
        self._results: list[PairMetrics] = []
        self.paper_baselines = paper_baselines or {}

    def add_result(self, result: PairMetrics) -> None:
        """Add a single pair result."""
        self._results.append(result)

    def add_results(self, results: list[PairMetrics]) -> None:
        """Add multiple pair results."""
        self._results.extend(results)

    @property
    def metric_names(self) -> list[str]:
        """All metric names across results (sorted)."""
        names: set[str] = set()
        for r in self._results:
            names.update(r.metrics.keys())
        return sorted(names)

    @property
    def count(self) -> int:
        """Number of pair results stored."""
        return len(self._results)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def overall_stats(self) -> dict[str, MetricStats]:
        """Compute overall statistics for each metric."""
        by_metric: dict[str, list[float]] = defaultdict(list)
        for r in self._results:
            for name, value in r.metrics.items():
                by_metric[name].append(value)
        return {name: compute_stats(values, name) for name, values in by_metric.items()}

    def stats_by_category(self) -> dict[str, dict[str, MetricStats]]:
        """Compute statistics grouped by target category (emotion/accent)."""
        groups: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in self._results:
            cat = r.target_category or "unknown"
            for name, value in r.metrics.items():
                groups[cat][name].append(value)

        return {
            cat: {name: compute_stats(values, name) for name, values in metrics.items()}
            for cat, metrics in groups.items()
        }

    def stats_by_label(self) -> dict[str, dict[str, MetricStats]]:
        """Compute statistics grouped by target label (happy, british, etc.)."""
        groups: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in self._results:
            label = r.target_label or "unknown"
            for name, value in r.metrics.items():
                groups[label][name].append(value)

        return {
            label: {name: compute_stats(values, name) for name, values in metrics.items()}
            for label, metrics in groups.items()
        }

    def stats_by_source_dataset(self) -> dict[str, dict[str, MetricStats]]:
        """Compute statistics grouped by source dataset."""
        groups: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in self._results:
            ds = r.source_dataset or "unknown"
            for name, value in r.metrics.items():
                groups[ds][name].append(value)

        return {
            ds: {name: compute_stats(values, name) for name, values in metrics.items()}
            for ds, metrics in groups.items()
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_detailed_csv(self, output_path: str | Path) -> None:
        """Save per-pair results to CSV.

        Parameters
        ----------
        output_path : str or Path
            Destination CSV file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metric_names = self.metric_names
        fieldnames = [
            "source_id",
            "target_id",
            "target_category",
            "target_label",
            "source_dataset",
        ] + metric_names

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                row: dict[str, Any] = {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "target_category": r.target_category,
                    "target_label": r.target_label,
                    "source_dataset": r.source_dataset,
                }
                row.update(r.metrics)
                writer.writerow(row)

        logger.info(
            "Saved detailed results to %s (%d pairs)", output_path, len(self._results)
        )

    def save_summary_json(self, output_path: str | Path) -> None:
        """Save aggregated summary to JSON.

        Parameters
        ----------
        output_path : str or Path
            Destination JSON file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {
            "total_pairs": self.count,
            "overall": {
                n: s.to_dict() for n, s in self.overall_stats().items()
            },
            "by_category": {
                cat: {n: s.to_dict() for n, s in metrics.items()}
                for cat, metrics in self.stats_by_category().items()
            },
            "by_label": {
                label: {n: s.to_dict() for n, s in metrics.items()}
                for label, metrics in self.stats_by_label().items()
            },
            "by_source_dataset": {
                ds: {n: s.to_dict() for n, s in metrics.items()}
                for ds, metrics in self.stats_by_source_dataset().items()
            },
        }

        if self.paper_baselines:
            summary["paper_baselines"] = self.paper_baselines

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved summary to %s", output_path)

    def to_markdown_table(self, mode: str = "offline") -> str:
        """Generate a Markdown comparison table against paper baselines.

        Parameters
        ----------
        mode : str
            ``"offline"`` or ``"streaming"`` -- selects which paper baseline
            to compare against.

        Returns
        -------
        str
            Markdown-formatted table.
        """
        overall = self.overall_stats()
        baseline_key = f"stylestream_{mode}"
        baseline = self.paper_baselines.get(baseline_key, {})

        lines = []
        lines.append("| Metric | Ours | Paper | Diff |")
        lines.append("|--------|------|-------|------|")

        for name, stats in overall.items():
            paper_val = baseline.get(name, None)
            if paper_val is not None:
                diff = stats.mean - paper_val
                diff_str = f"{diff:+.3f}"
                paper_str = f"{paper_val}"
            else:
                paper_str = "---"
                diff_str = "---"

            lines.append(
                f"| {name} | {stats.mean:.3f} +/- {stats.std:.3f} "
                f"| {paper_str} | {diff_str} |"
            )

        return "\n".join(lines)

    def to_latex_table(self, mode: str = "offline") -> str:
        """Generate a LaTeX comparison table.

        Parameters
        ----------
        mode : str
            ``"offline"`` or ``"streaming"``.

        Returns
        -------
        str
            LaTeX-formatted table.
        """
        overall = self.overall_stats()
        baseline_key = f"stylestream_{mode}"
        baseline = self.paper_baselines.get(baseline_key, {})

        lines = []
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Metric & Ours & Paper & $\Delta$ \\")
        lines.append(r"\midrule")

        for name, stats in overall.items():
            paper_val = baseline.get(name, None)
            if paper_val is not None:
                diff = stats.mean - paper_val
                paper_str = f"{paper_val:.3f}"
                diff_str = f"{diff:+.3f}"
            else:
                paper_str = "---"
                diff_str = "---"

            lines.append(
                f"{name} & {stats.mean:.3f} $\\pm$ {stats.std:.3f} "
                f"& {paper_str} & {diff_str} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        return "\n".join(lines)
