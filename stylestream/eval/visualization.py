"""Visualization utilities for StyleStream evaluation results.

Generates:
    - Radar chart (5-axis: WER^-1, S-SIM, A-SIM, E-SIM, UTMOS)
    - Category bar charts (emotion vs accent breakdown)
    - Box plots per metric per label
    - Paper comparison heatmap
    - Integrated HTML report

All plotting functions lazily import ``matplotlib`` so that the module
can be imported without the eval-only dependency installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_radar_chart(
    metrics: dict[str, float],
    paper_metrics: dict[str, float] | None = None,
    output_path: str | Path = "radar_chart.png",
    title: str = "StyleStream Evaluation",
) -> None:
    """Generate a radar/spider chart comparing metrics.

    The five axes are WER^{-1} (inverted so higher is better), S-SIM,
    A-SIM, E-SIM, and UTMOS, all normalised to a 0--100 scale.

    Parameters
    ----------
    metrics : dict
        Metric name -> value (our results).
    paper_metrics : dict or None
        Paper baseline values for overlay.
    output_path : Path
        Output image path.
    title : str
        Chart title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Normalise metrics for radar chart display
    # WER: lower is better, so use (100 - WER) for the chart
    labels: list[str] = []
    values: list[float] = []
    paper_values: list[float] = []

    metric_order = ["wer", "s_sim", "a_sim", "e_sim", "utmos"]
    display_names = {
        "wer": "1/WER",
        "s_sim": "S-SIM",
        "a_sim": "A-SIM",
        "e_sim": "E-SIM",
        "utmos": "UTMOS",
    }

    for m in metric_order:
        if m not in metrics:
            continue
        labels.append(display_names.get(m, m))
        if m == "wer":
            # Invert WER (lower is better -> higher on chart)
            values.append(max(0.0, 100.0 - metrics[m]))
            if paper_metrics and m in paper_metrics:
                paper_values.append(max(0.0, 100.0 - paper_metrics[m]))
        elif m == "utmos":
            # Scale MOS 1-5 to 0-100
            values.append((metrics[m] - 1.0) / 4.0 * 100.0)
            if paper_metrics and m in paper_metrics:
                paper_values.append((paper_metrics[m] - 1.0) / 4.0 * 100.0)
        else:
            # Similarity: already 0-1, scale to 0-100
            values.append(metrics[m] * 100.0)
            if paper_metrics and m in paper_metrics:
                paper_values.append(paper_metrics[m] * 100.0)

    n = len(labels)
    if n == 0:
        logger.warning("No metrics to plot for radar chart.")
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, "o-", linewidth=2, label="Ours")
    ax.fill(angles_closed, values_closed, alpha=0.25)

    if paper_values and len(paper_values) == n:
        paper_closed = paper_values + paper_values[:1]
        ax.plot(angles_closed, paper_closed, "s--", linewidth=2, label="Paper")
        ax.fill(angles_closed, paper_closed, alpha=0.1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved radar chart to %s", output_path)


def generate_category_bars(
    stats_by_category: dict[str, dict[str, Any]],
    output_path: str | Path = "category_bars.png",
    title: str = "Metrics by Category",
) -> None:
    """Generate grouped bar chart by category (emotion vs accent).

    Parameters
    ----------
    stats_by_category : dict
        Mapping of category name to metric stats.  Each metric stats entry
        can be either a :class:`~stylestream.eval.aggregator.MetricStats`
        object or a dict with ``"mean"`` and ``"std"`` keys.
    output_path : str or Path
        Output image path.
    title : str
        Chart title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    categories = sorted(stats_by_category.keys())
    if not categories:
        logger.warning("No categories to plot.")
        return

    # Collect all metric names
    all_metrics: set[str] = set()
    for cat_stats in stats_by_category.values():
        all_metrics.update(cat_stats.keys())
    metric_names = sorted(all_metrics)

    x = np.arange(len(metric_names))
    width = 0.8 / len(categories)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, cat in enumerate(categories):
        means: list[float] = []
        stds: list[float] = []
        for m in metric_names:
            if m in stats_by_category[cat]:
                s = stats_by_category[cat][m]
                means.append(s["mean"] if isinstance(s, dict) else s.mean)
                stds.append(s["std"] if isinstance(s, dict) else s.std)
            else:
                means.append(0.0)
                stds.append(0.0)

        ax.bar(x + i * width, means, width, yerr=stds, label=cat, capsize=3)

    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_title(title)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved category bars to %s", output_path)


def generate_label_boxplots(
    results: list[dict],
    metric: str,
    output_path: str | Path = "boxplot.png",
    title: str = "",
) -> None:
    """Generate box plots for a metric grouped by target label.

    Parameters
    ----------
    results : list[dict]
        List of result dicts, each containing ``"target_label"`` and a
        ``"metrics"`` sub-dict.
    metric : str
        Metric name to plot.
    output_path : str or Path
        Output image path.
    title : str
        Chart title.  Defaults to ``"{metric} by Target Label"``.
    """
    import matplotlib.pyplot as plt

    # Group values by label
    by_label: dict[str, list[float]] = {}
    for r in results:
        label = r.get("target_label", "unknown")
        val = r.get("metrics", {}).get(metric, None)
        if val is not None:
            by_label.setdefault(label, []).append(val)

    if not by_label:
        logger.warning("No data for metric '%s' to plot.", metric)
        return

    labels = sorted(by_label.keys())
    data = [by_label[lab] for lab in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=labels)
    ax.set_title(title or f"{metric} by Target Label")
    ax.set_ylabel(metric)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved boxplot to %s", output_path)


def generate_comparison_heatmap(
    our_metrics: dict[str, float],
    paper_metrics: dict[str, float],
    output_path: str | Path = "heatmap.png",
    title: str = "Paper Comparison",
) -> None:
    """Generate a heatmap showing achievement vs paper targets.

    Each cell shows the achievement ratio.  Values >= 1.0 (met or exceeded)
    are coloured green; values < 1.0 (below target) are coloured red.

    Parameters
    ----------
    our_metrics : dict
        Our metric name -> value.
    paper_metrics : dict
        Paper metric name -> value.
    output_path : str or Path
        Output image path.
    title : str
        Chart title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Direction: higher_is_better except WER/CER
    directions = {
        "wer": "lower_is_better",
        "cer": "lower_is_better",
        "s_sim": "higher_is_better",
        "a_sim": "higher_is_better",
        "e_sim": "higher_is_better",
        "utmos": "higher_is_better",
    }

    metrics = sorted(set(our_metrics.keys()) & set(paper_metrics.keys()))
    if not metrics:
        logger.warning("No overlapping metrics for heatmap.")
        return

    # Compute relative achievement
    achievements: list[float] = []
    for m in metrics:
        ours = our_metrics[m]
        paper = paper_metrics[m]
        direction = directions.get(m, "higher_is_better")

        if direction == "higher_is_better":
            achievement = ours / paper if paper != 0 else 1.0
        else:
            achievement = paper / ours if ours != 0 else 1.0

        achievements.append(achievement)

    fig, ax = plt.subplots(figsize=(8, 2 + 0.5 * len(metrics)))

    data = np.array(achievements).reshape(1, -1)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.8, vmax=1.2, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels(["Achievement"])
    ax.set_title(title)

    for i, (m, a) in enumerate(zip(metrics, achievements)):
        ax.text(i, 0, f"{a:.2f}", ha="center", va="center", fontweight="bold")

    fig.colorbar(im, ax=ax, label="Ratio (>=1.0 = met)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)


def generate_html_report(
    summary_json_path: str | Path,
    output_path: str | Path = "report.html",
    charts_dir: str | Path | None = None,
) -> None:
    """Generate an integrated HTML report from summary JSON.

    The report is self-contained with inline CSS styles and uses relative
    paths for chart images.

    Parameters
    ----------
    summary_json_path : Path
        Path to the summary JSON from :class:`MetricsAggregator`.
    output_path : Path
        Output HTML file path.
    charts_dir : Path or None
        Directory containing generated chart images.
    """
    summary_json_path = Path(summary_json_path)
    output_path = Path(output_path)

    with open(summary_json_path, encoding="utf-8") as f:
        summary = json.load(f)

    overall = summary.get("overall", {})
    by_category = summary.get("by_category", {})
    total_pairs = summary.get("total_pairs", 0)

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset=\"utf-8\">",
        "<title>StyleStream Evaluation Report</title>",
        "<style>",
        "body { font-family: sans-serif; max-width: 1200px; margin: auto; padding: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }",
        "th { background: #4a90d9; color: white; }",
        "tr:nth-child(even) { background: #f2f2f2; }",
        ".good { color: green; font-weight: bold; }",
        ".bad { color: red; }",
        "h1, h2, h3 { color: #333; }",
        "img { max-width: 100%; margin: 10px 0; }",
        "</style>",
        "</head><body>",
        "<h1>StyleStream Evaluation Report</h1>",
        f"<p>Total pairs evaluated: {total_pairs}</p>",
    ]

    # Overall results table
    html_parts.append("<h2>Overall Results</h2>")
    html_parts.append(
        "<table><tr><th>Metric</th><th>Mean</th><th>Std</th>"
        "<th>95% CI</th><th>Count</th></tr>"
    )
    for name, stats in overall.items():
        ci = stats.get("ci_95", [0, 0])
        html_parts.append(
            f"<tr><td>{name}</td><td>{stats['mean']:.4f}</td>"
            f"<td>{stats['std']:.4f}</td>"
            f"<td>[{ci[0]:.4f}, {ci[1]:.4f}]</td>"
            f"<td>{stats['count']}</td></tr>"
        )
    html_parts.append("</table>")

    # By category
    if by_category:
        html_parts.append("<h2>Results by Category</h2>")
        for cat, cat_metrics in by_category.items():
            html_parts.append(f"<h3>{cat.title()}</h3>")
            html_parts.append(
                "<table><tr><th>Metric</th><th>Mean</th><th>Std</th></tr>"
            )
            for name, stats in cat_metrics.items():
                html_parts.append(
                    f"<tr><td>{name}</td>"
                    f"<td>{stats['mean']:.4f}</td>"
                    f"<td>{stats['std']:.4f}</td></tr>"
                )
            html_parts.append("</table>")

    # Charts
    if charts_dir:
        charts_dir = Path(charts_dir)
        for img_name in ["radar_chart.png", "category_bars.png", "heatmap.png"]:
            img_path = charts_dir / img_name
            if img_path.exists():
                readable_name = (
                    img_name.replace("_", " ").replace(".png", "").title()
                )
                html_parts.append(f"<h2>{readable_name}</h2>")
                html_parts.append(
                    f'<img src="{img_path.name}" alt="{img_name}">'
                )

    html_parts.append("</body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info("Saved HTML report to %s", output_path)
