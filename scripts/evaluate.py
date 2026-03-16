"""Evaluate a trained StyleStream pipeline.

Computes the standard evaluation metrics from the paper:
  - WER / CER  (Whisper-large-v3)
  - S-SIM      (Resemblyzer speaker similarity)
  - A-SIM      (Accent-ID ECAPA accent similarity)
  - E-SIM      (emotion2vec emotion similarity)
  - UTMOS       (MOS prediction)

Usage:
    python scripts/evaluate.py --converted-dir eval_results/converted --pairs pairs.csv
    python scripts/evaluate.py --converted-dir eval_results/converted --pairs pairs.csv --metrics wer,s_sim
    python scripts/evaluate.py --converted-dir eval_results/converted --pairs pairs.csv --output-dir eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torchaudio


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        level=level,
    )


def load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file, convert to mono, resample.

    Parameters
    ----------
    path : str
        Path to audio file.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    Tensor
        Waveform shape (samples,) at *sample_rate*.
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a StyleStream pipeline on standard metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate.py --converted-dir converted/ --pairs pairs.csv\n"
            "  python scripts/evaluate.py --converted-dir converted/ --pairs pairs.csv --metrics wer,s_sim\n"
            "  python scripts/evaluate.py --converted-dir converted/ --pairs pairs.csv --output-dir results/\n"
        ),
    )
    parser.add_argument(
        "--converted-dir",
        type=str,
        required=True,
        help="Directory containing converted audio files (from batch inference).",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to evaluation pairs CSV.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated metrics to compute (default: all). "
            "Options: wer,cer,s_sim,a_sim,e_sim,utmos"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to write evaluation results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for evaluation models.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "streaming"],
        help="Mode for paper baseline comparison.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to eval YAML config for paper baselines.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    converted_dir = Path(args.converted_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs
    from stylestream.eval.dataset import EvalDataset

    dataset = EvalDataset(args.pairs)
    logger.info("Loaded %d evaluation pairs.", len(dataset))

    # Determine metrics
    if args.metrics:
        metric_names = [m.strip() for m in args.metrics.split(",")]
    else:
        metric_names = ["wer", "s_sim", "a_sim", "e_sim", "utmos"]

    # Load paper baselines if config provided
    paper_baselines: dict = {}
    if args.config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        if hasattr(cfg, "paper_baselines"):
            paper_baselines = OmegaConf.to_container(
                cfg.paper_baselines, resolve=True
            )

    # Initialize evaluators
    from stylestream.eval.registry import get_evaluator

    evaluators: dict = {}
    for metric in metric_names:
        try:
            evaluators[metric] = get_evaluator(metric, device=args.device)
            logger.info("Initialized %s evaluator.", metric)
        except Exception as e:
            logger.error("Failed to initialize %s evaluator: %s", metric, e)

    # Initialize aggregator
    from stylestream.eval.aggregator import MetricsAggregator, PairMetrics

    aggregator = MetricsAggregator(paper_baselines=paper_baselines)

    # Evaluate each pair
    for i, pair in enumerate(dataset):
        converted_path = converted_dir / f"{pair.source_id}__{pair.target_id}.wav"
        if not converted_path.exists():
            logger.warning("Converted file not found: %s", converted_path)
            continue

        converted_audio = load_audio(str(converted_path))
        target_audio = (
            dataset.load_audio(pair.target_path)
            if any(m in metric_names for m in ["s_sim", "a_sim", "e_sim"])
            else None
        )

        pair_metrics: dict[str, float] = {}

        for metric_name, evaluator in evaluators.items():
            try:
                result = evaluator.evaluate_pair(
                    converted_audio=converted_audio,
                    target_audio=target_audio,
                    source_text=(
                        pair.source_text
                        if metric_name in ["wer", "cer"]
                        else None
                    ),
                )
                pair_metrics[metric_name] = result.value
            except Exception as e:
                logger.error(
                    "Error computing %s for %s->%s: %s",
                    metric_name,
                    pair.source_id,
                    pair.target_id,
                    e,
                )

        aggregator.add_result(
            PairMetrics(
                source_id=pair.source_id,
                target_id=pair.target_id,
                target_category=pair.target_category,
                target_label=pair.target_label,
                source_dataset=pair.source_dataset,
                metrics=pair_metrics,
            )
        )

        if (i + 1) % 100 == 0:
            logger.info("Evaluated %d/%d pairs.", i + 1, len(dataset))

    # Unload evaluators
    for evaluator in evaluators.values():
        evaluator.unload()

    # Save results
    aggregator.save_detailed_csv(output_dir / "detailed_results.csv")
    aggregator.save_summary_json(output_dir / "summary.json")

    # Print summary
    md_table = aggregator.to_markdown_table(mode=args.mode)
    logger.info("\n%s", md_table)

    # Save markdown table
    with open(output_dir / "comparison_table.md", "w") as f:
        f.write(md_table)

    # Generate charts (if matplotlib available)
    try:
        from stylestream.eval.visualization import (
            generate_category_bars,
            generate_comparison_heatmap,
            generate_html_report,
            generate_radar_chart,
        )

        overall = aggregator.overall_stats()
        our_metrics = {name: stats.mean for name, stats in overall.items()}

        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Radar chart
        paper_offline = paper_baselines.get("stylestream_offline", {})
        generate_radar_chart(
            our_metrics, paper_offline, charts_dir / "radar_chart.png"
        )

        # Category bars
        cat_stats = aggregator.stats_by_category()
        cat_stats_dict = {
            cat: {n: s.to_dict() for n, s in metrics.items()}
            for cat, metrics in cat_stats.items()
        }
        generate_category_bars(
            cat_stats_dict, charts_dir / "category_bars.png"
        )

        # Heatmap
        if paper_offline:
            generate_comparison_heatmap(
                our_metrics, paper_offline, charts_dir / "heatmap.png"
            )

        # HTML report
        generate_html_report(
            output_dir / "summary.json",
            output_dir / "report.html",
            charts_dir,
        )

        logger.info("Charts and report saved to %s", output_dir)
    except ImportError:
        logger.info("matplotlib not available, skipping chart generation.")

    logger.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
