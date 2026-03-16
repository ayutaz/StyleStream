"""Preprocess audio data for StyleStream training.

Usage:
    python scripts/preprocess_data.py --manifest data/manifests/libritts.csv --output-dir data/processed
    python scripts/preprocess_data.py --manifest data/manifests/lmg.csv --output-dir data/processed --stages resample mel
    python scripts/preprocess_data.py --manifest data/manifests/lmg.csv --output-dir data/processed --stages all --num-workers 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from stylestream.data.manifest import Manifest
from stylestream.data.preprocessing import PreprocessingPipeline
from stylestream.utils.logging import setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess audio data for StyleStream training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/preprocess_data.py \\\n"
            "      --manifest data/manifests/libritts.csv \\\n"
            "      --output-dir data/processed\n"
            "\n"
            "  python scripts/preprocess_data.py \\\n"
            "      --manifest data/manifests/lmg.csv \\\n"
            "      --output-dir data/processed \\\n"
            "      --stages resample mel\n"
            "\n"
            "  python scripts/preprocess_data.py \\\n"
            "      --manifest data/manifests/lmg.csv \\\n"
            "      --output-dir data/processed \\\n"
            "      --stages all --num-workers 16\n"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to a manifest CSV file (see stylestream.data.manifest).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for preprocessed outputs.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["resample", "mel", "all"],
        default=["all"],
        help='Stages to run (default: "all"). '
        "Multiple stages can be specified: --stages resample mel",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-process files even if they already exist.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Run verification after preprocessing.",
    )
    parser.add_argument(
        "--save-resampled-manifest",
        type=str,
        default=None,
        help="Path to save the resampled manifest CSV. "
        "If not set, saves to output-dir/manifest_16k.csv.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Setup logging
    setup_logger("stylestream", level=args.log_level)
    logger = logging.getLogger("stylestream.scripts.preprocess_data")
    logger.info("Starting preprocessing pipeline")

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest file not found: %s", manifest_path)
        sys.exit(1)

    manifest = Manifest.load(manifest_path)
    logger.info("Loaded manifest: %d utterances", len(manifest))

    # Create pipeline
    output_dir = Path(args.output_dir)
    pipeline = PreprocessingPipeline(
        manifest=manifest,
        output_dir=output_dir,
        sample_rate=16000,
        num_workers=args.num_workers,
    )

    stages = set(args.stages)
    run_resample = "resample" in stages or "all" in stages
    run_mel = "mel" in stages or "all" in stages

    resampled_manifest = None

    # Stage 1: Resample
    if run_resample:
        resampled_manifest = pipeline.run_resample(skip_existing=args.skip_existing)

        # Save resampled manifest
        save_path = args.save_resampled_manifest
        if save_path is None:
            save_path = str(output_dir / "manifest_16k.csv")
        resampled_manifest.save(save_path)
        logger.info("Saved resampled manifest to %s", save_path)

    # Stage 2: Mel spectrograms
    if run_mel:
        if resampled_manifest is None:
            # If resample was not run, try to load a previously saved manifest
            default_resampled = output_dir / "manifest_16k.csv"
            if default_resampled.exists():
                resampled_manifest = Manifest.load(default_resampled)
                logger.info(
                    "Loaded existing resampled manifest from %s", default_resampled
                )
            else:
                # Use the original manifest (assume audio is already 16 kHz)
                logger.warning(
                    "No resampled manifest found. Using original manifest. "
                    "Audio must already be at 16 kHz."
                )
                resampled_manifest = manifest

        pipeline.run_mel(
            input_manifest=resampled_manifest, skip_existing=args.skip_existing
        )

    # Verification
    if args.verify:
        verify_manifest = resampled_manifest if resampled_manifest is not None else manifest
        stats = pipeline.verify(verify_manifest, check_mel=run_mel)
        logger.info("Verification results: %s", stats)

        if stats.get("audio_missing", 0) > 0 or stats.get("mel_missing", 0) > 0:
            logger.warning(
                "Verification found missing files: audio=%d, mel=%d",
                stats.get("audio_missing", 0),
                stats.get("mel_missing", 0),
            )

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
