"""Preprocess raw audio data for StyleStream training.

Handles:
  - Audio resampling to 16 kHz mono
  - Silence removal / VAD segmentation
  - Mel spectrogram pre-computation (optional)
  - Manifest (metadata) generation

Usage:
    python scripts/preprocess_data.py --input-dir raw/libritts --output-dir data/libritts
    python scripts/preprocess_data.py --config configs/preprocess.yaml
    python scripts/preprocess_data.py --input-dir raw/ --output-dir data/ --num-workers 8
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess audio data for StyleStream training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/preprocess_data.py --input-dir raw/libritts --output-dir data/libritts\n"
            "  python scripts/preprocess_data.py --config configs/preprocess.yaml\n"
            "  python scripts/preprocess_data.py --input-dir raw/ --output-dir data/ --num-workers 8\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (Hydra / OmegaConf).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing raw audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write preprocessed files and manifests.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--precompute-mel",
        action="store_true",
        help="Pre-compute and cache mel spectrograms.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides.",
    )

    parser.parse_args()
    print("preprocess_data: Not yet implemented.")
    sys.exit(0)


if __name__ == "__main__":
    main()
