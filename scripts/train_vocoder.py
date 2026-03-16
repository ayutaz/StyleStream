"""Train the Causal Vocos vocoder.

Warm-starts from the official Vocos checkpoint and replaces ConvNext blocks
with causal convolutions for streaming inference.

Usage:
    python scripts/train_vocoder.py
    python scripts/train_vocoder.py --config configs/vocoder.yaml
    python scripts/train_vocoder.py vocoder.causal=true training.batch_size=32
    accelerate launch scripts/train_vocoder.py
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the StyleStream Causal Vocos vocoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train_vocoder.py --config configs/vocoder.yaml\n"
            "  python scripts/train_vocoder.py training.batch_size=32\n"
            "  accelerate launch scripts/train_vocoder.py --config configs/vocoder.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (Hydra / OmegaConf).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides (e.g. vocoder.causal=true).",
    )

    parser.parse_args()
    print("train_vocoder: Not yet implemented.")
    sys.exit(0)


if __name__ == "__main__":
    main()
