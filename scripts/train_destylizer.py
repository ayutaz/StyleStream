"""Train the Destylizer component.

The Destylizer extracts content features from speech using HuBERT-Large
layer 18, a 6-layer Conformer, and FSQ [5,3,3] quantisation.  It is
trained with an ASR loss.

Usage:
    python scripts/train_destylizer.py
    python scripts/train_destylizer.py --config configs/destylizer.yaml
    python scripts/train_destylizer.py destylizer.fsq.levels=[7,5,5] training.batch_size=16
    accelerate launch scripts/train_destylizer.py
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the StyleStream Destylizer (HuBERT + Conformer + FSQ).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train_destylizer.py --config configs/destylizer.yaml\n"
            "  python scripts/train_destylizer.py training.batch_size=16\n"
            "  accelerate launch scripts/train_destylizer.py --config configs/destylizer.yaml\n"
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
    # Allow Hydra-style overrides as positional args
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides (e.g. training.batch_size=16).",
    )

    parser.parse_args()
    print("train_destylizer: Not yet implemented.")
    sys.exit(0)


if __name__ == "__main__":
    main()
