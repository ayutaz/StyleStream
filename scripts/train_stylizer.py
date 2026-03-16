"""Train the Stylizer component.

The Stylizer uses a 16-layer DiT with Conditional Flow Matching (CFM) and
classifier-free guidance.  Style conditioning is provided by a WavLM-TDNN
encoder via adaLN-Zero.  Trained with a spectrogram inpainting objective.

Usage:
    python scripts/train_stylizer.py
    python scripts/train_stylizer.py --config configs/stylizer.yaml
    python scripts/train_stylizer.py stylizer.dit.num_layers=16 training.batch_size=8
    accelerate launch scripts/train_stylizer.py
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the StyleStream Stylizer (DiT + CFM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train_stylizer.py --config configs/stylizer.yaml\n"
            "  python scripts/train_stylizer.py training.batch_size=8\n"
            "  accelerate launch scripts/train_stylizer.py --config configs/stylizer.yaml\n"
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
        help="Hydra-style config overrides (e.g. stylizer.dit.num_layers=16).",
    )

    parser.parse_args()
    print("train_stylizer: Not yet implemented.")
    sys.exit(0)


if __name__ == "__main__":
    main()
