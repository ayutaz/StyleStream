"""Run StyleStream voice style conversion inference.

Supports both offline (full-utterance) and streaming modes.

Usage:
    python scripts/inference.py --source source.wav --reference ref.wav --output out.wav
    python scripts/inference.py --source source.wav --reference ref.wav --streaming
    python scripts/inference.py --config configs/inference.yaml
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run StyleStream voice style conversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/inference.py --source s.wav --reference r.wav -o out.wav\n"
            "  python scripts/inference.py --source s.wav --reference r.wav --streaming\n"
            "  python scripts/inference.py --config configs/inference.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (Hydra / OmegaConf).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to the source audio file (content provider).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to the reference audio file (style provider).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.wav",
        help="Path for the converted output audio (default: output.wav).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained model checkpoint directory.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming (chunked-causal) inference mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides.",
    )

    parser.parse_args()
    print("inference: Not yet implemented.")
    sys.exit(0)


if __name__ == "__main__":
    main()
