"""Download pre-trained models for StyleStream.

Usage:
    python scripts/download_models.py --stage train
    python scripts/download_models.py --stage eval
    python scripts/download_models.py --stage all
    python scripts/download_models.py --verify
    python scripts/download_models.py --list
    python scripts/download_models.py --list --stage eval
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for StyleStream.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_models.py --stage train\n"
            "  python scripts/download_models.py --stage all --cache-dir /data/models\n"
            "  python scripts/download_models.py --verify\n"
            "  python scripts/download_models.py --list --stage eval\n"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "all"],
        default="all",
        help="Which models to download: 'train', 'eval', or 'all' (default: all).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override HuggingFace cache directory.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check which models are already cached and exit.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="Print the model registry and exit.",
    )
    args = parser.parse_args()

    from stylestream.utils.hub import download_all, list_models, verify_cache

    # --list: show registry and exit
    if args.list_models:
        models = list_models(stage=args.stage if args.stage != "all" else None)
        print(f"{'Key':<16} {'Stage':<8} {'HF ID':<45} Description")
        print("-" * 110)
        for m in models:
            print(f"{m['key']:<16} {m['stage']:<8} {m['hf_id']:<45} {m['description']}")
        return

    # --verify: check cache status and exit
    if args.verify:
        status = verify_cache(cache_dir=args.cache_dir)
        print(f"{'Model':<16} {'Cached':<8}")
        print("-" * 24)
        for key, cached in status.items():
            mark = "yes" if cached else "no"
            print(f"{key:<16} {mark:<8}")
        all_cached = all(status.values())
        sys.exit(0 if all_cached else 1)

    # Default: download models
    paths = download_all(stage=args.stage, cache_dir=args.cache_dir)
    print(f"\nDownloaded {len(paths)} model(s):")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
