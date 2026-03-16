"""Train the Destylizer component.

The Destylizer extracts content features from speech using HuBERT-Large
layer 18, a 6-layer Conformer, and FSQ [5,3,3] quantisation.  It is
trained with an ASR loss (CTC or seq2seq cross-entropy) on the LMG
dataset (~1,300 hours of speech).

Training spec (paper Section 10.8):
    - 100k steps, batch 32, AdamW, cosine annealing with 4k warmup
    - Peak LR 1e-4, betas (0.9, 0.98), weight decay 0.01
    - Gradient clip 1.0, bf16 mixed precision

Usage:
    uv run python scripts/train_destylizer.py \\
        --config configs/destylizer/offline.yaml

    uv run python scripts/train_destylizer.py \\
        --config configs/destylizer/offline.yaml \\
        --resume outputs/destylizer/checkpoints/step_50000

    uv run python scripts/train_destylizer.py \\
        --config configs/destylizer/offline.yaml \\
        training.batch_size=16 training.steps=50000

    accelerate launch scripts/train_destylizer.py \\
        --config configs/destylizer/offline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(config_path: str | None, overrides: list[str]):
    """Load and merge the YAML config with CLI overrides.

    Uses OmegaConf to load the base YAML, then applies dot-notation
    overrides (e.g. ``training.batch_size=16``).

    If no config path is provided, returns a default config from
    :mod:`stylestream.config`.

    Parameters
    ----------
    config_path :
        Path to a YAML configuration file.
    overrides :
        List of Hydra-style ``key=value`` overrides.

    Returns
    -------
    DictConfig
        Merged configuration.
    """
    from omegaconf import OmegaConf

    if config_path is not None:
        base = OmegaConf.load(config_path)
    else:
        # Minimal default config
        base = OmegaConf.create({
            "name": "destylizer",
            "output_dir": "outputs",
            "log_dir": "outputs/destylizer/logs",
            "conformer": {
                "num_layers": 6,
                "hidden_size": 768,
                "ffn_size": 3072,
                "num_heads": 12,
                "kernel_size": 31,
                "dropout": 0.1,
            },
            "fsq": {
                "levels": [5, 3, 3],
                "down_dim": 3,
                "up_dim": 768,
            },
            "asr_decoder": {
                "num_layers": 4,
                "hidden_size": 768,
                "ffn_size": 3072,
                "num_heads": 12,
                "dropout": 0.1,
                "vocab_size": 30,
                "loss_type": "ctc",
                "label_smoothing": 0.1,
            },
            "training": {
                "steps": 100000,
                "batch_size": 32,
                "peak_lr": 1e-4,
                "warmup_steps": 4000,
                "gradient_clip": 1.0,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "bf16",
                "log_interval": 100,
                "save_interval": 10000,
                "val_interval": 5000,
                "betas": [0.9, 0.98],
                "weight_decay": 0.01,
            },
            "data": {
                "name": "lmg",
                "sample_rate": 16000,
                "manifest_path": "data/manifests/lmg.csv",
                "features_dir": "data/processed",
                "val_manifest_path": None,
                "num_workers": 4,
                "max_frames": 3000,
            },
        })

    # Apply CLI overrides
    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, override_conf)

    # Ensure required fields have defaults
    if not OmegaConf.is_missing(base, "name"):
        pass
    else:
        base.name = "destylizer"

    OmegaConf.set_struct(base, False)

    if not hasattr(base, "output_dir"):
        base.output_dir = "outputs"
    if not hasattr(base, "log_dir"):
        base.log_dir = str(Path(base.output_dir) / "destylizer" / "logs")
    if not hasattr(base, "name"):
        base.name = "destylizer"

    # Ensure data sub-config has required fields
    if hasattr(base, "data"):
        if not hasattr(base.data, "manifest_path"):
            base.data.manifest_path = "data/manifests/lmg.csv"
        if not hasattr(base.data, "features_dir"):
            base.data.features_dir = "data/processed"
        if not hasattr(base.data, "val_manifest_path"):
            base.data.val_manifest_path = None
        if not hasattr(base.data, "num_workers"):
            base.data.num_workers = 4
        if not hasattr(base.data, "max_frames"):
            base.data.max_frames = 3000
    else:
        base.data = OmegaConf.create({
            "manifest_path": "data/manifests/lmg.csv",
            "features_dir": "data/processed",
            "val_manifest_path": None,
            "num_workers": 4,
            "max_frames": 3000,
        })

    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the StyleStream Destylizer (HuBERT + Conformer + FSQ).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train_destylizer.py \\\n"
            "      --config configs/destylizer/offline.yaml\n"
            "\n"
            "  uv run python scripts/train_destylizer.py \\\n"
            "      --config configs/destylizer/offline.yaml \\\n"
            "      --resume outputs/destylizer/checkpoints/step_50000\n"
            "\n"
            "  uv run python scripts/train_destylizer.py \\\n"
            "      --config configs/destylizer/offline.yaml \\\n"
            "      training.batch_size=16\n"
            "\n"
            "  accelerate launch scripts/train_destylizer.py \\\n"
            "      --config configs/destylizer/offline.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (OmegaConf format).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    # Allow Hydra-style overrides as positional args
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-notation config overrides (e.g. training.batch_size=16).",
    )

    args = parser.parse_args()

    # --- Load config -------------------------------------------------------
    config = _load_config(args.config, args.overrides)

    # --- Print config summary ----------------------------------------------
    try:
        from omegaconf import OmegaConf
        print("=" * 60)
        print("Destylizer Training Configuration")
        print("=" * 60)
        print(OmegaConf.to_yaml(config, resolve=True))
        print("=" * 60)
    except Exception:
        print(f"Config: {config}")

    # --- Build trainer and run ---------------------------------------------
    from stylestream.destylizer.trainer import DestylizerTrainer

    trainer = DestylizerTrainer(config)

    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        trainer.logger.info("Training interrupted by user at step %d.", trainer.global_step)
    finally:
        trainer.finish()

    sys.exit(0)


if __name__ == "__main__":
    main()
