"""Train the streaming Destylizer via MSE distillation.

The streaming Destylizer uses an unfrozen, causal StreamingHuBERT and
causal Conformer blocks.  It is trained to match the offline Destylizer's
continuous content features via an MSE distillation loss.

Training spec (paper Section 10.8):
    - 100k steps, batch 32, AdamW, cosine annealing with 4k warmup
    - Peak LR 1e-4, HuBERT LR scale 0.1
    - L_distill = MSE(fc_streaming, fc_offline.detach())
    - Chunk size: 30 frames (600ms @ 50Hz)

Usage:
    uv run python scripts/train_streaming_destylizer.py \
        --config configs/streaming/distillation.yaml

    uv run python scripts/train_streaming_destylizer.py \
        --config configs/streaming/distillation.yaml \
        --resume outputs/streaming_destylizer/checkpoints/step_50000

    uv run python scripts/train_streaming_destylizer.py \
        --config configs/streaming/distillation.yaml \
        training.batch_size=16 training.steps=50000

    accelerate launch scripts/train_streaming_destylizer.py \
        --config configs/streaming/distillation.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(config_path: str | None, overrides: list[str]):
    """Load and merge the YAML config with CLI overrides.

    Uses OmegaConf to load the base YAML, then applies dot-notation
    overrides (e.g. ``training.batch_size=16``).

    If no config path is provided, returns a default config suitable
    for streaming Destylizer distillation training.

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
            "name": "streaming_destylizer",
            "output_dir": "outputs",
            "log_dir": "outputs/streaming_destylizer/logs",
            "distillation": {
                "teacher_checkpoint": "checkpoints/destylizer/best",
                "student_init_checkpoint": "",
                "hubert_lr_scale": 0.1,
                "aux_asr_weight": 0.0,
                "chunk_size": 30,
            },
            "conformer": {
                "num_layers": 6,
                "hidden_size": 768,
                "ffn_size": 3072,
                "num_heads": 12,
                "kernel_size": 31,
                "dropout": 0.1,
            },
            "hubert": {
                "model_id": "facebook/hubert-large-ls960-ft",
                "layer": 18,
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
                "val_interval": 2000,
                "betas": [0.9, 0.98],
                "weight_decay": 0.01,
                "seed": 42,
            },
            "data": {
                "name": "lmg",
                "sample_rate": 16000,
                "manifest_path": "data/manifests/lmg.csv",
                "audio_dir": "data/processed/audio",
                "hubert_features_dir": "data/processed/hubert",
                "val_manifest_path": None,
                "num_workers": 4,
            },
        })

    # Apply CLI overrides
    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, override_conf)

    # Allow adding new keys
    OmegaConf.set_struct(base, False)

    # Ensure required top-level fields have defaults
    if not hasattr(base, "output_dir"):
        base.output_dir = "outputs"
    if not hasattr(base, "log_dir"):
        base.log_dir = str(
            Path(base.output_dir) / "streaming_destylizer" / "logs"
        )
    if not hasattr(base, "name"):
        base.name = "streaming_destylizer"

    # Ensure distillation sub-config has required fields
    if hasattr(base, "distillation"):
        if not hasattr(base.distillation, "teacher_checkpoint"):
            base.distillation.teacher_checkpoint = "checkpoints/destylizer/best"
        if not hasattr(base.distillation, "student_init_checkpoint"):
            base.distillation.student_init_checkpoint = ""
        if not hasattr(base.distillation, "hubert_lr_scale"):
            base.distillation.hubert_lr_scale = 0.1
        if not hasattr(base.distillation, "aux_asr_weight"):
            base.distillation.aux_asr_weight = 0.0
        if not hasattr(base.distillation, "chunk_size"):
            base.distillation.chunk_size = 30
    else:
        base.distillation = OmegaConf.create({
            "teacher_checkpoint": "checkpoints/destylizer/best",
            "student_init_checkpoint": "",
            "hubert_lr_scale": 0.1,
            "aux_asr_weight": 0.0,
            "chunk_size": 30,
        })

    # Ensure data sub-config has required fields
    if hasattr(base, "data"):
        if not hasattr(base.data, "manifest_path"):
            base.data.manifest_path = "data/manifests/lmg.csv"
        if not hasattr(base.data, "audio_dir"):
            base.data.audio_dir = "data/processed/audio"
        if not hasattr(base.data, "hubert_features_dir"):
            base.data.hubert_features_dir = "data/processed/hubert"
        if not hasattr(base.data, "val_manifest_path"):
            base.data.val_manifest_path = None
        if not hasattr(base.data, "num_workers"):
            base.data.num_workers = 4
    else:
        base.data = OmegaConf.create({
            "manifest_path": "data/manifests/lmg.csv",
            "audio_dir": "data/processed/audio",
            "hubert_features_dir": "data/processed/hubert",
            "val_manifest_path": None,
            "num_workers": 4,
        })

    # Ensure training sub-config has required fields
    if hasattr(base, "training"):
        if not hasattr(base.training, "betas"):
            base.training.betas = [0.9, 0.98]
        if not hasattr(base.training, "weight_decay"):
            base.training.weight_decay = 0.01
    else:
        base.training = OmegaConf.create({
            "steps": 100000,
            "batch_size": 32,
            "peak_lr": 1e-4,
            "warmup_steps": 4000,
            "gradient_clip": 1.0,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
            "log_interval": 100,
            "save_interval": 10000,
            "val_interval": 2000,
            "betas": [0.9, 0.98],
            "weight_decay": 0.01,
            "seed": 42,
        })

    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the StyleStream streaming Destylizer via MSE distillation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train_streaming_destylizer.py \\\n"
            "      --config configs/streaming/distillation.yaml\n"
            "\n"
            "  uv run python scripts/train_streaming_destylizer.py \\\n"
            "      --config configs/streaming/distillation.yaml \\\n"
            "      --resume outputs/streaming_destylizer/checkpoints/step_50000\n"
            "\n"
            "  uv run python scripts/train_streaming_destylizer.py \\\n"
            "      --config configs/streaming/distillation.yaml \\\n"
            "      training.batch_size=16\n"
            "\n"
            "  accelerate launch scripts/train_streaming_destylizer.py \\\n"
            "      --config configs/streaming/distillation.yaml\n"
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
        print("Streaming Destylizer Distillation Configuration")
        print("=" * 60)
        print(OmegaConf.to_yaml(config, resolve=True))
        print("=" * 60)
    except Exception:
        print(f"Config: {config}")

    # --- Build trainer and run ---------------------------------------------
    from stylestream.streaming.distillation import DistillationTrainer

    trainer = DistillationTrainer(config)

    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        trainer.logger.info(
            "Training interrupted by user at step %d.", trainer.global_step
        )
    finally:
        trainer.finish()

    sys.exit(0)


if __name__ == "__main__":
    main()
