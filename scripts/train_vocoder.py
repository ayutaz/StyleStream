"""Train the Causal Vocos vocoder.

Warm-starts from the official Vocos checkpoint and finetunes with
causal convolutions on LibriTTS.

Training spec (paper Section 10.8):
    - 500k steps, batch 64, AdamW, cosine annealing with 4k warmup
    - Peak LR 1e-4, betas (0.9, 0.999), weight decay 0.01
    - Gradient clip 1.0, bf16 mixed precision
    - Multi-scale discriminator GAN training

Usage:
    uv run python scripts/train_vocoder.py \
        --config configs/vocoder/causal_vocos.yaml

    uv run python scripts/train_vocoder.py \
        --config configs/vocoder/causal_vocos.yaml \
        --resume outputs/vocoder/checkpoints/step_50000

    uv run python scripts/train_vocoder.py \
        --config configs/vocoder/causal_vocos.yaml \
        training.batch_size=8 training.steps=50000

    accelerate launch scripts/train_vocoder.py \
        --config configs/vocoder/causal_vocos.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(config_path: str | None, overrides: list[str]):
    """Load and merge the YAML config with CLI overrides.

    Uses OmegaConf to load the base YAML, then applies dot-notation
    overrides (e.g. ``training.batch_size=8``).

    If no config path is provided, returns a default config suitable
    for Vocoder training.

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
            "name": "vocoder",
            "output_dir": "outputs",
            "log_dir": "outputs/vocoder/logs",
            "backbone": {
                "hidden_size": 512,
                "num_layers": 8,
                "intermediate_size": 1536,
                "kernel_size": 7,
                "causal": True,
            },
            "mel": {
                "n_mels": 100,
                "hop_length": 320,
                "n_fft": 1024,
                "sample_rate": 16000,
                "f_min": 0,
                "f_max": 8000,
            },
            "discriminator": {
                "type": "multi_scale",
                "scales": [1, 2, 4],
                "channels": 64,
            },
            "loss": {
                "reconstruction": 45.0,
                "gan_generator": 1.0,
                "gan_discriminator": 1.0,
                "feature_matching": 2.0,
            },
            "init_checkpoint": "charactr/vocos-mel-24khz",
            "causal": True,
            "training": {
                "steps": 500000,
                "batch_size": 64,
                "peak_lr": 1e-4,
                "warmup_steps": 4000,
                "gradient_clip": 1.0,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "bf16",
                "log_interval": 100,
                "save_interval": 10000,
                "val_interval": 5000,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "seed": 42,
            },
            "data": {
                "name": "libritts",
                "sample_rate": 16000,
                "manifest_path": "data/manifests/libritts.csv",
                "mel_dir": "data/processed/mel",
                "audio_dir": "data/processed/audio",
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
        base.log_dir = str(Path(base.output_dir) / "vocoder" / "logs")
    if not hasattr(base, "name"):
        base.name = "vocoder"

    # Ensure data sub-config has required fields
    if hasattr(base, "data"):
        if not hasattr(base.data, "manifest_path"):
            base.data.manifest_path = "data/manifests/libritts.csv"
        if not hasattr(base.data, "mel_dir"):
            base.data.mel_dir = "data/processed/mel"
        if not hasattr(base.data, "audio_dir"):
            base.data.audio_dir = "data/processed/audio"
        if not hasattr(base.data, "val_manifest_path"):
            base.data.val_manifest_path = None
        if not hasattr(base.data, "num_workers"):
            base.data.num_workers = 4
    else:
        base.data = OmegaConf.create({
            "manifest_path": "data/manifests/libritts.csv",
            "mel_dir": "data/processed/mel",
            "audio_dir": "data/processed/audio",
            "val_manifest_path": None,
            "num_workers": 4,
        })

    # Ensure training sub-config has required fields
    if hasattr(base, "training"):
        if not hasattr(base.training, "betas"):
            base.training.betas = [0.9, 0.999]
        if not hasattr(base.training, "weight_decay"):
            base.training.weight_decay = 0.01
    else:
        base.training = OmegaConf.create({
            "steps": 500000,
            "batch_size": 64,
            "peak_lr": 1e-4,
            "warmup_steps": 4000,
            "gradient_clip": 1.0,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
            "log_interval": 100,
            "save_interval": 10000,
            "val_interval": 5000,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "seed": 42,
        })

    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the StyleStream Causal Vocos vocoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train_vocoder.py \\\n"
            "      --config configs/vocoder/causal_vocos.yaml\n"
            "\n"
            "  uv run python scripts/train_vocoder.py \\\n"
            "      --config configs/vocoder/causal_vocos.yaml \\\n"
            "      --resume outputs/vocoder/checkpoints/step_50000\n"
            "\n"
            "  uv run python scripts/train_vocoder.py \\\n"
            "      --config configs/vocoder/causal_vocos.yaml \\\n"
            "      training.batch_size=8\n"
            "\n"
            "  accelerate launch scripts/train_vocoder.py \\\n"
            "      --config configs/vocoder/causal_vocos.yaml\n"
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
        help="Dot-notation config overrides (e.g. training.batch_size=8).",
    )

    args = parser.parse_args()

    # --- Load config -------------------------------------------------------
    config = _load_config(args.config, args.overrides)

    # --- Print config summary ----------------------------------------------
    try:
        from omegaconf import OmegaConf
        print("=" * 60)
        print("Vocoder Training Configuration")
        print("=" * 60)
        print(OmegaConf.to_yaml(config, resolve=True))
        print("=" * 60)
    except Exception:
        print(f"Config: {config}")

    # --- Build trainer and run ---------------------------------------------
    from stylestream.vocoder.trainer import VocoderTrainer

    trainer = VocoderTrainer(config)

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
