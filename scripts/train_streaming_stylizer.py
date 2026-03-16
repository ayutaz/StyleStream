"""Train the streaming Stylizer with chunked causal attention.

The streaming Stylizer uses a 16-layer DiT with chunked causal attention
and Conditional Flow Matching (CFM).  Style conditioning is provided by
a WavLM-TDNN encoder via adaLN-Zero.  The chunked causal attention mask
restricts each frame to attend only within its current and past chunks,
enabling real-time streaming inference.

Training uses the same spectrogram inpainting objective as the offline
Stylizer but with chunked causal attention masks applied to all DiT
self-attention layers.

Training spec (paper Section 10.8):
    - 400k steps, batch 64, AdamW, cosine annealing with 2k warmup
    - Peak LR 1e-4, betas (0.9, 0.999), weight decay 0.01
    - Gradient clip 1.0, bf16 mixed precision
    - Chunk size: 30 frames (600ms @ 50Hz)

Usage:
    uv run python scripts/train_streaming_stylizer.py \
        --config configs/streaming/stylizer.yaml

    uv run python scripts/train_streaming_stylizer.py \
        --config configs/streaming/stylizer.yaml \
        --resume outputs/streaming_stylizer/checkpoints/step_10000

    uv run python scripts/train_streaming_stylizer.py \
        --config configs/streaming/stylizer.yaml \
        training.batch_size=8 training.steps=50000

    accelerate launch scripts/train_streaming_stylizer.py \
        --config configs/streaming/stylizer.yaml
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
    for streaming Stylizer training.

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
            "name": "streaming_stylizer",
            "output_dir": "outputs",
            "log_dir": "outputs/streaming_stylizer/logs",
            "streaming": {
                "chunk_size_ms": 600,
                "chunk_size_frames": 30,
                "max_cache_frames": 250,
            },
            "dit": {
                "num_layers": 16,
                "hidden_size": 768,
                "ffn_size": 3072,
                "num_heads": 12,
                "adaln_zero": True,
                "dropout": 0.0,
                "attention_type": "chunked_causal",
            },
            "mel": {
                "n_mels": 100,
                "hop_length": 320,
                "n_fft": 1024,
                "sample_rate": 16000,
                "f_min": 0,
                "f_max": 8000,
            },
            "style_encoder": {
                "model_id": "microsoft/wavlm-base-plus-sv",
                "hidden_size": 768,
                "num_layers": 13,
                "pooling": "attentive_statistics",
                "frozen": True,
            },
            "cfm": {
                "nfe": 16,
                "sampling": "euler",
                "sigma_min": 1e-5,
            },
            "cfg": {
                "strength": 2.0,
                "content_drop": 0.2,
                "context_drop": 0.3,
                "style_drop": 0.3,
            },
            "mask": {
                "ratio_min": 0.7,
                "ratio_max": 1.0,
                "type": "contiguous",
            },
            "training": {
                "steps": 400000,
                "batch_size": 64,
                "peak_lr": 1e-4,
                "warmup_steps": 2000,
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
                "name": "emilia_en",
                "sample_rate": 16000,
                "manifest_path": "data/manifests/emilia.csv",
                "mel_dir": "data/processed/mel",
                "content_features_dir": "data/processed/streaming_content_features",
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
            Path(base.output_dir) / "streaming_stylizer" / "logs"
        )
    if not hasattr(base, "name"):
        base.name = "streaming_stylizer"

    # Ensure streaming sub-config has required fields
    if hasattr(base, "streaming"):
        if not hasattr(base.streaming, "chunk_size_ms"):
            base.streaming.chunk_size_ms = 600
        if not hasattr(base.streaming, "chunk_size_frames"):
            base.streaming.chunk_size_frames = 30
        if not hasattr(base.streaming, "max_cache_frames"):
            base.streaming.max_cache_frames = 250
    else:
        base.streaming = OmegaConf.create({
            "chunk_size_ms": 600,
            "chunk_size_frames": 30,
            "max_cache_frames": 250,
        })

    # Ensure data sub-config has required fields
    if hasattr(base, "data"):
        if not hasattr(base.data, "manifest_path"):
            base.data.manifest_path = "data/manifests/emilia.csv"
        if not hasattr(base.data, "mel_dir"):
            base.data.mel_dir = "data/processed/mel"
        if not hasattr(base.data, "content_features_dir"):
            base.data.content_features_dir = (
                "data/processed/streaming_content_features"
            )
        if not hasattr(base.data, "val_manifest_path"):
            base.data.val_manifest_path = None
        if not hasattr(base.data, "num_workers"):
            base.data.num_workers = 4
    else:
        base.data = OmegaConf.create({
            "manifest_path": "data/manifests/emilia.csv",
            "mel_dir": "data/processed/mel",
            "content_features_dir": "data/processed/streaming_content_features",
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
            "steps": 400000,
            "batch_size": 64,
            "peak_lr": 1e-4,
            "warmup_steps": 2000,
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

    # Enable chunked causal attention in the DiT config
    if hasattr(base, "dit"):
        base.dit.attention_type = "chunked_causal"
    else:
        base.dit = OmegaConf.create({
            "num_layers": 16,
            "hidden_size": 768,
            "ffn_size": 3072,
            "num_heads": 12,
            "adaln_zero": True,
            "dropout": 0.0,
            "attention_type": "chunked_causal",
        })

    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the StyleStream streaming Stylizer "
            "(DiT + CFM with chunked causal attention)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train_streaming_stylizer.py \\\n"
            "      --config configs/streaming/stylizer.yaml\n"
            "\n"
            "  uv run python scripts/train_streaming_stylizer.py \\\n"
            "      --config configs/streaming/stylizer.yaml \\\n"
            "      --resume outputs/streaming_stylizer/checkpoints/step_10000\n"
            "\n"
            "  uv run python scripts/train_streaming_stylizer.py \\\n"
            "      --config configs/streaming/stylizer.yaml \\\n"
            "      training.batch_size=8\n"
            "\n"
            "  accelerate launch scripts/train_streaming_stylizer.py \\\n"
            "      --config configs/streaming/stylizer.yaml\n"
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
        print("Streaming Stylizer Training Configuration")
        print("=" * 60)
        print(OmegaConf.to_yaml(config, resolve=True))
        print("=" * 60)
    except Exception:
        print(f"Config: {config}")

    # --- Build trainer and run ---------------------------------------------
    # The streaming Stylizer uses the same StylizerTrainer but with
    # chunked causal attention enabled via the config. The trainer
    # detects dit.attention_type == "chunked_causal" and builds a
    # StreamingStylizer model instead of the offline Stylizer.
    from stylestream.stylizer.trainer import StylizerTrainer

    trainer = StylizerTrainer(config)

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
