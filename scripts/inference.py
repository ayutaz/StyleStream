"""Run StyleStream voice style conversion inference.

Supports both offline (full-utterance) and streaming modes.
Single-file mode converts one source with one reference style.
Batch mode converts all pairs from an evaluation CSV.

Usage:
    # Single file conversion (offline)
    python scripts/inference.py --source source.wav --reference ref.wav --output out.wav

    # Single file conversion (streaming)
    python scripts/inference.py --source source.wav --reference ref.wav --streaming

    # Batch conversion from evaluation pairs CSV
    python scripts/inference.py --batch pairs.csv --output-dir converted/ \\
        --destylizer-checkpoint checkpoints/destylizer/best \\
        --stylizer-checkpoint checkpoints/stylizer/best \\
        --vocoder-checkpoint checkpoints/vocoder/best

    # With YAML config for checkpoint paths
    python scripts/inference.py --batch pairs.csv --output-dir converted/ \\
        --config configs/eval/stylestream_test.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torchaudio


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        level=level,
    )


def load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file, convert to mono, resample to target rate.

    Parameters
    ----------
    path : str
        Path to the audio file.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    Tensor
        Waveform shape (samples,) at the target sample rate.
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0)


def run_single(args: argparse.Namespace) -> None:
    """Single file conversion mode.

    Converts a single source audio using a reference style and saves
    the result to the specified output path.
    """
    logger = logging.getLogger(__name__)

    if not args.source or not args.reference:
        logger.error("--source and --reference are required for single-file mode.")
        sys.exit(1)

    source = load_audio(args.source)
    reference = load_audio(args.reference)

    logger.info("Source: %s (%.2fs)", args.source, len(source) / 16000)
    logger.info("Reference: %s (%.2fs)", args.reference, len(reference) / 16000)

    from stylestream.eval.batch_inference import BatchInference

    # Determine output directory from the output file path
    output_path = Path(args.output)
    output_dir = str(output_path.parent) if output_path.parent != Path() else "."

    pipeline = BatchInference(
        destylizer_checkpoint=args.destylizer_checkpoint or "",
        stylizer_checkpoint=args.stylizer_checkpoint or "",
        vocoder_checkpoint=args.vocoder_checkpoint or "",
        output_dir=output_dir,
        device=args.device,
        use_streaming=args.streaming,
        nfe=args.nfe,
        cfg_strength=args.cfg_strength,
    )

    start = time.time()
    converted = pipeline.convert_pair(source, reference)
    elapsed = time.time() - start

    torchaudio.save(args.output, converted.unsqueeze(0), 16000)

    audio_duration = len(source) / 16000
    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
    logger.info(
        "Saved to %s (%.2fs processing, RTF=%.3f)", args.output, elapsed, rtf
    )


def run_batch(args: argparse.Namespace) -> None:
    """Batch conversion mode.

    Converts all source-target pairs from an evaluation CSV and saves
    the results to the specified output directory.
    """
    logger = logging.getLogger(__name__)

    if not args.batch:
        logger.error("--batch is required for batch mode.")
        sys.exit(1)

    from stylestream.eval.batch_inference import BatchInference
    from stylestream.eval.dataset import EvalDataset

    dataset = EvalDataset(args.batch)
    logger.info("Loaded %d pairs from %s", len(dataset), args.batch)

    output_dir = args.output_dir or "eval_results/converted"

    pipeline = BatchInference(
        destylizer_checkpoint=args.destylizer_checkpoint or "",
        stylizer_checkpoint=args.stylizer_checkpoint or "",
        vocoder_checkpoint=args.vocoder_checkpoint or "",
        output_dir=output_dir,
        device=args.device,
        use_streaming=args.streaming,
        nfe=args.nfe,
        cfg_strength=args.cfg_strength,
    )

    stats = pipeline.run(list(dataset))
    logger.info("Batch inference stats: %s", stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run StyleStream voice style conversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single file (offline)\n"
            "  python scripts/inference.py --source s.wav --reference r.wav -o out.wav\n"
            "\n"
            "  # Single file (streaming)\n"
            "  python scripts/inference.py --source s.wav --reference r.wav --streaming\n"
            "\n"
            "  # Batch mode\n"
            "  python scripts/inference.py --batch pairs.csv --output-dir converted/\n"
        ),
    )

    # Mode selection
    parser.add_argument(
        "--source", type=str, default=None,
        help="Source audio file path (single-file mode).",
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Reference/target audio file path for style (single-file mode).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output.wav",
        help="Output path for single file mode (default: output.wav).",
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Pairs CSV for batch mode (from eval dataset).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for batch mode (default: eval_results/converted).",
    )

    # Model checkpoints
    parser.add_argument(
        "--destylizer-checkpoint", type=str, default=None,
        help="Path to Destylizer checkpoint directory or file.",
    )
    parser.add_argument(
        "--stylizer-checkpoint", type=str, default=None,
        help="Path to Stylizer checkpoint directory or file.",
    )
    parser.add_argument(
        "--vocoder-checkpoint", type=str, default=None,
        help="Path to Vocoder checkpoint directory or file.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file with checkpoint paths (eval section).",
    )

    # Inference settings
    parser.add_argument(
        "--streaming", action="store_true",
        help="Use streaming (chunked-causal) inference mode.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--nfe", type=int, default=16,
        help="Number of CFM Euler ODE steps (default: 16).",
    )
    parser.add_argument(
        "--cfg-strength", type=float, default=2.0,
        help="CFG guidance strength (default: 2.0).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load config if provided, fill in missing checkpoint paths
    if args.config:
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(args.config)
            # Support eval section with checkpoint paths
            eval_cfg = getattr(cfg, "eval", cfg)
            if not args.destylizer_checkpoint and hasattr(
                eval_cfg, "destylizer_checkpoint"
            ):
                args.destylizer_checkpoint = eval_cfg.destylizer_checkpoint
            if not args.stylizer_checkpoint and hasattr(
                eval_cfg, "stylizer_checkpoint"
            ):
                args.stylizer_checkpoint = eval_cfg.stylizer_checkpoint
            if not args.vocoder_checkpoint and hasattr(
                eval_cfg, "vocoder_checkpoint"
            ):
                args.vocoder_checkpoint = eval_cfg.vocoder_checkpoint
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Could not load config from %s: %s", args.config, e
            )

    # Route to the appropriate mode
    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
