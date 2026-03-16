"""Streaming voice style conversion demo.

Processes source audio chunk-by-chunk, converting to target style.
Demonstrates the full StreamingInferencePipeline with timing stats.

Usage:
    uv run python scripts/streaming_inference.py \
        --source source.wav \
        --target target.wav \
        --output converted.wav \
        --destylizer-checkpoint checkpoints/streaming_destylizer/best \
        --stylizer-checkpoint checkpoints/streaming_stylizer/best \
        --vocoder-checkpoint checkpoints/vocoder/best

    uv run python scripts/streaming_inference.py \
        --source source.wav \
        --target target.wav \
        --output converted.wav \
        --destylizer-checkpoint checkpoints/streaming_destylizer/best \
        --stylizer-checkpoint checkpoints/streaming_stylizer/best \
        --vocoder-checkpoint checkpoints/vocoder/best \
        --chunk-size-ms 600 \
        --nfe 16 \
        --cfg-strength 2.0 \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio


def _load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load and resample audio to the target sample rate.

    Parameters
    ----------
    path : str
        Path to the audio file.
    sample_rate : int
        Target sample rate (default 16000).

    Returns
    -------
    Tensor
        Audio waveform of shape ``(T_samples,)``.
    """
    waveform, sr = torchaudio.load(path)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (T_samples,)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform


def _load_destylizer(checkpoint_path: str, device: torch.device):
    """Load the streaming Destylizer from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the streaming Destylizer checkpoint.
    device : torch.device
        Target device.

    Returns
    -------
    StreamingDestylizer
        The loaded model in eval mode.
    """
    from stylestream.streaming.destylizer import StreamingDestylizer

    model = StreamingDestylizer()

    checkpoint_path_obj = Path(checkpoint_path)
    if checkpoint_path_obj.exists():
        state_dict = StreamingDestylizer._load_state_dict(checkpoint_path_obj)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded streaming Destylizer from {checkpoint_path}")
    else:
        print(
            f"WARNING: Destylizer checkpoint not found at {checkpoint_path}. "
            f"Using randomly initialized model."
        )

    model = model.to(device)
    model.eval()
    return model


def _load_stylizer(checkpoint_path: str, device: torch.device):
    """Load the streaming Stylizer from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the streaming Stylizer checkpoint.
    device : torch.device
        Target device.

    Returns
    -------
    StreamingStylizer
        The loaded model in eval mode.
    """
    from stylestream.streaming.stylizer import StreamingStylizer

    model = StreamingStylizer()

    checkpoint_path_obj = Path(checkpoint_path)
    if checkpoint_path_obj.exists():
        state_dict = _load_state_dict_generic(checkpoint_path_obj)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded streaming Stylizer from {checkpoint_path}")
    else:
        print(
            f"WARNING: Stylizer checkpoint not found at {checkpoint_path}. "
            f"Using randomly initialized model."
        )

    model = model.to(device)
    model.eval()
    return model


def _load_vocoder(checkpoint_path: str, device: torch.device):
    """Load the causal Vocos vocoder from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the vocoder checkpoint.
    device : torch.device
        Target device.

    Returns
    -------
    nn.Module
        The loaded vocoder in eval mode.
    """
    from stylestream.vocoder.model import CausalVocos

    model = CausalVocos()

    checkpoint_path_obj = Path(checkpoint_path)
    if checkpoint_path_obj.exists():
        state_dict = _load_state_dict_generic(checkpoint_path_obj)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded vocoder from {checkpoint_path}")
    else:
        print(
            f"WARNING: Vocoder checkpoint not found at {checkpoint_path}. "
            f"Using randomly initialized model."
        )

    model = model.to(device)
    model.eval()
    return model


def _load_state_dict_generic(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load model weights from a checkpoint path.

    Supports:
    - Directory with ``model.safetensors`` (CheckpointManager format)
    - Single ``.safetensors`` file
    - Single ``.pt`` / ``.bin`` file (torch.load)
    """
    if checkpoint_path.is_dir():
        safetensors_path = checkpoint_path / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.torch import load_file
            return load_file(str(safetensors_path))

        pt_files = list(checkpoint_path.glob("*.pt"))
        if pt_files:
            state = torch.load(
                pt_files[0], map_location="cpu", weights_only=True,
            )
            if isinstance(state, dict) and "model" in state:
                return state["model"]
            return state

        raise FileNotFoundError(
            f"No model weights found in checkpoint directory: "
            f"{checkpoint_path}"
        )

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(checkpoint_path))

    state = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=True,
    )
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming voice style conversion demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/streaming_inference.py \\\n"
            "      --source source.wav \\\n"
            "      --target target.wav \\\n"
            "      --output converted.wav \\\n"
            "      --destylizer-checkpoint "
            "checkpoints/streaming_destylizer/best \\\n"
            "      --stylizer-checkpoint "
            "checkpoints/streaming_stylizer/best \\\n"
            "      --vocoder-checkpoint "
            "checkpoints/vocoder/best\n"
            "\n"
            "  uv run python scripts/streaming_inference.py \\\n"
            "      --source source.wav \\\n"
            "      --target target.wav \\\n"
            "      --output converted.wav \\\n"
            "      --destylizer-checkpoint "
            "checkpoints/streaming_destylizer/best \\\n"
            "      --stylizer-checkpoint "
            "checkpoints/streaming_stylizer/best \\\n"
            "      --vocoder-checkpoint "
            "checkpoints/vocoder/best \\\n"
            "      --chunk-size-ms 600 \\\n"
            "      --nfe 16 \\\n"
            "      --cfg-strength 2.0 \\\n"
            "      --device cuda\n"
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source audio file (WAV, 16kHz mono).",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target audio file (WAV, 16kHz mono, ~5 seconds).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted output audio (WAV).",
    )
    parser.add_argument(
        "--destylizer-checkpoint",
        type=str,
        required=True,
        help="Path to streaming Destylizer checkpoint.",
    )
    parser.add_argument(
        "--stylizer-checkpoint",
        type=str,
        required=True,
        help="Path to streaming Stylizer checkpoint.",
    )
    parser.add_argument(
        "--vocoder-checkpoint",
        type=str,
        required=True,
        help="Path to causal Vocos vocoder checkpoint.",
    )
    parser.add_argument(
        "--chunk-size-ms",
        type=int,
        default=600,
        help="Chunk size in milliseconds (default 600).",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=16,
        help="Number of function evaluations for CFM (default 16).",
    )
    parser.add_argument(
        "--cfg-strength",
        type=float,
        default=2.0,
        help="CFG guidance strength (default 2.0).",
    )
    parser.add_argument(
        "--max-source-seconds",
        type=float,
        default=5.0,
        help="Maximum source buffer duration in seconds (default 5.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default 16000).",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    sample_rate = args.sample_rate

    print("=" * 60)
    print("StyleStream Streaming Inference")
    print("=" * 60)
    print(f"  Source:      {args.source}")
    print(f"  Target:      {args.target}")
    print(f"  Output:      {args.output}")
    print(f"  Device:      {device}")
    print(f"  Chunk size:  {args.chunk_size_ms}ms")
    print(f"  NFE:         {args.nfe}")
    print(f"  CFG:         {args.cfg_strength}")
    print(f"  Max source:  {args.max_source_seconds}s")
    print("=" * 60)

    # --- Load audio --------------------------------------------------------
    print("\nLoading audio...")
    t0 = time.monotonic()
    source_waveform = _load_audio(args.source, sample_rate)
    target_waveform = _load_audio(args.target, sample_rate)
    print(
        f"  Source: {source_waveform.shape[-1]} samples "
        f"({source_waveform.shape[-1] / sample_rate:.2f}s)"
    )
    print(
        f"  Target: {target_waveform.shape[-1]} samples "
        f"({target_waveform.shape[-1] / sample_rate:.2f}s)"
    )
    print(f"  Audio loading: {(time.monotonic() - t0) * 1000:.1f}ms")

    # --- Load models -------------------------------------------------------
    print("\nLoading models...")
    t0 = time.monotonic()
    destylizer = _load_destylizer(args.destylizer_checkpoint, device)
    stylizer = _load_stylizer(args.stylizer_checkpoint, device)
    vocoder = _load_vocoder(args.vocoder_checkpoint, device)
    print(f"  Model loading: {(time.monotonic() - t0) * 1000:.1f}ms")

    # --- Build pipeline ----------------------------------------------------
    from stylestream.streaming.pipeline import StreamingInferencePipeline

    pipeline = StreamingInferencePipeline(
        destylizer=destylizer,
        stylizer=stylizer,
        vocoder=vocoder,
        chunk_size_ms=args.chunk_size_ms,
        sample_rate=sample_rate,
        nfe=args.nfe,
        cfg_strength=args.cfg_strength,
        max_source_seconds=args.max_source_seconds,
        device=args.device,
    )

    # --- Run conversion ----------------------------------------------------
    print("\nRunning streaming conversion...")
    converted, stats = pipeline.convert_file(
        source_waveform=source_waveform,
        target_waveform=target_waveform,
    )

    # --- Save output -------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure output is on CPU and 2-D for torchaudio.save
    converted_cpu = converted.cpu()
    if converted_cpu.dim() == 1:
        converted_cpu = converted_cpu.unsqueeze(0)

    torchaudio.save(str(output_path), converted_cpu, sample_rate)
    print(f"\nOutput saved to {output_path}")

    # --- Print timing stats ------------------------------------------------
    print("\n" + "=" * 60)
    print("Timing Statistics")
    print("=" * 60)
    print(f"  Number of chunks:   {stats['num_chunks']}")
    print(f"  Total time:         {stats['total_time_ms']:.1f}ms")
    print(f"  Real-time factor:   {stats['rtf']:.4f}")

    chunk_times = stats["chunk_times_ms"]
    if chunk_times:
        avg_chunk = sum(chunk_times) / len(chunk_times)
        min_chunk = min(chunk_times)
        max_chunk = max(chunk_times)
        print(f"  Avg chunk time:     {avg_chunk:.1f}ms")
        print(f"  Min chunk time:     {min_chunk:.1f}ms")
        print(f"  Max chunk time:     {max_chunk:.1f}ms")

    audio_duration_ms = source_waveform.shape[-1] / sample_rate * 1000
    print(f"  Audio duration:     {audio_duration_ms:.1f}ms")

    is_realtime = stats["rtf"] < 1.0
    print(f"  Real-time capable:  {'YES' if is_realtime else 'NO'}")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
