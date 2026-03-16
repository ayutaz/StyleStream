"""End-to-end streaming inference pipeline for StyleStream.

Orchestrates chunk-by-chunk voice style conversion:
    1. Target initialization: mel + style embedding + content features (cached once)
    2. Per-chunk: Destylizer -> ring buffer -> Stylizer (CFM) -> Vocoder -> waveform

StyleStream spec:
    - Chunk size: 600ms (9600 samples @ 16kHz, 30 frames @ 50Hz)
    - Target: 5 seconds, cached at initialization
    - Source ring buffer: max 5 seconds (250 frames)
    - CFM Euler sampling: NFE=16
    - CFG: alpha=2
    - End-to-end latency: chunk_size + processing_time ~ 1 second
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stylestream.streaming.ring_buffer import StreamingContext
from stylestream.utils.mel import MelSpectrogramTransform

logger = logging.getLogger(__name__)


class StreamingInferencePipeline:
    """End-to-end streaming voice style conversion pipeline.

    Processes source audio chunk-by-chunk, converting it to match
    the target speaker's style in real time.

    Initialization:
        1. Compute target mel spectrogram (cached)
        2. Extract target style embedding via WavLM-TDNN (cached)
        3. Extract target content features via Destylizer (cached)

    Per-chunk processing:
        1. Destylizer: extract source content features
        2. Build Stylizer input (target context + source buffer)
        3. Stylizer: CFM Euler sampling -> mel spectrogram
        4. Vocoder: mel -> waveform

    Parameters
    ----------
    destylizer : StreamingDestylizer
        Streaming Destylizer model.
    stylizer : StreamingStylizer
        Streaming Stylizer model.
    vocoder : CausalVocos
        Causal Vocos vocoder.
    chunk_size_ms : int
        Chunk size in milliseconds (default 600).
    sample_rate : int
        Audio sample rate (default 16000).
    nfe : int
        Number of function evaluations for CFM Euler sampling (default 16).
    cfg_strength : float
        CFG guidance strength (default 2.0).
    max_source_seconds : float
        Max source buffer duration (default 5.0).
    device : torch.device or str
        Device for inference.
    """

    def __init__(
        self,
        destylizer: nn.Module,
        stylizer: nn.Module,
        vocoder: nn.Module,
        chunk_size_ms: int = 600,
        sample_rate: int = 16000,
        nfe: int = 16,
        cfg_strength: float = 2.0,
        max_source_seconds: float = 5.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.destylizer = destylizer
        self.stylizer = stylizer
        self.vocoder = vocoder
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.nfe = nfe
        self.cfg_strength = cfg_strength
        self.device = torch.device(device)

        # Derived constants
        self.chunk_samples = int(chunk_size_ms * sample_rate / 1000)  # 9600
        self.hop_length = 320
        self.chunk_frames = chunk_size_ms * 50 // 1000  # 30 frames
        self.max_source_frames = int(max_source_seconds * 50)  # 250

        # Mel transform for target initialization
        self.mel_transform = MelSpectrogramTransform()

        # State (set by initialize_target)
        self._context: StreamingContext | None = None
        self._initialized = False

        logger.info(
            "StreamingInferencePipeline: chunk=%dms (%d samples, %d frames), "
            "nfe=%d, cfg_strength=%.1f, max_source=%.1fs (%d frames), "
            "device=%s",
            chunk_size_ms,
            self.chunk_samples,
            self.chunk_frames,
            nfe,
            cfg_strength,
            max_source_seconds,
            self.max_source_frames,
            self.device,
        )

    # ------------------------------------------------------------------
    # Target initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_target(self, target_waveform: Tensor) -> None:
        """Initialize with target utterance (5 seconds).

        Caches target mel, style embedding, and content features.
        Must be called before process_chunk().

        Parameters
        ----------
        target_waveform : Tensor
            Target audio at 16kHz, shape (T_samples,) or (1, T_samples).
        """
        # Ensure (1, T_samples) shape
        if target_waveform.dim() == 1:
            target_waveform = target_waveform.unsqueeze(0)
        target_waveform = target_waveform.to(self.device)

        # 1. Compute target mel spectrogram
        mel_transform = self.mel_transform.to(self.device)
        target_mel = mel_transform(target_waveform)  # (1, n_mels, T_frames)
        # Transpose to (1, T_frames, n_mels) for Stylizer convention
        target_mel = target_mel.transpose(1, 2)  # (1, T_frames, 100)

        # 2. Extract style embedding via Stylizer's style encoder
        style_embedding = self.stylizer.style_encoder(
            target_waveform
        )  # (1, 768)

        # 3. Extract target content features via Destylizer
        target_content = self.destylizer.extract_content_features(
            target_waveform
        )  # (1, T_frames, 768)

        # Align content and mel frame counts (take the shorter)
        T_mel = target_mel.shape[1]
        T_content = target_content.shape[1]
        T_target = min(T_mel, T_content)
        target_mel = target_mel[:, :T_target, :]
        target_content = target_content[:, :T_target, :]

        # 4. Create streaming context
        self._context = StreamingContext(
            target_mel=target_mel,
            target_content=target_content,
            style_embedding=style_embedding,
            max_source_frames=self.max_source_frames,
        )

        self._initialized = True

        logger.info(
            "Target initialized: mel=%s, content=%s, style=%s",
            list(target_mel.shape),
            list(target_content.shape),
            list(style_embedding.shape),
        )

    # ------------------------------------------------------------------
    # Per-chunk processing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_chunk(self, source_chunk: Tensor) -> Tensor:
        """Process one source audio chunk.

        Parameters
        ----------
        source_chunk : Tensor
            Source audio chunk at 16kHz, shape (chunk_samples,) or
            (1, chunk_samples). Expected length: 9600 samples (600ms).

        Returns
        -------
        Tensor
            Converted audio chunk, shape (chunk_samples,).

        Raises
        ------
        RuntimeError
            If initialize_target() has not been called.
        """
        if not self._initialized:
            raise RuntimeError(
                "Pipeline not initialized. Call initialize_target() first."
            )

        # Ensure correct shape
        if source_chunk.dim() == 1:
            source_chunk = source_chunk.unsqueeze(0)
        source_chunk = source_chunk.to(self.device)

        # 1. Extract source content features via streaming Destylizer
        content = self.destylizer.extract_content_features(
            source_chunk
        )  # (1, T, 768)

        # 2. Add to ring buffer
        self._context.add_source_chunk(content)

        # 3. Build Stylizer input (target context + source buffer)
        stylizer_input = self._context.build_stylizer_input()

        content_features = stylizer_input["content_features"]  # (1, T_total, 768)
        context_mel = stylizer_input["context_mel"]  # (1, T_total, 100)
        mask = stylizer_input["mask"]  # (1, T_total) bool
        style_embedding = stylizer_input["style_embedding"]  # (1, 768)
        source_length = stylizer_input["source_length"]

        # Convert bool mask to float for CFM operations
        mask_float = mask.float()

        # 4. Run Stylizer CFM sampling with CFG
        # Build velocity function with classifier-free guidance
        def velocity_fn(
            x_t: Tensor, t_step: Tensor
        ) -> Tensor:
            return self.stylizer.cfg.guided_velocity(
                velocity_fn=lambda xt, ti, c, ctx, s: self.stylizer.dit(
                    xt, ti, c, ctx, s
                ),
                x_t=x_t,
                t=t_step,
                content_features=content_features,
                context_mel=context_mel,
                style_emb=style_embedding,
                guidance_strength=self.cfg_strength,
            )

        # Euler ODE sampling: x_0 (noise) -> x_1 (mel)
        T_total = content_features.shape[1]
        mel_dim = context_mel.shape[-1]
        shape = (1, T_total, mel_dim)

        generated_mel = self.stylizer.cfm.euler_sample(
            velocity_fn=velocity_fn,
            shape=shape,
            nfe=self.nfe,
            device=self.device,
            dtype=content_features.dtype,
        )  # (1, T_total, mel_dim)

        # Inpainting blend: keep target mel in unmasked region
        mask_expanded = mask_float.unsqueeze(-1)  # (1, T_total, 1)
        blended_mel = (
            mask_expanded * generated_mel
            + (1.0 - mask_expanded) * context_mel
        )  # (1, T_total, mel_dim)

        # 5. Extract the last chunk_frames from the source region
        # The source region occupies the final source_length frames.
        # We want only the most recent chunk_frames of the generated mel.
        chunk_frames_to_take = min(self.chunk_frames, source_length)
        mel_chunk = blended_mel[:, -chunk_frames_to_take:, :]  # (1, chunk_frames, mel_dim)

        # 6. Vocoder: mel -> waveform
        # CausalVocos expects (B, n_mels, T) channels-first layout
        mel_for_vocoder = mel_chunk.transpose(1, 2)  # (1, mel_dim, chunk_frames)
        waveform_chunk = self.vocoder(mel_for_vocoder)  # (1, T_samples)

        # 7. Trim or pad to exactly chunk_samples
        if waveform_chunk.shape[-1] >= self.chunk_samples:
            waveform_chunk = waveform_chunk[:, :self.chunk_samples]
        else:
            waveform_chunk = F.pad(
                waveform_chunk,
                (0, self.chunk_samples - waveform_chunk.shape[-1]),
            )

        return waveform_chunk.squeeze(0)  # (chunk_samples,)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset source state (keep target cache).

        Clears the source ring buffer so that subsequent calls to
        process_chunk() start from a clean slate. The cached target
        mel, content features, and style embedding are preserved.
        """
        if self._context is not None:
            self._context.reset_source()

    # ------------------------------------------------------------------
    # Convenience: full-file conversion
    # ------------------------------------------------------------------

    @torch.no_grad()
    def convert_file(
        self,
        source_waveform: Tensor,
        target_waveform: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Convert a full file chunk-by-chunk.

        Convenience method that initializes target, then processes
        source chunk by chunk, collecting results.

        Parameters
        ----------
        source_waveform : Tensor
            Full source audio at 16kHz.
        target_waveform : Tensor
            Target audio at 16kHz (5 seconds).

        Returns
        -------
        converted : Tensor
            Full converted audio.
        stats : dict
            'total_time_ms': total processing time
            'chunk_times_ms': list of per-chunk times
            'rtf': real-time factor (processing_time / audio_duration)
            'num_chunks': number of chunks processed
        """
        self.initialize_target(target_waveform)

        # Ensure (1, T) shape for splitting
        if source_waveform.dim() == 1:
            source_waveform = source_waveform.unsqueeze(0)

        num_samples = source_waveform.shape[-1]
        chunks: list[Tensor] = []
        chunk_times: list[float] = []

        for start in range(0, num_samples, self.chunk_samples):
            chunk = source_waveform[:, start:start + self.chunk_samples]

            # Pad last chunk if shorter than chunk_samples
            if chunk.shape[-1] < self.chunk_samples:
                chunk = F.pad(
                    chunk, (0, self.chunk_samples - chunk.shape[-1])
                )

            t0 = time.monotonic()
            output = self.process_chunk(chunk)
            chunk_times.append((time.monotonic() - t0) * 1000)

            chunks.append(output)

        # Concatenate all output chunks and trim to original length
        converted = torch.cat(chunks, dim=-1)[:num_samples]

        total_time = sum(chunk_times)
        audio_duration_ms = num_samples / self.sample_rate * 1000

        stats = {
            "total_time_ms": total_time,
            "chunk_times_ms": chunk_times,
            "rtf": total_time / audio_duration_ms if audio_duration_ms > 0 else 0.0,
            "num_chunks": len(chunks),
        }

        logger.info(
            "Converted %d chunks in %.1fms (RTF=%.3f, audio=%.1fms)",
            stats["num_chunks"],
            stats["total_time_ms"],
            stats["rtf"],
            audio_duration_ms,
        )

        return converted, stats

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """Whether target has been initialized."""
        return self._initialized

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  chunk_size_ms={self.chunk_size_ms}, "
            f"chunk_samples={self.chunk_samples}, "
            f"chunk_frames={self.chunk_frames}\n"
            f"  nfe={self.nfe}, cfg_strength={self.cfg_strength}\n"
            f"  max_source_frames={self.max_source_frames}\n"
            f"  device={self.device}\n"
            f"  initialized={self._initialized}\n"
            f")"
        )
