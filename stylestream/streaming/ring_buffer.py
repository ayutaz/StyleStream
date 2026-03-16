"""Ring buffer and streaming context for StyleStream streaming inference.

Manages the FIFO ring buffer for accumulating source content features
during chunk-by-chunk streaming inference, and the StreamingContext
that constructs Stylizer inputs from cached target + buffered source.

StyleStream spec:
    - Ring buffer: max 250 frames (5 seconds @ 50Hz)
    - FIFO eviction when buffer full
    - Target utterance: cached once at initialization (5 seconds)
    - Source features: accumulated chunk by chunk (30 frames = 600ms)
    - Stylizer input: [target(250) | source(up to 250)] = max 500 frames
"""

from __future__ import annotations

import torch
from torch import Tensor


class RingBuffer:
    """FIFO ring buffer for streaming feature accumulation.

    Stores fixed-size feature frames and evicts the oldest when
    capacity is exceeded. Used to manage source content features
    and mel spectrograms during streaming inference.

    Parameters
    ----------
    max_frames : int
        Maximum number of frames to store (default 250 = 5s @ 50Hz).
    feature_dim : int
        Feature dimension per frame (e.g. 768 for content features,
        100 for mel spectrograms).
    device : torch.device or None
        Device for stored tensors.
    """

    def __init__(
        self,
        max_frames: int = 250,
        feature_dim: int = 768,
        device: torch.device | None = None,
    ) -> None:
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        self.device = device
        self._buffer: Tensor | None = None  # (1, T_current, feature_dim)
        self._length: int = 0

    def append(self, features: Tensor) -> None:
        """Append new frames to the buffer.

        If the resulting buffer would exceed ``max_frames``, the oldest
        frames are evicted (FIFO). If the incoming chunk itself is longer
        than ``max_frames``, only the last ``max_frames`` frames are kept.

        Parameters
        ----------
        features : Tensor
            New frames of shape ``(B, T_new, feature_dim)`` or
            ``(T_new, feature_dim)``.  ``B`` must be 1 for streaming.

        Raises
        ------
        ValueError
            If the batch dimension is not 1, or if ``feature_dim``
            does not match the buffer's configured dimension.
        """
        # Normalise to 3-D: (1, T_new, feature_dim)
        if features.ndim == 2:
            features = features.unsqueeze(0)

        if features.ndim != 3:
            raise ValueError(
                f"Expected 2-D or 3-D tensor, got {features.ndim}-D"
            )

        if features.shape[0] != 1:
            raise ValueError(
                f"Streaming requires batch size 1, got {features.shape[0]}"
            )

        if features.shape[2] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: buffer expects {self.feature_dim}, "
                f"got {features.shape[2]}"
            )

        # Move to buffer device if needed.
        if self.device is not None:
            features = features.to(self.device)

        if self._buffer is None:
            self._buffer = features
        else:
            self._buffer = torch.cat([self._buffer, features], dim=1)

        # Trim to max_frames (keep newest frames).
        if self._buffer.shape[1] > self.max_frames:
            self._buffer = self._buffer[:, -self.max_frames:, :]

        self._length = self._buffer.shape[1]

    def get(self) -> Tensor | None:
        """Get all buffered frames.

        Returns
        -------
        Tensor or None
            Shape ``(1, T_current, feature_dim)``, or ``None`` if empty.
        """
        return self._buffer

    @property
    def length(self) -> int:
        """Current number of frames in buffer."""
        return self._length

    @property
    def is_full(self) -> bool:
        """Whether buffer has reached ``max_frames``."""
        return self._length >= self.max_frames

    @property
    def is_empty(self) -> bool:
        """Whether buffer has no frames."""
        return self._length == 0

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer = None
        self._length = 0


class StreamingContext:
    """Manages all streaming inference state.

    Holds the target utterance cache, source ring buffer, and
    constructs the Stylizer input for each chunk.

    The Stylizer input at each step is the concatenation of:

    - **Target portion** (fixed, cached at init): ``target_mel`` and
      ``target_content`` of length ``T_target``.  Unmasked (``False``).
    - **Source portion** (accumulated via ring buffer): up to
      ``max_source_frames`` frames.  Fully masked (``True``).

    Parameters
    ----------
    target_mel : Tensor
        Target mel spectrogram ``(1, T_target, 100)``. Cached once.
    target_content : Tensor
        Target content features ``(1, T_target, 768)``. Cached once.
    style_embedding : Tensor
        Style embedding ``(1, 768)``. Extracted once from target.
    max_source_frames : int
        Ring buffer capacity (default 250).
    frame_rate : float
        Frame rate in Hz (default 50.0).
    """

    def __init__(
        self,
        target_mel: Tensor,
        target_content: Tensor,
        style_embedding: Tensor,
        max_source_frames: int = 250,
        frame_rate: float = 50.0,
    ) -> None:
        self.target_mel = target_mel
        self.target_content = target_content
        self.style_embedding = style_embedding
        self.frame_rate = frame_rate

        self._num_chunks_processed: int = 0

        self.source_content_buffer = RingBuffer(
            max_frames=max_source_frames,
            feature_dim=target_content.shape[-1],  # 768
            device=target_content.device,
        )
        self.source_mel_buffer = RingBuffer(
            max_frames=max_source_frames,
            feature_dim=target_mel.shape[-1],  # 100
            device=target_mel.device,
        )

    def add_source_chunk(
        self,
        content_features: Tensor,  # (1, chunk_frames, 768)
    ) -> None:
        """Add a new source chunk to the ring buffers.

        Appends the content features to the source content ring buffer.
        The source mel buffer is padded with zeros (the source mel is
        unknown and will be generated by the Stylizer).

        Parameters
        ----------
        content_features : Tensor
            Source content features of shape ``(1, chunk_frames, 768)``
            or ``(chunk_frames, 768)``.
        """
        # Normalise to 3-D for consistent shape access.
        if content_features.ndim == 2:
            content_features = content_features.unsqueeze(0)

        chunk_frames = content_features.shape[1]
        device = content_features.device

        self.source_content_buffer.append(content_features)

        # Pad source mel with zeros (unknown, to be generated).
        mel_dim = self.target_mel.shape[-1]
        zero_mel = torch.zeros(
            1, chunk_frames, mel_dim, device=device, dtype=self.target_mel.dtype
        )
        self.source_mel_buffer.append(zero_mel)

        self._num_chunks_processed += 1

    def build_stylizer_input(self) -> dict[str, Tensor | int]:
        """Build the full Stylizer input from target + source buffers.

        Concatenates the cached target features with the accumulated
        source features from the ring buffers, and constructs the
        inpainting mask (``False`` for target, ``True`` for source).

        Returns
        -------
        dict
            ``'content_features'`` : Tensor
                ``(1, T_target + T_source, 768)``
            ``'context_mel'`` : Tensor
                ``(1, T_target + T_source, mel_dim)``
            ``'mask'`` : Tensor
                ``(1, T_target + T_source)`` bool --
                ``False`` for target, ``True`` for source.
            ``'style_embedding'`` : Tensor
                ``(1, 768)``
            ``'source_start_idx'`` : int
                Index where source frames begin.
            ``'source_length'`` : int
                Number of source frames.
        """
        device = self.target_content.device
        target_T = self.target_mel.shape[1]

        source_content = self.source_content_buffer.get()
        source_mel = self.source_mel_buffer.get()
        source_T = source_content.shape[1] if source_content is not None else 0

        total_T = target_T + source_T

        # --- Content features: concat target + source ---
        if source_content is not None:
            content = torch.cat([self.target_content, source_content], dim=1)
        else:
            content = self.target_content

        # --- Context mel: target mel for target portion, zeros for source ---
        if source_T > 0 and source_mel is not None:
            context_mel = torch.cat([self.target_mel, source_mel], dim=1)
        else:
            context_mel = self.target_mel

        # --- Mask: False (unmasked) for target, True (masked) for source ---
        mask = torch.zeros(1, total_T, dtype=torch.bool, device=device)
        if source_T > 0:
            mask[:, target_T:] = True

        return {
            "content_features": content,
            "context_mel": context_mel,
            "mask": mask,
            "style_embedding": self.style_embedding,
            "source_start_idx": target_T,
            "source_length": source_T,
        }

    @property
    def num_chunks_processed(self) -> int:
        """Number of source chunks processed so far."""
        return self._num_chunks_processed

    def reset_source(self) -> None:
        """Reset source buffers (keep target cache)."""
        self.source_content_buffer.reset()
        self.source_mel_buffer.reset()
        self._num_chunks_processed = 0
