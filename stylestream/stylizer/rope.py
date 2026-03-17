"""Rotary Position Embedding (RoPE) for StyleStream Stylizer DiT.

Implements RoPE (Su et al., 2021) for the DiT's multi-head self-attention.
RoPE encodes absolute position information by rotating query and key vectors,
which naturally yields relative position awareness in attention scores.

StyleStream Stylizer spec:
    - 16 DiT layers, hidden_size 768, 12 heads, head_dim 64
    - 50 Hz frame rate, typical sequence length ~300 frames (6 seconds)
    - Position offset support for streaming inference

Reference:
    Su, Lu, Pan, Murtadha, Wen & Liu.  "RoFormer: Enhanced Transformer
    with Rotary Position Embedding."  Neurocomputing, 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding for DiT self-attention.

    Precomputes and caches cos/sin tables up to ``max_seq_len`` positions.
    The tables are stored as non-learnable buffers so they follow the module
    to the correct device automatically.

    Parameters
    ----------
    dim : int
        Dimension per head (e.g., 64 for hidden_size=768, num_heads=12).
    max_seq_len : int
        Maximum sequence length for precomputation cache. Default 4096.
    base : float
        Frequency base. Default 10000.0.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute and register as buffers
        cos, sin = self._build_cache(max_seq_len, dim, base)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    @torch.no_grad()
    def _build_cache(
        seq_len: int,
        dim: int,
        base: float,
    ) -> tuple[Tensor, Tensor]:
        """Build cos/sin lookup tables.

        Parameters
        ----------
        seq_len : int
            Number of positions to precompute.
        dim : int
            Head dimension (must be even).
        base : float
            Frequency base.

        Returns
        -------
        cos, sin : Tensor
            Each of shape ``(1, 1, seq_len, dim)``.
        """
        # Inverse frequencies: theta_i = base^(-2i / dim), i = 0 .. dim/2 - 1
        # Shape: (dim / 2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        # Position indices: (seq_len,)
        positions = torch.arange(seq_len, dtype=torch.float32)

        # Outer product -> (seq_len, dim / 2)
        angles = torch.outer(positions, inv_freq)

        # Duplicate to full dim: (seq_len, dim)
        # Each pair of adjacent dimensions shares the same angle
        angles = angles.repeat(1, 2)

        # Reshape to (1, 1, seq_len, dim) for broadcasting with
        # (B, num_heads, T, head_dim) attention tensors
        cos = angles.cos().unsqueeze(0).unsqueeze(0)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)

        return cos, sin

    def forward(
        self,
        x: Tensor,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Return (cos, sin) for positions ``[offset, offset + seq_len)``.

        The cos/sin tables are pre-allocated at ``__init__`` time for
        ``max_seq_len`` positions.  This method simply slices the
        pre-allocated buffers -- no conditional logic, no recomputation,
        making it compatible with ``torch.compile``.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, num_heads, T, head_dim)`` or
            ``(B, T, num_heads, head_dim)``.  Only used to determine
            ``seq_len`` and device/dtype.
        offset : int
            Position offset for streaming / causal inference. Default 0.

        Returns
        -------
        cos : Tensor
            Shape ``(1, 1, T, head_dim)``.
        sin : Tensor
            Shape ``(1, 1, T, head_dim)``.
        """
        # x can be (B, H, T, D) or (B, T, H, D); T is always dim -2
        seq_len = x.shape[-2]

        cos = self.cos_cached[:, :, offset:offset + seq_len, :].to(
            device=x.device, dtype=x.dtype
        )
        sin = self.sin_cached[:, :, offset:offset + seq_len, :].to(
            device=x.device, dtype=x.dtype
        )

        return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    """Split the last dimension in half and swap with a sign change.

    Given ``x = [x1, x2]`` along the last axis, returns ``[-x2, x1]``.

    Parameters
    ----------
    x : Tensor
        Shape ``(..., dim)`` where ``dim`` is even.

    Returns
    -------
    Tensor
        Same shape as input, with halves swapped and negated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Rotation formula::

        x_rot = x * cos + rotate_half(x) * sin

    Parameters
    ----------
    q, k : Tensor
        Shape ``(B, num_heads, T, head_dim)``.
    cos, sin : Tensor
        Shape ``(1, 1, T, head_dim)``, as returned by
        :meth:`RotaryPositionEmbedding.forward`.

    Returns
    -------
    q_rot, k_rot : Tensor
        Shape ``(B, num_heads, T, head_dim)``.
    """
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
