"""Chunked causal attention mask utilities for StyleStream streaming.

Generates block lower-triangular masks for chunked causal attention.
Each chunk attends to itself and all past chunks, blocking future chunks.

StyleStream spec:
    - Default chunk size: 30 frames (600ms @ 50Hz)
    - Configurable: 10/20/30/40/50 frames (200/400/600/800/1000ms)
    - Compatible with ALiBi (Destylizer) and RoPE (Stylizer)
"""

from __future__ import annotations

import torch
from torch import Tensor

from stylestream.destylizer.alibi import get_alibi_slopes


def build_chunked_causal_mask(
    seq_len: int,
    chunk_size: int = 30,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """Generate a block lower-triangular attention mask for chunked causal attention.

    The mask divides the sequence into chunks of ``chunk_size`` frames.
    Within a chunk, all frames can attend to each other (full attention).
    Each chunk can attend to all *past* chunks but not future chunks.

    For ``seq_len=9, chunk_size=3``::

        1 1 1 | 0 0 0 | 0 0 0
        1 1 1 | 0 0 0 | 0 0 0
        1 1 1 | 0 0 0 | 0 0 0
        ------+-------+------
        1 1 1 | 1 1 1 | 0 0 0
        1 1 1 | 1 1 1 | 0 0 0
        1 1 1 | 1 1 1 | 0 0 0
        ------+-------+------
        1 1 1 | 1 1 1 | 1 1 1
        1 1 1 | 1 1 1 | 1 1 1
        1 1 1 | 1 1 1 | 1 1 1

    Parameters
    ----------
    seq_len : int
        Total sequence length in frames.
    chunk_size : int
        Number of frames per chunk.  Default 30 (600 ms @ 50 Hz).
    device : torch.device or None
        Target device.
    dtype : torch.dtype
        Data type of the output mask.  Default ``torch.bool``.

    Returns
    -------
    Tensor
        Bool mask of shape ``(seq_len, seq_len)`` where ``True`` means the
        query position is allowed to attend to the key position.
    """
    # When chunk_size covers the whole sequence, return all-True (full attention).
    if chunk_size >= seq_len:
        return torch.ones(seq_len, seq_len, device=device, dtype=dtype)

    # Assign each position to its chunk index: chunk_idx[i] = i // chunk_size
    chunk_idx = torch.arange(seq_len, device=device).div(chunk_size, rounding_mode="floor")

    # query_chunk[i] >= key_chunk[j]  <=>  position i can attend to position j
    # This creates the block lower-triangular structure.
    mask = chunk_idx.unsqueeze(1) >= chunk_idx.unsqueeze(0)  # (S, S)

    return mask.to(dtype=dtype)


def chunked_causal_mask_to_attn_bias(
    mask: Tensor,
    num_heads: int | None = None,
) -> Tensor:
    """Convert a bool attention mask to an additive attention bias.

    Allowed positions receive bias ``0.0``; blocked positions receive
    ``-inf`` so they are zeroed out after softmax.

    Parameters
    ----------
    mask : Tensor
        Bool mask of shape ``(seq_len, seq_len)`` where ``True`` = allowed.
    num_heads : int or None
        Unused.  Kept for API symmetry; the returned tensor broadcasts
        over both batch and head dimensions.

    Returns
    -------
    Tensor
        Float tensor of shape ``(1, 1, seq_len, seq_len)`` suitable for
        adding to attention scores of shape ``(B, H, S, S)``.
    """
    bias = torch.zeros_like(mask, dtype=torch.float32)
    bias.masked_fill_(~mask, float("-inf"))
    return bias.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)


def build_chunked_causal_alibi_bias(
    seq_len: int,
    chunk_size: int,
    num_heads: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build ALiBi positional bias combined with chunked causal masking.

    This is used by the Conformer in the Destylizer during streaming mode.
    ALiBi distances are computed from *global* (absolute) positions, not
    chunk-local positions.  Future chunks receive ``-inf`` bias.

    Parameters
    ----------
    seq_len : int
        Total sequence length in frames.
    chunk_size : int
        Number of frames per chunk.
    num_heads : int
        Number of attention heads (e.g. 12).
    device : torch.device or None
        Target device.
    dtype : torch.dtype
        Data type.  Default ``torch.float32``.

    Returns
    -------
    Tensor
        Shape ``(1, num_heads, seq_len, seq_len)``.
    """
    # 1. Build ALiBi distance-based bias using global positions.
    #    slopes: (num_heads,) on CPU
    slopes = get_alibi_slopes(num_heads)

    positions = torch.arange(seq_len, device=device, dtype=dtype)
    # rel_dist[i, j] = i - j  (signed distance)
    rel_dist = positions.unsqueeze(1) - positions.unsqueeze(0)  # (S, S)
    abs_dist = rel_dist.abs()  # (S, S)

    # slopes -> (H, 1, 1) for broadcasting
    slopes_dev = slopes.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
    # ALiBi bias: -slope * |i - j|  (bidirectional distance within allowed region)
    alibi_bias = -slopes_dev * abs_dist.unsqueeze(0)  # (H, S, S)

    # 2. Build chunked causal mask.
    mask = build_chunked_causal_mask(seq_len, chunk_size, device=device)  # (S, S) bool

    # 3. Set blocked positions (future chunks) to -inf.
    alibi_bias[:, ~mask] = float("-inf")

    # Add batch dimension: (1, H, S, S)
    return alibi_bias.unsqueeze(0)
