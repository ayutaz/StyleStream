"""ALiBi (Attention with Linear Biases) for StyleStream Destylizer.

Implements ALiBi positional encoding (Press et al., 2022) for Conformer
multi-head self-attention.  ALiBi replaces explicit positional embeddings
with a simple linear bias added to attention scores, improving length
generalization without learned parameters.

StyleStream Destylizer spec:
    - 12 attention heads, head_dim 64 (768 / 12)
    - 6 Conformer blocks sharing the same ALiBi slopes
    - Non-causal (bidirectional) during offline training
    - Causal with chunked attention during streaming (Phase 5)

Reference:
    Press, Smith & Lewis.  "Train Short, Test Long: Attention with Linear
    Biases Enables Input Length Extrapolation."  ICLR 2022.
"""

from __future__ import annotations

import functools
import math

import torch


# ---------------------------------------------------------------------------
# ALiBi bias cache
# ---------------------------------------------------------------------------

_alibi_cache: dict[tuple, torch.Tensor] = {}


def clear_alibi_cache() -> None:
    """Clear the ALiBi bias cache.

    Call this when switching devices, dtypes, or to free memory.
    """
    _alibi_cache.clear()


# ---------------------------------------------------------------------------
# Slope computation
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Return ALiBi per-head slopes.

    For *n* heads that are a power of two the slopes are::

        2^{-8/n},  2^{-16/n},  ...,  2^{-8}

    When *n* is **not** a power of two the algorithm computes slopes for
    the next power of two, takes every other element to fill the first
    half, then interleaves slopes from twice that size for the remainder.

    Parameters
    ----------
    num_heads : int
        Number of attention heads (e.g. 12 for the Destylizer).

    Returns
    -------
    torch.Tensor
        Shape ``(num_heads,)`` on CPU, dtype ``float32``.
    """

    def _slopes_power_of_2(n: int) -> list[float]:
        # ratio = 2^(-8/n); slopes = ratio^1, ratio^2, ..., ratio^n
        ratio = 2.0 ** (-8.0 / n)
        return [ratio ** i for i in range(1, n + 1)]

    if _is_power_of_2(num_heads):
        slopes = _slopes_power_of_2(num_heads)
    else:
        # Canonical ALiBi for non-power-of-2 head counts:
        # 1. Get all slopes from the closest *smaller* power-of-2
        # 2. Fill the remaining heads with every-other slope from 2x that size
        closest_pow2 = 2 ** math.floor(math.log2(num_heads))
        base_slopes = _slopes_power_of_2(closest_pow2)
        remaining = num_heads - closest_pow2
        # Extra slopes from double the resolution, taking every other one
        # (indices 1, 3, 5, ...) so they interleave with the base set.
        extra_slopes = _slopes_power_of_2(2 * closest_pow2)
        extra = [extra_slopes[i] for i in range(1, 2 * remaining, 2)]
        slopes = base_slopes + extra

    return torch.tensor(slopes, dtype=torch.float32)


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# Bias matrix construction
# ---------------------------------------------------------------------------

def build_alibi_bias(
    seq_len: int,
    num_heads: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    causal: bool = False,
) -> torch.Tensor:
    """Build the ALiBi attention-score bias matrix.

    Results are cached by ``(seq_len, num_heads, device, dtype, causal)``
    so that repeated calls with the same arguments (e.g. across layers in
    the same forward pass) return the cached tensor without recomputation.

    The returned tensor can be added directly to the raw attention logits
    **before** softmax::

        attn_scores = Q @ K^T / sqrt(d_k)     # (B, H, S, S)
        attn_scores = attn_scores + alibi_bias  # broadcasts over batch

    Parameters
    ----------
    seq_len : int
        Sequence (time) length in frames.
    num_heads : int
        Number of attention heads.
    device : torch.device or None
        Target device.  ``None`` keeps the tensor on CPU.
    dtype : torch.dtype
        Data type of the output bias tensor.  Default ``float32``.
    causal : bool
        If ``False`` (default, offline training) the bias is
        ``-slope * |i - j|`` for all positions.
        If ``True`` (streaming) future positions (``j > i``) are masked
        with ``-inf`` and the bias for valid positions is
        ``-slope * (i - j)``.

    Returns
    -------
    torch.Tensor
        Shape ``(1, num_heads, seq_len, seq_len)`` – broadcastable over
        the batch dimension of attention scores.
    """

    cache_key = (seq_len, num_heads, str(device), str(dtype), causal)
    if cache_key in _alibi_cache:
        return _alibi_cache[cache_key]

    # slopes: (num_heads,) on CPU
    slopes = get_alibi_slopes(num_heads)

    # Position indices
    # arange on the target device avoids a later .to() copy
    positions = torch.arange(seq_len, device=device, dtype=dtype)

    # Relative distance matrix: (seq_len, seq_len)
    # rel_dist[i, j] = i - j
    rel_dist = positions.unsqueeze(1) - positions.unsqueeze(0)

    if causal:
        # For causal attention keep only past-and-present (i >= j).
        # bias = -slope * (i - j) for i >= j, else -inf
        # Use torch.where to avoid an unnecessary .clone()
        future_mask = rel_dist < 0
        distance = torch.where(future_mask, torch.zeros_like(rel_dist), rel_dist)

        # slopes -> (num_heads, 1, 1) for broadcasting
        slopes_dev = slopes.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        bias = -slopes_dev * distance.unsqueeze(0)  # (H, S, S)

        # Apply causal mask (-inf for future positions)
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
        bias = torch.where(future_mask.unsqueeze(0), neg_inf, bias)
    else:
        # Bidirectional: bias = -slope * |i - j|
        abs_dist = rel_dist.abs()  # (S, S)
        slopes_dev = slopes.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        bias = -slopes_dev * abs_dist.unsqueeze(0)  # (H, S, S)

    # Add batch dimension: (1, H, S, S)
    result = bias.unsqueeze(0)
    _alibi_cache[cache_key] = result
    return result
