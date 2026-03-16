"""Chunked causal multi-head attention for StyleStream streaming.

Provides a unified attention module supporting:
    - Full attention (offline training)
    - Chunked causal attention (streaming training/inference)
    - KV cache for incremental inference
    - ALiBi and RoPE positional encodings
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stylestream.streaming.attention_mask import (
    build_chunked_causal_alibi_bias,
    build_chunked_causal_mask,
    chunked_causal_mask_to_attn_bias,
)
from stylestream.stylizer.rope import apply_rotary_pos_emb


class ChunkedCausalMultiHeadAttention(nn.Module):
    """Multi-head attention with chunked causal masking for streaming.

    Supports both training (full sequence with chunked mask) and
    incremental inference (new chunk + KV cache from past chunks).

    Parameters
    ----------
    hidden_size : int
        Model dimension (768).
    num_heads : int
        Number of attention heads (12).
    dropout : float
        Attention dropout (default 0.1).
    chunk_size : int or None
        Chunk size in frames. None = full attention. Default 30.
    pos_encoding : str
        Position encoding type: ``"alibi"``, ``"rope"``, or ``"none"``.
        Default ``"none"``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        chunk_size: int | None = 30,
        pos_encoding: str = "none",
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.pos_encoding = pos_encoding
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass with optional KV cache for incremental decoding.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, T, hidden_size)``.
        padding_mask : Tensor or None
            Bool mask of shape ``(B, T)`` where ``True`` = valid (non-pad).
            Used in training mode to mask padded positions.
        kv_cache : tuple[Tensor, Tensor] or None
            Past key and value tensors, each ``(B, H, T_past, D)``.
            ``None`` for training (full sequence) mode.
        rope_cos : Tensor or None
            RoPE cosine table of shape ``(1, 1, T, head_dim)``.
            Provided externally when ``pos_encoding == "rope"``.
        rope_sin : Tensor or None
            RoPE sine table of shape ``(1, 1, T, head_dim)``.

        Returns
        -------
        output : Tensor
            Shape ``(B, T, hidden_size)``.
        new_kv_cache : tuple[Tensor, Tensor]
            Updated (K, V) including current chunk,
            each ``(B, H, T_total, D)``.
        """
        B, T, _ = x.shape

        # --- Project to Q, K, V ---
        qkv = self.qkv_proj(x)  # (B, T, 3 * hidden_size)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(dim=0)  # each (B, H, T, D)

        # --- Apply RoPE if provided ---
        if rope_cos is not None and rope_sin is not None:
            # For incremental mode, the caller must supply cos/sin tables
            # already offset to the correct global positions.
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # --- KV cache: incremental inference ---
        if kv_cache is not None:
            past_k, past_v = kv_cache  # each (B, H, T_past, D)
            k = torch.cat([past_k, k], dim=2)  # (B, H, T_past + T, D)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)

        if kv_cache is not None:
            # Incremental mode: current Q attends to all past + current K/V.
            # No chunked mask needed because all past context is valid.
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T_total)

            # ALiBi bias in incremental mode: use global distances
            if self.pos_encoding == "alibi":
                attn_scores = self._apply_alibi_incremental(
                    attn_scores, past_len=kv_cache[0].shape[2], cur_len=T,
                )

            # Mask padded keys if needed
            if padding_mask is not None:
                # padding_mask is (B, T) for current; extend to cover full KV
                # In incremental mode, past tokens are assumed valid.
                T_total = k.shape[2]
                T_past = T_total - T
                past_valid = torch.ones(B, T_past, device=x.device, dtype=torch.bool)
                full_key_mask = torch.cat([past_valid, padding_mask], dim=1)  # (B, T_total)
                full_key_mask = full_key_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_total)
                attn_scores = attn_scores.masked_fill(~full_key_mask, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v)  # (B, H, T, D)

        else:
            # Training mode: full sequence with chunked causal mask
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

            # Apply positional bias / mask
            if self.pos_encoding == "alibi":
                chunk_size = self.chunk_size if self.chunk_size is not None else T
                alibi_bias = build_chunked_causal_alibi_bias(
                    seq_len=T,
                    chunk_size=chunk_size,
                    num_heads=self.num_heads,
                    device=x.device,
                    dtype=attn_scores.dtype,
                )  # (1, H, T, T)
                attn_scores = attn_scores + alibi_bias
            elif self.chunk_size is not None and self.chunk_size < T:
                # Non-ALiBi chunked causal: apply mask as additive bias
                mask = build_chunked_causal_mask(
                    seq_len=T,
                    chunk_size=self.chunk_size,
                    device=x.device,
                )  # (T, T) bool
                attn_bias = chunked_causal_mask_to_attn_bias(mask)  # (1, 1, T, T)
                attn_scores = attn_scores + attn_bias

            # Apply padding mask: mask out padded key positions
            if padding_mask is not None:
                # padding_mask: (B, T), True = valid
                key_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
                attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # --- Reshape and project output ---
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.hidden_size)
        output = self.out_proj(out)

        return output, new_kv_cache

    def _apply_alibi_incremental(
        self,
        attn_scores: Tensor,
        past_len: int,
        cur_len: int,
    ) -> Tensor:
        """Apply ALiBi bias in incremental (KV cache) mode.

        Uses global position distances: current positions are
        ``[past_len, past_len + cur_len)`` and key positions span
        ``[0, past_len + cur_len)``.

        Parameters
        ----------
        attn_scores : Tensor
            Shape ``(B, H, cur_len, past_len + cur_len)``.
        past_len : int
            Number of cached past positions.
        cur_len : int
            Number of current query positions.

        Returns
        -------
        Tensor
            ``attn_scores`` with ALiBi bias added, same shape.
        """
        from stylestream.destylizer.alibi import get_alibi_slopes

        total_len = past_len + cur_len
        device = attn_scores.device
        dtype = attn_scores.dtype

        slopes = get_alibi_slopes(self.num_heads).to(device=device, dtype=dtype)

        # Query positions: [past_len, past_len + cur_len)
        q_pos = torch.arange(past_len, past_len + cur_len, device=device, dtype=dtype)
        # Key positions: [0, total_len)
        k_pos = torch.arange(total_len, device=device, dtype=dtype)

        # Distance: |q_i - k_j|  -> (cur_len, total_len)
        dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()

        # slopes: (H,) -> (H, 1, 1)
        bias = -slopes.unsqueeze(-1).unsqueeze(-1) * dist.unsqueeze(0)  # (H, cur_len, total_len)
        bias = bias.unsqueeze(0)  # (1, H, cur_len, total_len)

        return attn_scores + bias
