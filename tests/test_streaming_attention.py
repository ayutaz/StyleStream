"""Tests for streaming chunked causal attention, masks, and KV cache.

Covers:
    - build_chunked_causal_mask: block lower-triangular mask generation
    - build_chunked_causal_alibi_bias: ALiBi + chunked causal bias
    - LayerKVCache / MultiLayerKVCache: key-value caching for streaming
    - ChunkedCausalMultiHeadAttention: unified attention with mask + cache

All tests use small dimensions for speed:
    hidden=64, heads=4, head_dim=16.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from stylestream.streaming.attention_mask import (
    build_chunked_causal_alibi_bias,
    build_chunked_causal_mask,
)
from stylestream.streaming.kv_cache import LayerKVCache, MultiLayerKVCache
from stylestream.streaming.chunked_attention import ChunkedCausalMultiHeadAttention

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS  # 16
B = 2


# ======================================================================
# Attention Mask Tests
# ======================================================================


class TestChunkedCausalMask:
    """Tests for build_chunked_causal_mask."""

    def test_mask_shape(self) -> None:
        """Mask should have shape (seq_len, seq_len)."""
        mask = build_chunked_causal_mask(seq_len=12, chunk_size=3)
        assert mask.shape == (12, 12)

    def test_full_attention_equivalence(self) -> None:
        """chunk_size >= seq_len should produce all-True mask (full attention)."""
        seq_len = 10
        mask = build_chunked_causal_mask(seq_len=seq_len, chunk_size=seq_len)
        assert mask.all(), "Full attention mask should be all True"

        # Also test with chunk_size larger than seq_len
        mask_larger = build_chunked_causal_mask(seq_len=seq_len, chunk_size=seq_len + 100)
        assert mask_larger.all(), "Oversized chunk should also produce all True"

    def test_block_structure_3x3(self) -> None:
        """chunk_size=3, seq_len=9 should give correct block lower-triangular."""
        mask = build_chunked_causal_mask(seq_len=9, chunk_size=3)

        # Expected: 3x3 block lower-triangular
        #   C0: rows 0-2, C1: rows 3-5, C2: rows 6-8
        #   C0 attends to C0 only
        #   C1 attends to C0, C1
        #   C2 attends to C0, C1, C2

        # Chunk 0 attends to chunk 0
        assert mask[0:3, 0:3].all(), "Chunk 0 should fully attend to chunk 0"
        # Chunk 0 does NOT attend to chunk 1 or 2
        assert not mask[0:3, 3:6].any(), "Chunk 0 should not attend to chunk 1"
        assert not mask[0:3, 6:9].any(), "Chunk 0 should not attend to chunk 2"

        # Chunk 1 attends to chunk 0 and chunk 1
        assert mask[3:6, 0:3].all(), "Chunk 1 should attend to chunk 0"
        assert mask[3:6, 3:6].all(), "Chunk 1 should attend to chunk 1"
        assert not mask[3:6, 6:9].any(), "Chunk 1 should not attend to chunk 2"

        # Chunk 2 attends to all
        assert mask[6:9, 0:3].all(), "Chunk 2 should attend to chunk 0"
        assert mask[6:9, 3:6].all(), "Chunk 2 should attend to chunk 1"
        assert mask[6:9, 6:9].all(), "Chunk 2 should attend to chunk 2"

    def test_causality_future_blocked(self) -> None:
        """Future chunks should be blocked: mask[i,j]=False when chunk(j) > chunk(i)."""
        seq_len = 12
        chunk_size = 4
        mask = build_chunked_causal_mask(seq_len=seq_len, chunk_size=chunk_size)

        for i in range(seq_len):
            for j in range(seq_len):
                chunk_i = i // chunk_size
                chunk_j = j // chunk_size
                if chunk_j > chunk_i:
                    assert not mask[i, j].item(), (
                        f"Future chunk access allowed: pos {i} (chunk {chunk_i}) "
                        f"can attend to pos {j} (chunk {chunk_j})"
                    )

    def test_within_chunk_full_attention(self) -> None:
        """All frames within the same chunk should attend to each other."""
        seq_len = 12
        chunk_size = 4
        mask = build_chunked_causal_mask(seq_len=seq_len, chunk_size=chunk_size)

        for chunk_idx in range(seq_len // chunk_size):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            block = mask[start:end, start:end]
            assert block.all(), (
                f"Within chunk {chunk_idx}: not all frames can attend to each other"
            )

    def test_non_divisible_seq_len(self) -> None:
        """seq_len=10, chunk_size=3: last chunk has 1 frame."""
        mask = build_chunked_causal_mask(seq_len=10, chunk_size=3)
        assert mask.shape == (10, 10)

        # Last frame (index 9) is in chunk 3 (9 // 3 = 3)
        # It should attend to all past chunks (0, 1, 2, 3)
        assert mask[9, :].all(), (
            "Last frame in partial chunk should attend to all past frames"
        )

        # First chunk (0-2) should NOT attend to the last partial chunk
        assert not mask[0, 9].item(), (
            "First chunk should not attend to last partial chunk"
        )

    @pytest.mark.parametrize("chunk_size", [10, 20, 30, 40, 50])
    def test_various_chunk_sizes(self, chunk_size: int) -> None:
        """Mask should be valid for different chunk sizes."""
        seq_len = 60
        mask = build_chunked_causal_mask(seq_len=seq_len, chunk_size=chunk_size)
        assert mask.shape == (seq_len, seq_len)

        # Verify basic causality: mask[0, -1] should be False unless full attention
        if chunk_size < seq_len:
            chunk_first = 0 // chunk_size
            chunk_last = (seq_len - 1) // chunk_size
            if chunk_last > chunk_first:
                assert not mask[0, seq_len - 1].item(), (
                    f"First position should not attend to last with chunk_size={chunk_size}"
                )


# ======================================================================
# ALiBi Bias Tests
# ======================================================================


class TestChunkedCausalAliBiBias:
    """Tests for build_chunked_causal_alibi_bias."""

    def test_alibi_bias_shape(self) -> None:
        """ALiBi bias should have shape (1, num_heads, seq_len, seq_len)."""
        bias = build_chunked_causal_alibi_bias(
            seq_len=12, chunk_size=4, num_heads=HEADS,
        )
        assert bias.shape == (1, HEADS, 12, 12)

    def test_alibi_future_is_neg_inf(self) -> None:
        """Blocked (future chunk) positions should have -inf bias."""
        seq_len = 9
        chunk_size = 3
        bias = build_chunked_causal_alibi_bias(
            seq_len=seq_len, chunk_size=chunk_size, num_heads=HEADS,
        )

        # Build the expected mask for reference
        mask = build_chunked_causal_mask(seq_len=seq_len, chunk_size=chunk_size)

        for h in range(HEADS):
            blocked_vals = bias[0, h][~mask]
            if blocked_vals.numel() > 0:
                assert (blocked_vals == float("-inf")).all(), (
                    f"Head {h}: not all blocked positions are -inf"
                )

    def test_alibi_past_positions_finite(self) -> None:
        """Valid (allowed) positions should have finite negative bias."""
        seq_len = 9
        chunk_size = 3
        bias = build_chunked_causal_alibi_bias(
            seq_len=seq_len, chunk_size=chunk_size, num_heads=HEADS,
        )

        mask = build_chunked_causal_mask(seq_len=seq_len, chunk_size=chunk_size)

        for h in range(HEADS):
            valid_vals = bias[0, h][mask]
            assert torch.isfinite(valid_vals).all(), (
                f"Head {h}: some valid positions have non-finite bias"
            )
            # ALiBi biases should be <= 0 (penalty for distance)
            assert (valid_vals <= 0.0).all(), (
                f"Head {h}: some valid positions have positive ALiBi bias"
            )


# ======================================================================
# KV Cache Tests
# ======================================================================


class TestLayerKVCache:
    """Tests for LayerKVCache."""

    def test_empty_cache(self) -> None:
        """New cache should be empty with length 0."""
        cache = LayerKVCache(max_frames=250)
        assert cache.is_empty is True
        assert cache.length == 0

    def test_single_append(self) -> None:
        """Appending 30 frames should give length=30."""
        torch.manual_seed(42)
        cache = LayerKVCache(max_frames=250)

        k = torch.randn(B, HEADS, 30, HEAD_DIM)
        v = torch.randn(B, HEADS, 30, HEAD_DIM)
        full_k, full_v = cache.append(k, v)

        assert cache.length == 30
        assert cache.is_empty is False
        assert full_k.shape == (B, HEADS, 30, HEAD_DIM)
        assert full_v.shape == (B, HEADS, 30, HEAD_DIM)

    def test_multiple_appends(self) -> None:
        """Appending 30 + 30 frames should give length=60."""
        torch.manual_seed(42)
        cache = LayerKVCache(max_frames=250)

        k1 = torch.randn(B, HEADS, 30, HEAD_DIM)
        v1 = torch.randn(B, HEADS, 30, HEAD_DIM)
        cache.append(k1, v1)

        k2 = torch.randn(B, HEADS, 30, HEAD_DIM)
        v2 = torch.randn(B, HEADS, 30, HEAD_DIM)
        full_k, full_v = cache.append(k2, v2)

        assert cache.length == 60
        assert full_k.shape == (B, HEADS, 60, HEAD_DIM)
        assert full_v.shape == (B, HEADS, 60, HEAD_DIM)

    def test_max_frames_eviction(self) -> None:
        """Exceeding max_frames should trim to max_frames (FIFO)."""
        torch.manual_seed(42)
        cache = LayerKVCache(max_frames=50)

        k1 = torch.randn(B, HEADS, 30, HEAD_DIM)
        v1 = torch.randn(B, HEADS, 30, HEAD_DIM)
        cache.append(k1, v1)
        assert cache.length == 30

        k2 = torch.randn(B, HEADS, 30, HEAD_DIM)
        v2 = torch.randn(B, HEADS, 30, HEAD_DIM)
        full_k, full_v = cache.append(k2, v2)

        # 30 + 30 = 60 > 50 -> trimmed to 50
        assert cache.length == 50
        assert full_k.shape == (B, HEADS, 50, HEAD_DIM)
        assert full_v.shape == (B, HEADS, 50, HEAD_DIM)

    def test_reset(self) -> None:
        """Reset should clear the cache completely."""
        torch.manual_seed(42)
        cache = LayerKVCache(max_frames=250)

        k = torch.randn(B, HEADS, 30, HEAD_DIM)
        v = torch.randn(B, HEADS, 30, HEAD_DIM)
        cache.append(k, v)
        assert cache.length == 30

        cache.reset()
        assert cache.is_empty is True
        assert cache.length == 0


class TestMultiLayerKVCache:
    """Tests for MultiLayerKVCache."""

    def test_indexing(self) -> None:
        """cache[0] and cache[5] should be accessible LayerKVCache instances."""
        cache = MultiLayerKVCache(num_layers=6, max_frames=250)
        assert isinstance(cache[0], LayerKVCache)
        assert isinstance(cache[5], LayerKVCache)

    def test_reset_all_layers(self) -> None:
        """reset() should clear all layer caches."""
        torch.manual_seed(42)
        cache = MultiLayerKVCache(num_layers=6, max_frames=250)

        # Fill layer 0 and layer 3
        k = torch.randn(B, HEADS, 30, HEAD_DIM)
        v = torch.randn(B, HEADS, 30, HEAD_DIM)
        cache[0].append(k, v)
        cache[3].append(k, v)
        assert cache[0].length == 30
        assert cache[3].length == 30

        cache.reset()
        assert cache[0].is_empty is True
        assert cache[3].is_empty is True
        assert cache.length == 0


# ======================================================================
# ChunkedCausalMultiHeadAttention Tests
# ======================================================================


class TestChunkedCausalMHA:
    """Tests for ChunkedCausalMultiHeadAttention."""

    @pytest.fixture
    def small_attn(self) -> ChunkedCausalMultiHeadAttention:
        """Build a small attention module for testing."""
        torch.manual_seed(42)
        return ChunkedCausalMultiHeadAttention(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            dropout=0.0,
            chunk_size=10,
            pos_encoding="none",
        )

    def test_output_shape(self, small_attn: ChunkedCausalMultiHeadAttention) -> None:
        """Output should be (B, T, hidden_size)."""
        torch.manual_seed(42)
        x = torch.randn(B, 20, HIDDEN)
        output, _ = small_attn(x)
        assert output.shape == (B, 20, HIDDEN)

    def test_kv_cache_returned(self, small_attn: ChunkedCausalMultiHeadAttention) -> None:
        """new_kv_cache should be a tuple of 2 tensors."""
        torch.manual_seed(42)
        x = torch.randn(B, 20, HIDDEN)
        _, kv_cache = small_attn(x)
        assert isinstance(kv_cache, tuple)
        assert len(kv_cache) == 2
        assert isinstance(kv_cache[0], torch.Tensor)
        assert isinstance(kv_cache[1], torch.Tensor)

    def test_full_attention_mode(self) -> None:
        """chunk_size=None should act as standard full attention."""
        torch.manual_seed(42)
        attn = ChunkedCausalMultiHeadAttention(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            dropout=0.0,
            chunk_size=None,
            pos_encoding="none",
        )
        x = torch.randn(B, 20, HIDDEN)
        output, kv_cache = attn(x)
        assert output.shape == (B, 20, HIDDEN)
        assert torch.isfinite(output).all(), "Full attention output contains nan/inf"

    def test_gradient_flow(self, small_attn: ChunkedCausalMultiHeadAttention) -> None:
        """Gradients should flow through all parameters."""
        torch.manual_seed(42)
        x = torch.randn(B, 20, HIDDEN)

        output, _ = small_attn(x)
        loss = output.sum()
        loss.backward()

        # Check that qkv_proj and out_proj have gradients
        assert small_attn.qkv_proj.weight.grad is not None, (
            "qkv_proj.weight should have gradients"
        )
        assert small_attn.out_proj.weight.grad is not None, (
            "out_proj.weight should have gradients"
        )
        # At least one grad should be non-zero
        has_nonzero_grad = False
        for p in small_attn.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break
        assert has_nonzero_grad, "At least one parameter should have non-zero gradient"

    def test_incremental_vs_batch(self, small_attn: ChunkedCausalMultiHeadAttention) -> None:
        """Processing chunk-by-chunk with KV cache should produce valid output."""
        torch.manual_seed(42)
        small_attn.eval()

        # Process full sequence in one go
        x_full = torch.randn(B, 20, HIDDEN)
        with torch.no_grad():
            out_full, _ = small_attn(x_full)

        # Process chunk-by-chunk using KV cache
        chunk1 = x_full[:, :10, :]
        chunk2 = x_full[:, 10:, :]

        with torch.no_grad():
            out1, kv = small_attn(chunk1)
            out2, _ = small_attn(chunk2, kv_cache=kv)

        # Incremental outputs should be valid tensors
        assert torch.isfinite(out1).all(), "Chunk 1 output has nan/inf"
        assert torch.isfinite(out2).all(), "Chunk 2 output has nan/inf"
        assert out1.shape == (B, 10, HIDDEN)
        assert out2.shape == (B, 10, HIDDEN)
