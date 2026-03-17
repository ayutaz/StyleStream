"""Tests for Rotary Position Embedding (RoPE) in the Stylizer DiT.

All tests are self-contained and use random tensors on CPU.
Small dimensions are used for speed: dim=32, seq_len=20, batch=2, heads=4.
"""

from __future__ import annotations

import pytest
import torch

from stylestream.stylizer.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    _rotate_half,
)

# ------------------------------------------------------------------
# Shared constants for fast tests
# ------------------------------------------------------------------

DIM = 32       # head dimension (must be even)
SEQ_LEN = 20   # default sequence length
B = 2           # batch size
HEADS = 4       # number of attention heads


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_rope(dim: int = DIM, max_seq_len: int = 4096) -> RotaryPositionEmbedding:
    """Return a RotaryPositionEmbedding instance."""
    return RotaryPositionEmbedding(dim=dim, max_seq_len=max_seq_len)


def _random_qk(
    batch: int = B,
    heads: int = HEADS,
    seq_len: int = SEQ_LEN,
    dim: int = DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return random (q, k) tensors of shape (B, heads, T, dim)."""
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    return q, k


# ======================================================================
# RotaryPositionEmbedding Tests
# ======================================================================


class TestRotaryPositionEmbedding:
    """Tests for the RotaryPositionEmbedding module."""

    def test_output_shapes(self) -> None:
        """cos and sin should each have shape (1, 1, T, dim)."""
        rope = _make_rope()
        x = torch.randn(B, HEADS, SEQ_LEN, DIM)
        cos, sin = rope(x)

        assert cos.shape == (1, 1, SEQ_LEN, DIM), f"cos shape: {cos.shape}"
        assert sin.shape == (1, 1, SEQ_LEN, DIM), f"sin shape: {sin.shape}"

    @pytest.mark.parametrize("seq_len", [5, 10, 50, 100])
    def test_different_seq_lengths(self, seq_len: int) -> None:
        """RoPE should work correctly with various sequence lengths."""
        rope = _make_rope()
        x = torch.randn(B, HEADS, seq_len, DIM)
        cos, sin = rope(x)

        assert cos.shape == (1, 1, seq_len, DIM)
        assert sin.shape == (1, 1, seq_len, DIM)

    def test_offset(self) -> None:
        """With offset > 0, the returned positions should shift accordingly.

        RoPE with offset=5 for T=10 should match positions 5..14 of a
        longer RoPE computation for T=15.
        """
        rope = _make_rope()

        # Full computation for 15 positions
        x_full = torch.randn(B, HEADS, 15, DIM)
        cos_full, sin_full = rope(x_full, offset=0)

        # Offset computation: positions 5..14
        x_short = torch.randn(B, HEADS, 10, DIM)
        cos_offset, sin_offset = rope(x_short, offset=5)

        torch.testing.assert_close(
            cos_offset, cos_full[:, :, 5:15, :],
            atol=1e-6, rtol=1e-6,
        )
        torch.testing.assert_close(
            sin_offset, sin_full[:, :, 5:15, :],
            atol=1e-6, rtol=1e-6,
        )

    def test_cache_consistency(self) -> None:
        """Calling forward multiple times with the same input should return identical results."""
        rope = _make_rope()
        x = torch.randn(B, HEADS, SEQ_LEN, DIM)

        cos1, sin1 = rope(x)
        cos2, sin2 = rope(x)

        torch.testing.assert_close(cos1, cos2, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(sin1, sin2, atol=1e-7, rtol=1e-7)

    def test_preallocated_cache_slicing(self) -> None:
        """Slicing the pre-allocated cache should return the correct subsequence.

        With a large pre-allocated max_seq_len, requesting a short
        sequence should return the correct slice without recomputation.
        """
        rope = _make_rope(max_seq_len=4096)
        x_short = torch.randn(B, HEADS, 20, DIM)

        cos, sin = rope(x_short, offset=0)
        assert cos.shape == (1, 1, 20, DIM)

        # Requesting with an offset should give positions from the same table
        x_chunk = torch.randn(B, HEADS, 10, DIM)
        cos_off, sin_off = rope(x_chunk, offset=10)
        assert cos_off.shape == (1, 1, 10, DIM)

        # Verify consistency: offset=10 for T=10 should match positions 10..19
        # from a T=20 fetch
        torch.testing.assert_close(
            cos_off, cos[:, :, 10:20, :], atol=1e-7, rtol=1e-7
        )

    def test_cos_sin_values_finite(self) -> None:
        """All cos/sin values should be finite (no nan/inf)."""
        rope = _make_rope()
        x = torch.randn(B, HEADS, SEQ_LEN, DIM)
        cos, sin = rope(x)

        assert torch.isfinite(cos).all(), "Found non-finite values in cos"
        assert torch.isfinite(sin).all(), "Found non-finite values in sin"

    def test_cos_sin_bounded(self) -> None:
        """cos and sin values should be in [-1, 1]."""
        rope = _make_rope()
        x = torch.randn(B, HEADS, SEQ_LEN, DIM)
        cos, sin = rope(x)

        assert (cos >= -1.0 - 1e-6).all() and (cos <= 1.0 + 1e-6).all()
        assert (sin >= -1.0 - 1e-6).all() and (sin <= 1.0 + 1e-6).all()


# ======================================================================
# apply_rotary_pos_emb Tests
# ======================================================================


class TestApplyRotaryPosEmb:
    """Tests for the apply_rotary_pos_emb function."""

    def test_output_shapes(self) -> None:
        """Output shapes should match input shapes."""
        q, k = _random_qk()
        rope = _make_rope()
        cos, sin = rope(q)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_equivariance(self) -> None:
        """The same position should receive the same rotation regardless of batch.

        For position p, the rotation applied to q[b1, h, p, :] should be the
        same transformation as applied to q[b2, h, p, :].
        """
        torch.manual_seed(42)
        q = torch.randn(3, HEADS, SEQ_LEN, DIM)
        k = torch.randn(3, HEADS, SEQ_LEN, DIM)
        rope = _make_rope()
        cos, sin = rope(q)

        # Apply with identical q across batch items at position 5
        q_same = torch.zeros(3, HEADS, SEQ_LEN, DIM)
        q_same[:, :, 5, :] = q[0, :, 5, :]  # same value across batch
        k_same = torch.zeros_like(q_same)

        q_rot, _ = apply_rotary_pos_emb(q_same, k_same, cos, sin)

        # All batch items at position 5 should produce the same rotated output
        torch.testing.assert_close(
            q_rot[0, :, 5, :], q_rot[1, :, 5, :], atol=1e-6, rtol=1e-6,
        )
        torch.testing.assert_close(
            q_rot[0, :, 5, :], q_rot[2, :, 5, :], atol=1e-6, rtol=1e-6,
        )

    def test_different_positions_different_rotation(self) -> None:
        """Vectors at different positions should be rotated differently."""
        torch.manual_seed(42)
        # Use the same vector at two different positions
        vec = torch.randn(1, 1, 1, DIM)
        q = vec.expand(1, 1, SEQ_LEN, DIM).clone()
        k = q.clone()
        rope = _make_rope()
        cos, sin = rope(q)

        q_rot, _ = apply_rotary_pos_emb(q, k, cos, sin)

        # Positions 0 and 5 should yield different results
        assert not torch.allclose(q_rot[0, 0, 0, :], q_rot[0, 0, 5, :], atol=1e-4), (
            "Same vector at different positions should be rotated differently"
        )

    def test_gradient_flow(self) -> None:
        """Gradients should propagate through the rotation operation."""
        q, k = _random_qk()
        q.requires_grad_(True)
        k.requires_grad_(True)

        rope = _make_rope()
        cos, sin = rope(q.detach())

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None, "q should receive gradients"
        assert k.grad is not None, "k should receive gradients"
        assert q.grad.abs().sum() > 0, "q gradients should be non-zero"
        assert k.grad.abs().sum() > 0, "k gradients should be non-zero"

    def test_rotation_preserves_norm(self) -> None:
        """RoPE is a rotation; it should approximately preserve vector norms.

        The _rotate_half + cos/sin formula applies a rotation in 2D subspaces,
        which preserves the L2 norm of each (dim/2)-pair.
        """
        torch.manual_seed(42)
        q = torch.randn(B, HEADS, SEQ_LEN, DIM)
        k = torch.randn(B, HEADS, SEQ_LEN, DIM)
        rope = _make_rope()
        cos, sin = rope(q)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Check norms are preserved per vector
        q_norms = q.norm(dim=-1)
        q_rot_norms = q_rot.norm(dim=-1)
        torch.testing.assert_close(q_norms, q_rot_norms, atol=1e-5, rtol=1e-5)
