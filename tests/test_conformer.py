"""Tests for ALiBi and Conformer implementations.

All tests are self-contained and use random tensors on CPU.
Smaller dimensions are used for speed: hidden_size=64, ffn_size=256,
num_heads=4, num_layers=2.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from stylestream.destylizer.alibi import get_alibi_slopes, build_alibi_bias
from stylestream.destylizer.conformer import (
    ConformerBlock,
    ConformerEncoder,
    ConvolutionModule,
    FeedForwardModule,
    MultiHeadSelfAttention,
)

# ------------------------------------------------------------------
# Shared constants for fast tests
# ------------------------------------------------------------------

HIDDEN = 64
FFN = 256
HEADS = 4
KERNEL = 7
LAYERS = 2
B = 2
T = 30
DROPOUT = 0.0  # deterministic during testing


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _random_input(batch: int = B, seq_len: int = T, dim: int = HIDDEN) -> torch.Tensor:
    """Return a random (B, T, D) tensor with requires_grad=True."""
    torch.manual_seed(42)
    return torch.randn(batch, seq_len, dim, requires_grad=True)


def _padding_mask(batch: int, seq_len: int, valid_len: int) -> torch.Tensor:
    """Return a bool mask (B, T) where positions >= valid_len are True (padded)."""
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[:, valid_len:] = True
    return mask


# ======================================================================
# ALiBi Tests
# ======================================================================


class TestALiBi:
    """Tests for get_alibi_slopes and build_alibi_bias."""

    # --- Slopes ---

    def test_slopes_shape(self) -> None:
        """Slopes shape is (num_heads,)."""
        for n in [1, 4, 8, 12, 16]:
            slopes = get_alibi_slopes(n)
            assert slopes.shape == (n,), f"Expected ({n},), got {slopes.shape}"

    def test_slopes_decreasing(self) -> None:
        """Slopes should decrease monotonically (each head attends at different range)."""
        slopes = get_alibi_slopes(8)
        for i in range(len(slopes) - 1):
            assert slopes[i] > slopes[i + 1], (
                f"Slope at head {i} ({slopes[i]:.6f}) should be > "
                f"slope at head {i + 1} ({slopes[i + 1]:.6f})"
            )

    def test_slopes_positive(self) -> None:
        """All slopes must be positive."""
        for n in [1, 4, 8, 12, 16]:
            slopes = get_alibi_slopes(n)
            assert (slopes > 0).all(), f"Found non-positive slopes for {n} heads"

    def test_slopes_power_of_two(self) -> None:
        """For 8 heads (power of 2), verify known closed-form values.

        slopes[k] = 2^(-8/8 * (k+1)) = 2^(-(k+1)) for k=0..7
        i.e. [0.5, 0.25, 0.125, 0.0625, ...]
        """
        slopes = get_alibi_slopes(8)
        expected = torch.tensor([2 ** (-(i + 1)) for i in range(8)])
        torch.testing.assert_close(slopes, expected, atol=1e-7, rtol=1e-6)

    def test_slopes_non_power_of_two(self) -> None:
        """For 12 heads (non-power-of-2), should still produce 12 valid positive slopes.

        The canonical ALiBi algorithm for non-power-of-2 head counts takes
        all slopes from the closest smaller power-of-2 (8 heads -> 8 slopes)
        and fills the remaining 4 with interleaved slopes from double the
        resolution.  This produces duplicate values by design, so we only
        check shape, positivity, and correct count.
        """
        slopes = get_alibi_slopes(12)
        assert slopes.shape == (12,)
        assert (slopes > 0).all()
        # First 8 slopes come from the power-of-2 formula for n=8
        expected_base = torch.tensor([2 ** (-(i + 1)) for i in range(8)])
        torch.testing.assert_close(slopes[:8], expected_base, atol=1e-7, rtol=1e-6)

    # --- Bias matrix ---

    @pytest.mark.parametrize("seq_len,num_heads", [(10, 4), (20, 8), (50, 12)])
    def test_bias_shape(self, seq_len: int, num_heads: int) -> None:
        """Output shape is (1, num_heads, seq_len, seq_len)."""
        bias = build_alibi_bias(seq_len, num_heads)
        assert bias.shape == (1, num_heads, seq_len, seq_len)

    def test_bias_diagonal_zero(self) -> None:
        """Diagonal elements (position attending to itself) should be 0."""
        bias = build_alibi_bias(seq_len=20, num_heads=4)
        for h in range(4):
            diag = torch.diagonal(bias[0, h])
            torch.testing.assert_close(
                diag, torch.zeros_like(diag), atol=1e-7, rtol=0.0,
            )

    def test_bias_negative_off_diagonal(self) -> None:
        """Off-diagonal elements should be <= 0 (penalizing distant positions)."""
        bias = build_alibi_bias(seq_len=20, num_heads=4)
        S = 20
        mask = ~torch.eye(S, dtype=torch.bool).unsqueeze(0).expand(4, -1, -1)
        off_diag = bias[0][mask]
        assert (off_diag <= 0).all(), "Off-diagonal ALiBi bias should be non-positive"

    def test_bias_symmetry_non_causal(self) -> None:
        """Non-causal bias should be symmetric: bias[i,j] == bias[j,i]."""
        bias = build_alibi_bias(seq_len=15, num_heads=4, causal=False)
        for h in range(4):
            mat = bias[0, h]
            torch.testing.assert_close(mat, mat.T, atol=1e-7, rtol=0.0)

    def test_bias_causal_mask(self) -> None:
        """In causal mode, future positions (j > i) should be -inf."""
        S = 10
        bias = build_alibi_bias(seq_len=S, num_heads=4, causal=True)
        for h in range(4):
            for i in range(S):
                for j in range(i + 1, S):
                    assert bias[0, h, i, j] == float("-inf"), (
                        f"Expected -inf at head={h}, i={i}, j={j}, "
                        f"got {bias[0, h, i, j]}"
                    )
            # Also verify valid (past/present) positions are finite
            for i in range(S):
                for j in range(0, i + 1):
                    assert torch.isfinite(bias[0, h, i, j]), (
                        f"Expected finite at head={h}, i={i}, j={j}"
                    )

    def test_bias_device(self) -> None:
        """Bias tensor should respect the device argument."""
        bias = build_alibi_bias(seq_len=10, num_heads=4, device=torch.device("cpu"))
        assert bias.device == torch.device("cpu")


# ======================================================================
# FeedForwardModule Tests
# ======================================================================


class TestFeedForward:
    """Tests for FeedForwardModule."""

    def test_output_shape(self) -> None:
        """(B, T, D) -> (B, T, D)."""
        ffn = FeedForwardModule(HIDDEN, FFN, dropout=DROPOUT)
        x = _random_input()
        out = ffn(x)
        assert out.shape == (B, T, HIDDEN)

    def test_gradient_flow(self) -> None:
        """loss.backward() should produce gradients on all parameters."""
        ffn = FeedForwardModule(HIDDEN, FFN, dropout=DROPOUT)
        x = _random_input()
        out = ffn(x)
        loss = out.sum()
        loss.backward()

        for name, param in ffn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ======================================================================
# ConvolutionModule Tests
# ======================================================================


class TestConvolution:
    """Tests for ConvolutionModule."""

    def test_output_shape(self) -> None:
        """Preserves (B, T, D) shape."""
        conv = ConvolutionModule(HIDDEN, kernel_size=KERNEL, dropout=DROPOUT)
        x = _random_input()
        out = conv(x.detach())
        assert out.shape == (B, T, HIDDEN)

    def test_causal_no_future_leakage(self) -> None:
        """In causal mode, output at time t must not depend on input at t+1.

        Strategy: run the full sequence, then modify a future frame and
        re-run.  The output at earlier frames must remain unchanged.
        """
        torch.manual_seed(42)
        conv = ConvolutionModule(HIDDEN, kernel_size=KERNEL, dropout=DROPOUT, causal=True)
        conv.eval()

        x = torch.randn(1, T, HIDDEN)
        out_full = conv(x.clone())

        # Modify the last frame
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(HIDDEN) * 100.0
        out_modified = conv(x_modified)

        # All frames except the last should be identical
        torch.testing.assert_close(
            out_full[:, :-1, :],
            out_modified[:, :-1, :],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_padding_mask_respected(self) -> None:
        """Padded positions should not affect unpadded outputs.

        Strategy: run with zeros in padded positions vs. random values;
        the valid-region output should match.
        """
        torch.manual_seed(42)
        conv = ConvolutionModule(HIDDEN, kernel_size=KERNEL, dropout=DROPOUT)
        conv.eval()

        valid_len = 15
        mask = _padding_mask(1, T, valid_len)

        # Input 1: padded region is zeros
        x1 = torch.randn(1, T, HIDDEN)
        x1[:, valid_len:, :] = 0.0

        # Input 2: padded region is large random values
        x2 = x1.clone()
        x2[:, valid_len:, :] = torch.randn(1, T - valid_len, HIDDEN) * 100.0

        out1 = conv(x1, padding_mask=mask)
        out2 = conv(x2, padding_mask=mask)

        # Valid region should produce identical outputs
        torch.testing.assert_close(
            out1[:, :valid_len, :],
            out2[:, :valid_len, :],
            atol=1e-5,
            rtol=1e-5,
        )


# ======================================================================
# MultiHeadSelfAttention Tests
# ======================================================================


class TestMHSA:
    """Tests for MultiHeadSelfAttention."""

    def test_output_shape(self) -> None:
        """(B, T, D) -> (B, T, D)."""
        mhsa = MultiHeadSelfAttention(HIDDEN, HEADS, dropout=DROPOUT)
        x = _random_input()
        out = mhsa(x)
        assert out.shape == (B, T, HIDDEN)

    def test_padding_mask(self) -> None:
        """Padded key positions should be ignored (softmax weight ~ 0).

        Strategy: change values at padded positions; the output at
        valid positions should remain the same.
        """
        torch.manual_seed(42)
        mhsa = MultiHeadSelfAttention(HIDDEN, HEADS, dropout=DROPOUT)
        mhsa.eval()

        valid_len = 15
        mask = _padding_mask(1, T, valid_len)

        x1 = torch.randn(1, T, HIDDEN)
        x2 = x1.clone()
        x2[:, valid_len:, :] = torch.randn(1, T - valid_len, HIDDEN) * 100.0

        out1 = mhsa(x1, padding_mask=mask)
        out2 = mhsa(x2, padding_mask=mask)

        torch.testing.assert_close(
            out1[:, :valid_len, :],
            out2[:, :valid_len, :],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_causal_mode(self) -> None:
        """causal=True should prevent attending to future positions.

        Strategy: modify the last frame and verify earlier frames are
        unchanged.
        """
        torch.manual_seed(42)
        mhsa = MultiHeadSelfAttention(HIDDEN, HEADS, dropout=DROPOUT)
        mhsa.eval()

        x = torch.randn(1, T, HIDDEN)
        out_orig = mhsa(x.clone(), causal=True)

        x_mod = x.clone()
        x_mod[:, -1, :] = torch.randn(HIDDEN) * 100.0
        out_mod = mhsa(x_mod, causal=True)

        # All frames except the last should be identical
        torch.testing.assert_close(
            out_orig[:, :-1, :],
            out_mod[:, :-1, :],
            atol=1e-5,
            rtol=1e-5,
        )


# ======================================================================
# ConformerBlock Tests
# ======================================================================


class TestConformerBlock:
    """Tests for ConformerBlock."""

    def test_output_shape(self) -> None:
        """(B, T, D) -> (B, T, D)."""
        block = ConformerBlock(
            hidden_size=HIDDEN, ffn_size=FFN, num_heads=HEADS,
            kernel_size=KERNEL, dropout=DROPOUT,
        )
        x = _random_input()
        out = block(x)
        assert out.shape == (B, T, HIDDEN)

    def test_residual_connection(self) -> None:
        """Output should differ from input (residuals are applied)."""
        block = ConformerBlock(
            hidden_size=HIDDEN, ffn_size=FFN, num_heads=HEADS,
            kernel_size=KERNEL, dropout=DROPOUT,
        )
        block.eval()
        x = _random_input().detach()
        out = block(x)
        assert not torch.allclose(out, x, atol=1e-6), (
            "Output is identical to input; residual connections may be broken"
        )

    def test_gradient_flow(self) -> None:
        """Gradients should flow to all parameters."""
        block = ConformerBlock(
            hidden_size=HIDDEN, ffn_size=FFN, num_heads=HEADS,
            kernel_size=KERNEL, dropout=DROPOUT,
        )
        x = _random_input()
        out = block(x)
        loss = out.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ======================================================================
# ConformerEncoder Tests
# ======================================================================


class TestConformerEncoder:
    """Tests for ConformerEncoder."""

    def _make_encoder(self, **overrides) -> ConformerEncoder:
        """Create a small encoder with test defaults."""
        kwargs = dict(
            num_layers=LAYERS,
            hidden_size=HIDDEN,
            ffn_size=FFN,
            num_heads=HEADS,
            kernel_size=KERNEL,
            dropout=DROPOUT,
        )
        kwargs.update(overrides)
        return ConformerEncoder(**kwargs)

    def test_output_shape(self) -> None:
        """(B, T, 768) -> (B, T, 768) with default config."""
        encoder = ConformerEncoder(
            num_layers=2, hidden_size=768, ffn_size=3072,
            num_heads=12, kernel_size=KERNEL, dropout=DROPOUT,
        )
        torch.manual_seed(42)
        x = torch.randn(1, 20, 768)
        out = encoder(x)
        assert out.shape == (1, 20, 768)

    def test_small_encoder(self) -> None:
        """Works with 2 layers and small hidden size for speed."""
        encoder = self._make_encoder()
        x = _random_input()
        out = encoder(x)
        assert out.shape == (B, T, HIDDEN)

    @pytest.mark.parametrize("seq_len", [10, 20, 50])
    def test_variable_length(self, seq_len: int) -> None:
        """Different T values should produce correct output shapes."""
        encoder = self._make_encoder()
        torch.manual_seed(42)
        x = torch.randn(B, seq_len, HIDDEN)
        out = encoder(x)
        assert out.shape == (B, seq_len, HIDDEN)

    def test_padding_mask(self) -> None:
        """Padding mask should propagate through all layers.

        Strategy: different padding content should not affect valid region.
        """
        torch.manual_seed(42)
        encoder = self._make_encoder()
        encoder.eval()

        valid_len = 15
        mask = _padding_mask(1, T, valid_len)

        x1 = torch.randn(1, T, HIDDEN)
        x2 = x1.clone()
        x2[:, valid_len:, :] = torch.randn(1, T - valid_len, HIDDEN) * 100.0

        out1 = encoder(x1, padding_mask=mask)
        out2 = encoder(x2, padding_mask=mask)

        torch.testing.assert_close(
            out1[:, :valid_len, :],
            out2[:, :valid_len, :],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_parameter_count(self) -> None:
        """Parameter count should be reasonable (not 0, not absurdly large)."""
        encoder = self._make_encoder()
        num_params = sum(p.numel() for p in encoder.parameters())
        assert num_params > 0, "Encoder has no parameters"
        # For 2 layers with hidden=64, ffn=256, heads=4, kernel=7
        # rough estimate: each block ~200K params, 2 blocks ~400K
        # allow generous bounds
        assert num_params < 10_000_000, (
            f"Parameter count {num_params} seems unreasonably large "
            f"for a small test encoder"
        )

    def test_gradient_flow_full(self) -> None:
        """Gradients should flow through all layers."""
        encoder = self._make_encoder()
        x = _random_input()
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_deterministic(self) -> None:
        """Same input should produce same output in eval mode."""
        encoder = self._make_encoder()
        encoder.eval()

        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)

        out1 = encoder(x.clone())
        out2 = encoder(x.clone())
        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)

    def test_batch_independence(self) -> None:
        """Each batch item should be processed independently.

        Strategy: run a batch, then run each item individually and
        verify the results match.
        """
        encoder = self._make_encoder()
        encoder.eval()

        torch.manual_seed(42)
        x = torch.randn(3, T, HIDDEN)

        out_batch = encoder(x)

        for i in range(3):
            out_single = encoder(x[i : i + 1])
            torch.testing.assert_close(
                out_batch[i : i + 1],
                out_single,
                atol=1e-5,
                rtol=1e-5,
                msg=f"Mismatch at batch index {i}",
            )
