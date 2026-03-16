"""Tests for Adaptive Layer Norm Zero (adaLN-Zero) in the Stylizer DiT.

All tests are self-contained and use random tensors on CPU.
Small hidden size (64) is used for speed.
"""

from __future__ import annotations

import pytest
import torch

from stylestream.stylizer.adaln_zero import AdaLNZero, AdaLNModulation, FinalAdaLN

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

HIDDEN = 64
B = 2
T = 20


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _random_conditioning(batch: int = B, hidden: int = HIDDEN) -> torch.Tensor:
    """Return a random conditioning vector (B, hidden)."""
    torch.manual_seed(42)
    return torch.randn(batch, hidden)


def _random_sequence(batch: int = B, seq_len: int = T, hidden: int = HIDDEN) -> torch.Tensor:
    """Return a random sequence tensor (B, T, hidden)."""
    torch.manual_seed(42)
    return torch.randn(batch, seq_len, hidden)


# ======================================================================
# AdaLNZero Tests
# ======================================================================


class TestAdaLNZero:
    """Tests for the AdaLNZero conditioning module."""

    def test_output_count(self) -> None:
        """Forward should return exactly 6 tensors."""
        adaln = AdaLNZero(HIDDEN)
        c = _random_conditioning()
        outputs = adaln(c)

        assert len(outputs) == 6, f"Expected 6 outputs, got {len(outputs)}"

    def test_output_shapes(self) -> None:
        """Each output tensor should have shape (B, 1, hidden_size)."""
        adaln = AdaLNZero(HIDDEN)
        c = _random_conditioning()
        outputs = adaln(c)

        for i, out in enumerate(outputs):
            assert out.shape == (B, 1, HIDDEN), (
                f"Output {i}: expected ({B}, 1, {HIDDEN}), got {out.shape}"
            )

    def test_zero_initialization(self) -> None:
        """At initialization, all 6 outputs should be zero (weight and bias are zero-init)."""
        adaln = AdaLNZero(HIDDEN)
        c = _random_conditioning()
        outputs = adaln(c)

        for i, out in enumerate(outputs):
            assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), (
                f"Output {i} should be zero at init, max value: {out.abs().max()}"
            )

    def test_initial_gate_zero(self) -> None:
        """alpha_1 (index 2) and alpha_2 (index 5) should be all zeros at init.

        This is the key property of adaLN-Zero: the gates start at zero,
        so each DiT block initially acts as an identity function.
        """
        adaln = AdaLNZero(HIDDEN)
        c = _random_conditioning()
        outputs = adaln(c)

        alpha_1 = outputs[2]
        alpha_2 = outputs[5]

        assert torch.allclose(alpha_1, torch.zeros_like(alpha_1), atol=1e-7), (
            "alpha_1 should be zero at init"
        )
        assert torch.allclose(alpha_2, torch.zeros_like(alpha_2), atol=1e-7), (
            "alpha_2 should be zero at init"
        )

    def test_gradient_flow(self) -> None:
        """Gradients should flow through all 6 outputs back to the linear layer."""
        adaln = AdaLNZero(HIDDEN)
        c = _random_conditioning()
        c.requires_grad_(True)

        outputs = adaln(c)
        loss = sum(o.sum() for o in outputs)
        loss.backward()

        # Check that the linear layer received gradients
        for name, param in adaln.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

        # Check that the input conditioning vector received gradients
        assert c.grad is not None, "Conditioning vector should receive gradients"

    def test_nonzero_after_training(self) -> None:
        """After parameter update, outputs should no longer be zero."""
        adaln = AdaLNZero(HIDDEN)
        # Simulate a training step by manually changing the weights
        with torch.no_grad():
            adaln.linear[1].weight.fill_(0.01)
            adaln.linear[1].bias.fill_(0.01)

        c = _random_conditioning()
        outputs = adaln(c)

        any_nonzero = any(o.abs().sum() > 0 for o in outputs)
        assert any_nonzero, "After weight update, outputs should be non-zero"


# ======================================================================
# AdaLNModulation Tests
# ======================================================================


class TestAdaLNModulation:
    """Tests for the AdaLNModulation module (scale + shift after LayerNorm)."""

    def test_output_shape(self) -> None:
        """Output shape should match input shape (B, T, hidden)."""
        mod = AdaLNModulation(HIDDEN)
        x = _random_sequence()
        gamma = torch.zeros(B, 1, HIDDEN)
        beta = torch.zeros(B, 1, HIDDEN)

        out = mod(x, gamma, beta)
        assert out.shape == (B, T, HIDDEN)

    def test_identity_at_zero(self) -> None:
        """When gamma=0 and beta=0, output should equal LayerNorm(x)."""
        mod = AdaLNModulation(HIDDEN)
        x = _random_sequence()
        gamma = torch.zeros(B, 1, HIDDEN)
        beta = torch.zeros(B, 1, HIDDEN)

        out = mod(x, gamma, beta)
        expected = mod.norm(x)

        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_scale_effect(self) -> None:
        """When gamma != 0, the magnitude of the output should change."""
        mod = AdaLNModulation(HIDDEN)
        x = _random_sequence()
        gamma_zero = torch.zeros(B, 1, HIDDEN)
        gamma_nonzero = torch.ones(B, 1, HIDDEN) * 2.0  # scale by 3x
        beta = torch.zeros(B, 1, HIDDEN)

        out_zero = mod(x, gamma_zero, beta)
        out_scaled = mod(x, gamma_nonzero, beta)

        # (1 + 2.0) * LN(x) = 3.0 * LN(x)
        expected_scaled = 3.0 * mod.norm(x)
        torch.testing.assert_close(out_scaled, expected_scaled, atol=1e-5, rtol=1e-5)

        # Scaled output should differ from unscaled
        assert not torch.allclose(out_zero, out_scaled, atol=1e-3)

    def test_shift_effect(self) -> None:
        """When beta != 0, the output should be shifted."""
        mod = AdaLNModulation(HIDDEN)
        x = _random_sequence()
        gamma = torch.zeros(B, 1, HIDDEN)
        beta_zero = torch.zeros(B, 1, HIDDEN)
        beta_shift = torch.ones(B, 1, HIDDEN) * 5.0

        out_zero = mod(x, gamma, beta_zero)
        out_shifted = mod(x, gamma, beta_shift)

        # Difference should be exactly the beta shift
        diff = out_shifted - out_zero
        expected_diff = torch.ones(B, T, HIDDEN) * 5.0
        torch.testing.assert_close(diff, expected_diff, atol=1e-5, rtol=1e-5)


# ======================================================================
# FinalAdaLN Tests
# ======================================================================


class TestFinalAdaLN:
    """Tests for the FinalAdaLN module (final layer before mel projection)."""

    def test_output_shape(self) -> None:
        """Output should have shape (B, T, output_size)."""
        output_size = 10
        final = FinalAdaLN(HIDDEN, output_size)
        x = _random_sequence()
        c = _random_conditioning()

        out = final(x, c)
        assert out.shape == (B, T, output_size)

    @pytest.mark.parametrize("output_size", [10, 50, 100, 200])
    def test_different_output_sizes(self, output_size: int) -> None:
        """FinalAdaLN should work with various output sizes."""
        torch.manual_seed(42)
        final = FinalAdaLN(HIDDEN, output_size)
        x = _random_sequence()
        c = _random_conditioning()

        out = final(x, c)
        assert out.shape == (B, T, output_size)

    def test_zero_init_output(self) -> None:
        """At initialization, the output should be near zero.

        Both the adaLN linear and the output linear are zero-initialized,
        so the initial output should be zero.
        """
        final = FinalAdaLN(HIDDEN, 10)
        x = _random_sequence()
        c = _random_conditioning()

        out = final(x, c)

        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
            f"Initial output should be near zero, max: {out.abs().max()}"
        )

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the FinalAdaLN module."""
        final = FinalAdaLN(HIDDEN, 10)
        x = _random_sequence()
        x.requires_grad_(True)
        c = _random_conditioning()
        c.requires_grad_(True)

        out = final(x, c)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "x should receive gradients"
        assert c.grad is not None, "c should receive gradients"

        for name, param in final.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_conditioning_effect(self) -> None:
        """Different conditioning vectors should produce different outputs.

        Only after the zero-init params have been changed.
        """
        final = FinalAdaLN(HIDDEN, 10)
        # Break zero-init so conditioning has effect
        with torch.no_grad():
            final.adaln_linear[1].weight.fill_(0.01)
            final.output_linear.weight.fill_(0.01)

        x = _random_sequence()
        c1 = torch.randn(B, HIDDEN)
        c2 = torch.randn(B, HIDDEN) * 5.0

        out1 = final(x, c1)
        out2 = final(x, c2)

        assert not torch.allclose(out1, out2, atol=1e-3), (
            "Different conditioning should produce different outputs"
        )
