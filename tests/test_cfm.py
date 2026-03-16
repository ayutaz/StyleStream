"""Tests for Conditional Flow Matching (CFM) in the Stylizer.

All tests are self-contained and use random tensors on CPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from stylestream.stylizer.cfm import ConditionalFlowMatching

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

B = 4
T = 20
MEL_DIM = 10


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_cfm(sigma_min: float = 1e-5) -> ConditionalFlowMatching:
    """Create a ConditionalFlowMatching instance."""
    return ConditionalFlowMatching(sigma_min=sigma_min)


def _random_mel(batch: int = B, seq_len: int = T, mel_dim: int = MEL_DIM) -> torch.Tensor:
    """Return a random mel spectrogram tensor."""
    torch.manual_seed(42)
    return torch.randn(batch, seq_len, mel_dim)


def _random_mask(batch: int = B, seq_len: int = T) -> torch.Tensor:
    """Return a random binary mask: 1 = masked (generate), 0 = context."""
    torch.manual_seed(42)
    return (torch.rand(batch, seq_len) > 0.3).float()


# ======================================================================
# ConditionalFlowMatching Tests
# ======================================================================


class TestConditionalFlowMatching:
    """Tests for the CFM module."""

    # ------------------------------------------------------------------
    # Interpolation tests
    # ------------------------------------------------------------------

    def test_interpolate_at_zero(self) -> None:
        """At t=0, x_t should be approximately x_0 (noise).

        x_t = (1 - (1 - sigma_min) * 0) * x_0 + 0 * x_1 = x_0
        """
        cfm = _make_cfm()
        x_0 = _random_mel()
        x_1 = torch.randn_like(x_0)
        t = torch.zeros(B)

        x_t = cfm.interpolate(x_0, x_1, t)
        torch.testing.assert_close(x_t, x_0, atol=1e-5, rtol=1e-5)

    def test_interpolate_at_one(self) -> None:
        """At t=1, x_t should be approximately x_1 (target).

        x_t = (1 - (1 - sigma_min) * 1) * x_0 + 1 * x_1
            = sigma_min * x_0 + x_1
            ~ x_1  (sigma_min is tiny)
        """
        cfm = _make_cfm()
        x_0 = _random_mel()
        x_1 = torch.randn_like(x_0)
        t = torch.ones(B)

        x_t = cfm.interpolate(x_0, x_1, t)
        # With sigma_min = 1e-5, the x_0 contribution is negligible
        torch.testing.assert_close(x_t, x_1, atol=1e-3, rtol=1e-3)

    def test_interpolate_midpoint(self) -> None:
        """At t=0.5, x_t should be approximately (x_0 + x_1) / 2.

        With sigma_min ~ 0:
        x_t = (1 - 0.5) * x_0 + 0.5 * x_1 = 0.5 * (x_0 + x_1)
        """
        cfm = _make_cfm(sigma_min=0.0)  # exact linear interp
        x_0 = _random_mel()
        x_1 = torch.randn_like(x_0)
        t = torch.full((B,), 0.5)

        x_t = cfm.interpolate(x_0, x_1, t)
        expected = 0.5 * (x_0 + x_1)
        torch.testing.assert_close(x_t, expected, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Noise sampling tests
    # ------------------------------------------------------------------

    def test_sample_noise_shape(self) -> None:
        """sample_noise should return the requested shape."""
        cfm = _make_cfm()
        shape = (B, T, MEL_DIM)
        noise = cfm.sample_noise(shape, device=torch.device("cpu"))
        assert noise.shape == shape

    def test_sample_noise_distribution(self) -> None:
        """Sampled noise should be approximately N(0, 1)."""
        cfm = _make_cfm()
        shape = (1000, 100, MEL_DIM)
        noise = cfm.sample_noise(shape, device=torch.device("cpu"))

        # Check mean ~ 0 and std ~ 1
        mean = noise.mean().item()
        std = noise.std().item()
        assert abs(mean) < 0.05, f"Mean {mean} should be near 0"
        assert abs(std - 1.0) < 0.05, f"Std {std} should be near 1.0"

    # ------------------------------------------------------------------
    # Timestep sampling tests
    # ------------------------------------------------------------------

    def test_sample_timestep_range(self) -> None:
        """All sampled timesteps should be in [0, 1]."""
        cfm = _make_cfm()
        t = cfm.sample_timestep(1000, device=torch.device("cpu"))

        assert (t >= 0.0).all(), "Timesteps must be >= 0"
        assert (t <= 1.0).all(), "Timesteps must be <= 1"

    # ------------------------------------------------------------------
    # Loss computation tests
    # ------------------------------------------------------------------

    def test_compute_loss_masked(self) -> None:
        """Loss should only consider masked positions.

        If we set the mask to zero everywhere, the loss should be ~0
        (divided by eps, so very small).
        """
        cfm = _make_cfm()
        velocity_pred = torch.randn(B, T, MEL_DIM)
        x_0 = torch.randn(B, T, MEL_DIM)
        x_1 = torch.randn(B, T, MEL_DIM)
        mask = torch.zeros(B, T)  # no masked positions

        loss = cfm.compute_loss(velocity_pred, x_1, x_0, mask)
        # With all-zero mask, numerator is 0 and denominator is eps
        assert loss.item() < 1e-4, f"Loss should be ~0 with no mask, got {loss.item()}"

    def test_compute_loss_zero_on_match(self) -> None:
        """If predicted velocity matches target velocity, loss should be ~0."""
        cfm = _make_cfm()
        x_0 = torch.randn(B, T, MEL_DIM)
        x_1 = torch.randn(B, T, MEL_DIM)

        # Compute target velocity
        target_velocity = cfm.target_velocity(x_0, x_1)
        mask = torch.ones(B, T)  # all masked

        loss = cfm.compute_loss(target_velocity, x_1, x_0, mask)
        assert loss.item() < 1e-6, f"Loss should be ~0 when prediction matches target, got {loss.item()}"

    def test_compute_loss_gradient(self) -> None:
        """Gradients should flow through the loss computation."""
        cfm = _make_cfm()
        velocity_pred = torch.randn(B, T, MEL_DIM, requires_grad=True)
        x_0 = torch.randn(B, T, MEL_DIM)
        x_1 = torch.randn(B, T, MEL_DIM)
        mask = torch.ones(B, T)

        loss = cfm.compute_loss(velocity_pred, x_1, x_0, mask)
        loss.backward()

        assert velocity_pred.grad is not None, "velocity_pred should receive gradients"
        assert velocity_pred.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_loss_ignores_context(self) -> None:
        """Changing the unmasked (context) region should not change the loss."""
        cfm = _make_cfm()
        torch.manual_seed(42)
        x_0 = torch.randn(B, T, MEL_DIM)
        x_1 = torch.randn(B, T, MEL_DIM)
        velocity_pred = torch.randn(B, T, MEL_DIM)

        # Mask first 10 frames, leave last 10 as context
        mask = torch.zeros(B, T)
        mask[:, :10] = 1.0

        loss1 = cfm.compute_loss(velocity_pred, x_1, x_0, mask)

        # Modify unmasked region
        velocity_pred2 = velocity_pred.clone()
        velocity_pred2[:, 10:, :] = torch.randn(B, T - 10, MEL_DIM) * 100.0

        loss2 = cfm.compute_loss(velocity_pred2, x_1, x_0, mask)
        torch.testing.assert_close(loss1, loss2, atol=1e-6, rtol=1e-6)

    # ------------------------------------------------------------------
    # Euler sampling tests
    # ------------------------------------------------------------------

    def test_euler_sample_shape(self) -> None:
        """Output shape should match the requested shape."""
        cfm = _make_cfm()
        shape = (B, T, MEL_DIM)
        velocity_fn = lambda x_t, t: torch.zeros_like(x_t)

        out = cfm.euler_sample(velocity_fn, shape, nfe=4)
        assert out.shape == shape

    def test_euler_sample_nfe_steps(self) -> None:
        """velocity_fn should be called exactly nfe times."""
        cfm = _make_cfm()
        shape = (B, T, MEL_DIM)
        call_count = 0

        def counting_fn(x_t, t):
            nonlocal call_count
            call_count += 1
            return torch.zeros_like(x_t)

        nfe = 8
        cfm.euler_sample(counting_fn, shape, nfe=nfe)
        assert call_count == nfe, f"Expected {nfe} calls, got {call_count}"

    def test_euler_sample_deterministic(self) -> None:
        """With a fixed seed, euler_sample should produce consistent results."""
        cfm = _make_cfm()
        shape = (B, T, MEL_DIM)
        velocity_fn = lambda x_t, t: torch.ones_like(x_t) * 0.1

        torch.manual_seed(42)
        out1 = cfm.euler_sample(velocity_fn, shape, nfe=4)
        torch.manual_seed(42)
        out2 = cfm.euler_sample(velocity_fn, shape, nfe=4)

        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)

    def test_sigma_min_effect(self) -> None:
        """sigma_min > 0 should prevent degenerate t=0 interpolation.

        With sigma_min > 0, at t=0 we get x_t = x_0 (not a degenerate point).
        """
        cfm_no_sigma = _make_cfm(sigma_min=0.0)
        cfm_with_sigma = _make_cfm(sigma_min=0.1)

        x_0 = _random_mel()
        x_1 = torch.randn_like(x_0)
        t = torch.full((B,), 0.5)

        x_t_no = cfm_no_sigma.interpolate(x_0, x_1, t)
        x_t_with = cfm_with_sigma.interpolate(x_0, x_1, t)

        # They should differ because sigma_min changes the interpolation formula
        assert not torch.allclose(x_t_no, x_t_with, atol=1e-4), (
            "Different sigma_min values should produce different interpolation"
        )

    def test_full_mask_vs_partial(self) -> None:
        """Full mask (all 1s) should use all positions in the loss."""
        cfm = _make_cfm()
        torch.manual_seed(42)
        velocity_pred = torch.randn(B, T, MEL_DIM)
        x_0 = torch.randn(B, T, MEL_DIM)
        x_1 = torch.randn(B, T, MEL_DIM)

        full_mask = torch.ones(B, T)
        partial_mask = torch.ones(B, T)
        partial_mask[:, T // 2:] = 0.0

        loss_full = cfm.compute_loss(velocity_pred, x_1, x_0, full_mask)
        loss_partial = cfm.compute_loss(velocity_pred, x_1, x_0, partial_mask)

        # Full mask includes more error terms; generally they should differ
        assert not torch.isclose(loss_full, loss_partial, atol=1e-4), (
            "Full and partial mask losses should generally differ"
        )

    def test_negative_sigma_min_raises(self) -> None:
        """sigma_min < 0 should raise a ValueError."""
        with pytest.raises(ValueError, match="sigma_min must be non-negative"):
            ConditionalFlowMatching(sigma_min=-0.1)

    def test_nfe_zero_raises(self) -> None:
        """nfe < 1 should raise a ValueError in euler_sample."""
        cfm = _make_cfm()
        with pytest.raises(ValueError, match="nfe must be >= 1"):
            cfm.euler_sample(
                velocity_fn=lambda x, t: x,
                shape=(B, T, MEL_DIM),
                nfe=0,
            )
