"""Tests for Classifier-Free Guidance (CFG) in the Stylizer.

All tests are self-contained and use random tensors on CPU.
"""

from __future__ import annotations

import pytest
import torch

from stylestream.stylizer.cfg import ClassifierFreeGuidance

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

B = 8
T = 20
CONTENT_DIM = 64
MEL_DIM = 10
STYLE_DIM = 64


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_cfg(**overrides) -> ClassifierFreeGuidance:
    """Create a CFG instance with defaults."""
    kwargs = dict(
        content_drop_prob=0.2,
        context_drop_prob=0.3,
        style_drop_prob=0.3,
        guidance_strength=2.0,
    )
    kwargs.update(overrides)
    return ClassifierFreeGuidance(**kwargs)


def _random_conditions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random (content_features, context_mel, style_emb)."""
    torch.manual_seed(42)
    content = torch.randn(B, T, CONTENT_DIM)
    context = torch.randn(B, T, MEL_DIM)
    style = torch.randn(B, STYLE_DIM)
    return content, context, style


# ======================================================================
# ClassifierFreeGuidance Tests
# ======================================================================


class TestClassifierFreeGuidance:
    """Tests for the CFG class."""

    # ------------------------------------------------------------------
    # Training dropout tests
    # ------------------------------------------------------------------

    def test_no_dropout_when_prob_zero(self) -> None:
        """With drop probability = 0, no conditions should be zeroed."""
        cfg = _make_cfg(content_drop_prob=0.0, context_drop_prob=0.0, style_drop_prob=0.0)
        content, context, style = _random_conditions()

        c_out, ctx_out, s_out = cfg.apply_training_dropout(content, context, style)

        torch.testing.assert_close(c_out, content, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(ctx_out, context, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(s_out, style, atol=1e-7, rtol=1e-7)

    def test_full_dropout_when_prob_one(self) -> None:
        """With drop probability = 1, all conditions should be zeroed."""
        cfg = _make_cfg(content_drop_prob=1.0, context_drop_prob=1.0, style_drop_prob=1.0)
        content, context, style = _random_conditions()

        c_out, ctx_out, s_out = cfg.apply_training_dropout(content, context, style)

        assert torch.allclose(c_out, torch.zeros_like(c_out))
        assert torch.allclose(ctx_out, torch.zeros_like(ctx_out))
        assert torch.allclose(s_out, torch.zeros_like(s_out))

    def test_dropout_shapes_preserved(self) -> None:
        """Output shapes should match input shapes."""
        cfg = _make_cfg()
        content, context, style = _random_conditions()

        c_out, ctx_out, s_out = cfg.apply_training_dropout(content, context, style)

        assert c_out.shape == content.shape
        assert ctx_out.shape == context.shape
        assert s_out.shape == style.shape

    def test_pre_sampled_dropout(self) -> None:
        """When pre-sampled drop masks are provided, they should be used directly."""
        cfg = _make_cfg()
        content, context, style = _random_conditions()

        # Pre-sample: drop first 2 content, first 3 context, first 3 style
        drop_c = torch.zeros(B, dtype=torch.bool)
        drop_c[:2] = True
        drop_ctx = torch.zeros(B, dtype=torch.bool)
        drop_ctx[:3] = True
        drop_s = torch.zeros(B, dtype=torch.bool)
        drop_s[:3] = True

        c_out, ctx_out, s_out = cfg.apply_training_dropout(
            content, context, style,
            cfg_drop_content=drop_c,
            cfg_drop_context=drop_ctx,
            cfg_drop_style=drop_s,
        )

        # Dropped samples should be zero
        assert torch.allclose(c_out[:2], torch.zeros_like(c_out[:2]))
        assert torch.allclose(ctx_out[:3], torch.zeros_like(ctx_out[:3]))
        assert torch.allclose(s_out[:3], torch.zeros_like(s_out[:3]))

        # Kept samples should be unchanged
        torch.testing.assert_close(c_out[2:], content[2:], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(ctx_out[3:], context[3:], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(s_out[3:], style[3:], atol=1e-7, rtol=1e-7)

    def test_dropout_statistics(self) -> None:
        """Over many samples, dropout rates should approximately match probabilities."""
        cfg = _make_cfg(content_drop_prob=0.2, context_drop_prob=0.3, style_drop_prob=0.3)

        n_trials = 2000
        content_drops = 0
        context_drops = 0
        style_drops = 0

        for _ in range(n_trials):
            content = torch.randn(1, T, CONTENT_DIM)
            context = torch.randn(1, T, MEL_DIM)
            style = torch.randn(1, STYLE_DIM)

            c_out, ctx_out, s_out = cfg.apply_training_dropout(content, context, style)

            if torch.allclose(c_out, torch.zeros_like(c_out)):
                content_drops += 1
            if torch.allclose(ctx_out, torch.zeros_like(ctx_out)):
                context_drops += 1
            if torch.allclose(s_out, torch.zeros_like(s_out)):
                style_drops += 1

        # Check rates are within reasonable bounds (3 sigma for binomial)
        content_rate = content_drops / n_trials
        context_rate = context_drops / n_trials
        style_rate = style_drops / n_trials

        assert abs(content_rate - 0.2) < 0.05, f"Content drop rate {content_rate:.3f}, expected ~0.2"
        assert abs(context_rate - 0.3) < 0.05, f"Context drop rate {context_rate:.3f}, expected ~0.3"
        assert abs(style_rate - 0.3) < 0.05, f"Style drop rate {style_rate:.3f}, expected ~0.3"

    def test_independent_dropout(self) -> None:
        """Content, context, and style should be dropped independently."""
        cfg = _make_cfg(content_drop_prob=0.5, context_drop_prob=0.5, style_drop_prob=0.5)

        # Run many trials and check that drops are independent
        both_zero = 0
        content_zero_only = 0
        n_trials = 1000

        for _ in range(n_trials):
            content = torch.randn(1, T, CONTENT_DIM)
            context = torch.randn(1, T, MEL_DIM)
            style = torch.randn(1, STYLE_DIM)

            c_out, ctx_out, s_out = cfg.apply_training_dropout(content, context, style)

            c_dropped = torch.allclose(c_out, torch.zeros_like(c_out))
            ctx_dropped = torch.allclose(ctx_out, torch.zeros_like(ctx_out))

            if c_dropped and ctx_dropped:
                both_zero += 1
            if c_dropped and not ctx_dropped:
                content_zero_only += 1

        # If independent: P(both) ~ 0.25, P(c only) ~ 0.25
        # Should see both patterns with non-trivial frequency
        assert both_zero > 50, "Both conditions should be dropped together sometimes"
        assert content_zero_only > 50, "Content should be dropped alone sometimes"

    # ------------------------------------------------------------------
    # Guided velocity tests
    # ------------------------------------------------------------------

    def test_guided_velocity_alpha_zero(self) -> None:
        """With guidance_strength=0, output should equal the conditional velocity."""
        cfg = _make_cfg(guidance_strength=0.0)
        content, context, style = _random_conditions()
        x_t = torch.randn(B, T, MEL_DIM)
        t = torch.rand(B)

        expected = torch.randn(B, T, MEL_DIM)

        def velocity_fn(xt, ti, c, ctx, s):
            return expected

        v_guided = cfg.guided_velocity(velocity_fn, x_t, t, content, context, style)
        torch.testing.assert_close(v_guided, expected, atol=1e-6, rtol=1e-6)

    def test_guided_velocity_formula(self) -> None:
        """CFG formula: v_cfg = (1 + alpha) * v_cond - alpha * v_uncond."""
        cfg = _make_cfg(guidance_strength=2.0)

        torch.manual_seed(42)
        content = torch.randn(B, T, CONTENT_DIM)
        context = torch.randn(B, T, MEL_DIM)
        style = torch.randn(B, STYLE_DIM)
        x_t = torch.randn(B, T, MEL_DIM)
        t = torch.rand(B)

        # Build a velocity_fn that returns different values for cond vs uncond
        v_cond = torch.randn(B, T, MEL_DIM)
        v_uncond = torch.randn(B, T, MEL_DIM)

        def velocity_fn(xt, ti, c, ctx, s):
            # First B samples are conditional, last B are unconditional
            out = torch.zeros(xt.shape[0], T, MEL_DIM)
            out[:B] = v_cond
            out[B:] = v_uncond
            return out

        v_guided = cfg.guided_velocity(velocity_fn, x_t, t, content, context, style)

        alpha = 2.0
        expected = (1.0 + alpha) * v_cond - alpha * v_uncond
        torch.testing.assert_close(v_guided, expected, atol=1e-5, rtol=1e-5)

    def test_guided_velocity_batch_doubling(self) -> None:
        """velocity_fn should be called once with batch size 2B (for alpha != 0)."""
        cfg = _make_cfg(guidance_strength=2.0)
        content, context, style = _random_conditions()
        x_t = torch.randn(B, T, MEL_DIM)
        t = torch.rand(B)

        call_batch_sizes = []

        def velocity_fn(xt, ti, c, ctx, s):
            call_batch_sizes.append(xt.shape[0])
            return torch.zeros_like(xt)

        cfg.guided_velocity(velocity_fn, x_t, t, content, context, style)

        assert len(call_batch_sizes) == 1, "velocity_fn should be called exactly once"
        assert call_batch_sizes[0] == 2 * B, (
            f"Batch should be doubled: expected {2 * B}, got {call_batch_sizes[0]}"
        )

    def test_zero_condition_is_zeros(self) -> None:
        """The unconditional input should be all-zeros tensors."""
        cfg = _make_cfg(guidance_strength=2.0)
        content, context, style = _random_conditions()
        x_t = torch.randn(B, T, MEL_DIM)
        t = torch.rand(B)

        captured_inputs = {}

        def velocity_fn(xt, ti, c, ctx, s):
            captured_inputs["content"] = c
            captured_inputs["context"] = ctx
            captured_inputs["style"] = s
            return torch.zeros_like(xt)

        cfg.guided_velocity(velocity_fn, x_t, t, content, context, style)

        # Second half of the doubled batch should be zeros
        assert torch.allclose(
            captured_inputs["content"][B:],
            torch.zeros(B, T, CONTENT_DIM),
        ), "Unconditional content should be zeros"
        assert torch.allclose(
            captured_inputs["context"][B:],
            torch.zeros(B, T, MEL_DIM),
        ), "Unconditional context should be zeros"
        assert torch.allclose(
            captured_inputs["style"][B:],
            torch.zeros(B, STYLE_DIM),
        ), "Unconditional style should be zeros"

    def test_dropout_per_sample(self) -> None:
        """Different samples in the batch can have different drop decisions."""
        cfg = _make_cfg(content_drop_prob=0.5)
        content, context, style = _random_conditions()

        # Run dropout and check that not all samples are treated the same
        # (statistically improbable with large B and 50% drop rate)
        found_mixed = False
        for _ in range(20):
            c_out, _, _ = cfg.apply_training_dropout(content, context, style)
            # Check per-sample: some zero, some non-zero
            sample_norms = c_out.view(B, -1).norm(dim=-1)
            has_zeros = (sample_norms == 0).any()
            has_nonzeros = (sample_norms > 0).any()
            if has_zeros and has_nonzeros:
                found_mixed = True
                break

        assert found_mixed, (
            "With 50% drop rate, should find some batches with mixed drop/keep"
        )

    def test_deterministic_with_seed(self) -> None:
        """With a fixed random seed, dropout should be reproducible."""
        cfg = _make_cfg(content_drop_prob=0.5, context_drop_prob=0.5, style_drop_prob=0.5)
        content, context, style = _random_conditions()

        torch.manual_seed(123)
        c1, ctx1, s1 = cfg.apply_training_dropout(content, context, style)

        torch.manual_seed(123)
        c2, ctx2, s2 = cfg.apply_training_dropout(content, context, style)

        torch.testing.assert_close(c1, c2, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(ctx1, ctx2, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(s1, s2, atol=1e-7, rtol=1e-7)

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_invalid_content_drop_prob(self) -> None:
        """content_drop_prob outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError):
            ClassifierFreeGuidance(content_drop_prob=1.5)

    def test_invalid_context_drop_prob(self) -> None:
        """context_drop_prob outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError):
            ClassifierFreeGuidance(context_drop_prob=-0.1)

    def test_invalid_style_drop_prob(self) -> None:
        """style_drop_prob outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError):
            ClassifierFreeGuidance(style_drop_prob=2.0)
