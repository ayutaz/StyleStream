"""Tests for the sinusoidal timestep embedding module.

All tests are self-contained and use random tensors on CPU.
"""

from __future__ import annotations

import pytest
import torch

from stylestream.stylizer.timestep_embedding import TimestepEmbedding

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

B = 4
HIDDEN = 64  # small hidden size for fast tests


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_embedding(hidden_size: int = HIDDEN) -> TimestepEmbedding:
    """Return a TimestepEmbedding instance with deterministic init."""
    torch.manual_seed(42)
    return TimestepEmbedding(hidden_size=hidden_size)


# ======================================================================
# TimestepEmbedding Tests
# ======================================================================


class TestTimestepEmbedding:
    """Tests for the TimestepEmbedding module."""

    def test_output_shape(self) -> None:
        """Output should be (B, hidden_size) from (B,) input."""
        emb = _make_embedding()
        t = torch.tensor([0.0, 0.25, 0.5, 0.75])
        out = emb(t)
        assert out.shape == (4, HIDDEN), f"Expected (4, {HIDDEN}), got {out.shape}"

    def test_different_timesteps_different_embeddings(self) -> None:
        """Different timestep values should produce different embeddings."""
        emb = _make_embedding()
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])

        out1 = emb(t1)
        out2 = emb(t2)

        assert not torch.allclose(out1, out2, atol=1e-4), (
            "t=0.1 and t=0.9 should produce different embeddings"
        )

    def test_deterministic(self) -> None:
        """Same timestep should produce the same output."""
        emb = _make_embedding()
        emb.eval()

        t = torch.tensor([0.3, 0.7])
        out1 = emb(t)
        out2 = emb(t)

        torch.testing.assert_close(out1, out2, atol=1e-7, rtol=1e-7)

    def test_batch_independence(self) -> None:
        """Changing one sample in the batch should not affect others."""
        emb = _make_embedding()
        emb.eval()

        t1 = torch.tensor([0.1, 0.5, 0.9])
        out1 = emb(t1)

        t2 = torch.tensor([0.1, 0.8, 0.9])  # only index 1 changed
        out2 = emb(t2)

        # Samples 0 and 2 should be identical
        torch.testing.assert_close(out1[0], out2[0], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(out1[2], out2[2], atol=1e-7, rtol=1e-7)

        # Sample 1 should differ
        assert not torch.allclose(out1[1], out2[1], atol=1e-4)

    def test_scalar_input(self) -> None:
        """Should work with a scalar (0-dim) timestep input."""
        emb = _make_embedding()
        t = torch.tensor(0.5)
        out = emb(t)

        assert out.shape == (1, HIDDEN), f"Expected (1, {HIDDEN}), got {out.shape}"
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        """MLP parameters should receive gradients through the embedding."""
        emb = _make_embedding()
        t = torch.tensor([0.1, 0.5, 0.9])

        out = emb(t)
        loss = out.sum()
        loss.backward()

        for name, param in emb.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_boundary_values(self) -> None:
        """t=0 and t=1 should not produce nan or inf."""
        emb = _make_embedding()

        for val in [0.0, 1.0]:
            t = torch.tensor([val])
            out = emb(t)
            assert torch.isfinite(out).all(), (
                f"Non-finite values at t={val}: {out}"
            )

    @pytest.mark.parametrize("hidden_size", [128, 256, 512, 768])
    def test_hidden_size_variants(self, hidden_size: int) -> None:
        """TimestepEmbedding should work with various hidden sizes."""
        torch.manual_seed(42)
        emb = TimestepEmbedding(hidden_size=hidden_size)
        t = torch.tensor([0.25, 0.75])
        out = emb(t)

        assert out.shape == (2, hidden_size)
        assert torch.isfinite(out).all()
