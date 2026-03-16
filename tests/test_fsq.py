"""Tests for Finite Scalar Quantization (FSQ).

All tests are self-contained and use synthetic tensors (torch.randn) so no
external data is required.  Smaller dimensions (hidden_size=64) are used
throughout to keep tests fast on CPU.
"""

from __future__ import annotations

import pytest
import torch

from stylestream.destylizer.fsq import FSQ

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 2       # batch size
T = 20      # sequence length
H = 64      # hidden_size (small for tests)
DEFAULT_LEVELS = [5, 3, 3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def fsq() -> FSQ:
    """Return a default FSQ instance with levels=[5,3,3], hidden_size=64."""
    torch.manual_seed(42)
    return FSQ(levels=DEFAULT_LEVELS, hidden_size=H)


def _random_input(batch: int = B, seq_len: int = T, hidden: int = H) -> torch.Tensor:
    """Return a random input tensor of shape (B, T, hidden_size)."""
    torch.manual_seed(42)
    return torch.randn(batch, seq_len, hidden)


# ---------------------------------------------------------------------------
# 1. codebook_size
# ---------------------------------------------------------------------------


class TestFSQ:
    @pytest.mark.parametrize(
        "levels, expected",
        [
            ([5, 3, 3], 45),
            ([3, 3, 3], 27),
            ([5, 5, 5], 125),
            ([3], 3),
            ([7, 5], 35),
        ],
    )
    def test_codebook_size(self, levels: list[int], expected: int) -> None:
        """Product of levels determines codebook size."""
        model = FSQ(levels=levels, hidden_size=H)
        assert model.codebook_size == expected

    # -----------------------------------------------------------------------
    # 2. num_dimensions
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "levels, expected",
        [
            ([5, 3, 3], 3),
            ([3, 3], 2),
            ([7, 5, 5, 5, 5], 5),
            ([3], 1),
        ],
    )
    def test_num_dimensions(self, levels: list[int], expected: int) -> None:
        """num_dimensions equals len(levels)."""
        model = FSQ(levels=levels, hidden_size=H)
        assert model.num_dimensions == expected

    # -----------------------------------------------------------------------
    # 3. output_shape
    # -----------------------------------------------------------------------

    def test_output_shape(self, fsq: FSQ) -> None:
        """Forward must return (B, T, H) quantized tensor and an info dict."""
        x = _random_input()
        quantized, info = fsq(x)

        assert quantized.shape == (B, T, H)
        assert isinstance(info, dict)

    # -----------------------------------------------------------------------
    # 4. quantized_values_discrete
    # -----------------------------------------------------------------------

    def test_quantized_values_discrete(self, fsq: FSQ) -> None:
        """After down projection + quantize, values are integers in correct ranges.

        Dim 0 (level 5): {-2, -1, 0, 1, 2}
        Dim 1 (level 3): {-1, 0, 1}
        Dim 2 (level 3): {-1, 0, 1}
        """
        x = _random_input()
        z = fsq.down_proj(x)       # (B, T, 3)
        z_q = fsq.quantize(z)      # (B, T, 3)

        # All values should be integer-valued
        assert torch.allclose(z_q, z_q.round()), "Quantized values should be integers"

        # Check ranges per dimension
        valid_vals = [
            {-2, -1, 0, 1, 2},  # level 5
            {-1, 0, 1},          # level 3
            {-1, 0, 1},          # level 3
        ]
        for d, expected_set in enumerate(valid_vals):
            unique = set(z_q[:, :, d].detach().long().flatten().tolist())
            assert unique.issubset(expected_set), (
                f"Dim {d}: found {unique}, expected subset of {expected_set}"
            )

    # -----------------------------------------------------------------------
    # 5. STE gradient exists
    # -----------------------------------------------------------------------

    def test_ste_gradient_exists(self, fsq: FSQ) -> None:
        """loss.backward() should produce non-zero gradients on down_proj."""
        x = _random_input()
        x.requires_grad_(True)
        quantized, _info = fsq(x)
        loss = quantized.sum()
        loss.backward()

        assert fsq.down_proj.weight.grad is not None
        assert (fsq.down_proj.weight.grad != 0).any(), (
            "Gradients on down_proj weight should be non-zero"
        )

    # -----------------------------------------------------------------------
    # 6. STE gradient identity
    # -----------------------------------------------------------------------

    def test_ste_gradient_identity(self, fsq: FSQ) -> None:
        """Gradient through quantization is identity (STE property).

        The STE formula: z_q = z + (round(clamp(z)) - z).detach()
        So d(z_q)/d(z) = 1 (the stop_gradient on the rounding residual).
        """
        torch.manual_seed(42)
        z = torch.randn(B, T, fsq.num_dimensions, requires_grad=True)
        z_q = fsq.quantize(z)

        # Use a simple scalar output for gradient checking
        loss = z_q.sum()
        loss.backward()

        # STE: gradient should be 1.0 for each element
        expected_grad = torch.ones_like(z)
        torch.testing.assert_close(
            z.grad, expected_grad,
            atol=1e-6, rtol=1e-6,
            msg="STE gradient should be identity (ones)",
        )

    # -----------------------------------------------------------------------
    # 7. codes_to_indices roundtrip
    # -----------------------------------------------------------------------

    def test_codes_to_indices_roundtrip(self, fsq: FSQ) -> None:
        """codes -> indices -> codes should recover the original codes."""
        x = _random_input()
        z = fsq.down_proj(x)
        z_q = fsq.quantize(z)  # (B, T, D) integer-valued codes

        indices = fsq.codes_to_indices(z_q)       # (B, T)
        recovered = fsq.indices_to_codes(indices)  # (B, T, D)

        torch.testing.assert_close(
            recovered, z_q,
            atol=1e-6, rtol=1e-6,
            msg="Roundtrip codes -> indices -> codes should be lossless",
        )

    # -----------------------------------------------------------------------
    # 8. indices_range
    # -----------------------------------------------------------------------

    def test_indices_range(self, fsq: FSQ) -> None:
        """All indices must be in [0, codebook_size)."""
        x = _random_input()
        _quantized, info = fsq(x)
        indices = info["indices"]

        assert (indices >= 0).all(), "Negative index found"
        assert (indices < fsq.codebook_size).all(), (
            f"Index >= codebook_size ({fsq.codebook_size}) found"
        )

    # -----------------------------------------------------------------------
    # 9. codebook_usage nonzero
    # -----------------------------------------------------------------------

    def test_codebook_usage_nonzero(self, fsq: FSQ) -> None:
        """With random input, at least some codebook entries should be used."""
        x = _random_input(batch=4, seq_len=50)
        _quantized, info = fsq(x)

        assert info["codebook_usage"] > 0, "Codebook usage should be > 0"

    # -----------------------------------------------------------------------
    # 10. perplexity range
    # -----------------------------------------------------------------------

    def test_perplexity_range(self, fsq: FSQ) -> None:
        """Perplexity should be in [1, codebook_size]."""
        x = _random_input(batch=4, seq_len=50)
        _quantized, info = fsq(x)

        perplexity = info["perplexity"]
        assert 1.0 <= perplexity <= fsq.codebook_size, (
            f"Perplexity {perplexity} not in [1, {fsq.codebook_size}]"
        )

    # -----------------------------------------------------------------------
    # 11. info dict keys
    # -----------------------------------------------------------------------

    def test_info_dict_keys(self, fsq: FSQ) -> None:
        """Returned info dict must contain the expected keys."""
        x = _random_input()
        _quantized, info = fsq(x)

        expected_keys = {"indices", "codebook_usage", "perplexity", "pre_quant"}
        assert set(info.keys()) == expected_keys

    # -----------------------------------------------------------------------
    # 12. pre_quant continuous
    # -----------------------------------------------------------------------

    def test_pre_quant_continuous(self, fsq: FSQ) -> None:
        """pre_quant values should be continuous (not integer-valued)."""
        x = _random_input()
        _quantized, info = fsq(x)
        pre_quant = info["pre_quant"]

        # Continuous values should differ from their rounded versions
        # (extremely unlikely to be all integers with random data)
        residual = (pre_quant - pre_quant.round()).abs()
        assert residual.sum() > 0, "pre_quant should contain non-integer values"

    # -----------------------------------------------------------------------
    # 13. different levels
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "levels",
        [
            [3, 3],
            [7, 5, 5, 5, 5],
            [5],
            [9, 7, 3],
            [3, 3, 3, 3],
        ],
    )
    def test_different_levels(self, levels: list[int]) -> None:
        """FSQ should work correctly with various level configurations."""
        torch.manual_seed(42)
        model = FSQ(levels=levels, hidden_size=H)
        x = _random_input()
        quantized, info = model(x)

        assert quantized.shape == (B, T, H)
        assert info["indices"].shape == (B, T)
        assert (info["indices"] >= 0).all()
        assert (info["indices"] < model.codebook_size).all()

    # -----------------------------------------------------------------------
    # 14. deterministic
    # -----------------------------------------------------------------------

    def test_deterministic(self, fsq: FSQ) -> None:
        """Same input should produce identical output in eval mode."""
        fsq.eval()
        x = _random_input()

        q1, info1 = fsq(x)
        q2, info2 = fsq(x)

        torch.testing.assert_close(q1, q2, msg="Outputs should be deterministic")
        torch.testing.assert_close(
            info1["indices"], info2["indices"],
            msg="Indices should be deterministic",
        )

    # -----------------------------------------------------------------------
    # 15. gradient_flows_to_input
    # -----------------------------------------------------------------------

    def test_gradient_flows_to_input(self, fsq: FSQ) -> None:
        """Gradients should propagate back through FSQ to the input tensor."""
        x = _random_input()
        x.requires_grad_(True)

        quantized, _info = fsq(x)
        loss = quantized.sum()
        loss.backward()

        assert x.grad is not None, "Input should receive gradients"
        assert (x.grad != 0).any(), "Input gradients should be non-zero"
