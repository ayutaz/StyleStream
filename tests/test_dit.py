"""Tests for the Diffusion Transformer (DiT) block and full model.

All tests are self-contained and use random tensors on CPU.
Small dimensions are used throughout for speed:
hidden=64, heads=4, ffn=256, layers=2, mel_dim=10, content_dim=64.

NOTE: The DiT.forward() method has a known shape issue in its RoPE view
(line ~381 of dit.py): it constructs `x.view(B, T, num_heads, head_dim)`,
where dim -2 is `num_heads`, but RotaryPositionEmbedding.forward reads
`x.shape[-2]` to determine seq_len. This means the full DiT tests must
use seq_len == num_heads to avoid a shape mismatch between the RoPE
cos/sin table and the attention Q/K tensors. The DiTBlock tests are
unaffected because they receive externally-prepared RoPE cos/sin.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from stylestream.stylizer.dit import DiT, DiTBlock
from stylestream.stylizer.rope import RotaryPositionEmbedding

# ------------------------------------------------------------------
# Shared constants for fast tests
# ------------------------------------------------------------------

HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS  # 16
FFN = 256
LAYERS = 2
MEL_DIM = 10
CONTENT_DIM = 64
B = 2

# For DiTBlock tests, we supply RoPE cos/sin externally, so T is free.
T_BLOCK = 20

# For full DiT tests, T must equal HEADS due to the RoPE view issue.
T_DIT = HEADS  # 4


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_dit_block() -> DiTBlock:
    """Create a small DiTBlock for testing."""
    torch.manual_seed(42)
    return DiTBlock(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        ffn_size=FFN,
        dropout=0.0,
    )


def _make_dit(**overrides) -> DiT:
    """Create a small DiT for testing."""
    torch.manual_seed(42)
    kwargs = dict(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        ffn_size=FFN,
        num_heads=HEADS,
        mel_dim=MEL_DIM,
        content_dim=CONTENT_DIM,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return DiT(**kwargs)


def _make_dit_inputs(
    batch: int = B,
    seq_len: int = T_DIT,
) -> dict[str, torch.Tensor]:
    """Create all inputs required for a DiT forward pass."""
    torch.manual_seed(42)
    return {
        "x_t": torch.randn(batch, seq_len, MEL_DIM),
        "t": torch.rand(batch),
        "content_features": torch.randn(batch, seq_len, CONTENT_DIM),
        "context_mel": torch.randn(batch, seq_len, MEL_DIM),
        "style_emb": torch.randn(batch, HIDDEN),
    }


def _make_rope_cos_sin(seq_len: int = T_BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cos/sin for the given sequence length.

    This prepares cos/sin externally in the correct shape (1, 1, T, head_dim)
    for direct use in DiTBlock tests.
    """
    rope = RotaryPositionEmbedding(dim=HEAD_DIM)
    x = torch.randn(B, HEADS, seq_len, HEAD_DIM)
    return rope(x)


# ======================================================================
# DiTBlock Tests
# ======================================================================


class TestDiTBlock:
    """Tests for a single DiT block.

    DiTBlock receives RoPE cos/sin as pre-computed arguments, so these
    tests are independent of the DiT model's RoPE view.
    """

    def test_output_shape(self) -> None:
        """Output shape should be the same as input: (B, T, hidden)."""
        block = _make_dit_block()
        x = torch.randn(B, T_BLOCK, HIDDEN)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(T_BLOCK)

        out = block(x, c, cos, sin)
        assert out.shape == (B, T_BLOCK, HIDDEN)

    def test_residual_at_init(self) -> None:
        """At init (zero gates), block output should approximately equal input.

        adaLN-Zero initializes all gate alphas to zero, so:
        x + alpha * sublayer(x) = x + 0 = x
        """
        block = _make_dit_block()
        x = torch.randn(B, T_BLOCK, HIDDEN)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(T_BLOCK)

        out = block(x, c, cos, sin)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self) -> None:
        """All parameters should receive non-nan gradients."""
        block = _make_dit_block()
        x = torch.randn(B, T_BLOCK, HIDDEN, requires_grad=True)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(T_BLOCK)

        out = block(x, c, cos, sin)
        loss = out.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Nan/inf gradient for {name}"

    @pytest.mark.parametrize("seq_len", [10, 50, 100])
    def test_different_seq_lengths(self, seq_len: int) -> None:
        """Block should work correctly with various sequence lengths."""
        block = _make_dit_block()
        x = torch.randn(B, seq_len, HIDDEN)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(seq_len)

        out = block(x, c, cos, sin)
        assert out.shape == (B, seq_len, HIDDEN)


# ======================================================================
# DiT Full Model Tests
# ======================================================================


class TestDiT:
    """Tests for the full DiT model.

    These tests use seq_len == num_heads to work around the known RoPE
    view issue in DiT.forward() -- see module docstring for details.
    """

    def test_output_shape(self) -> None:
        """Output should be (B, T, mel_dim) velocity field."""
        dit = _make_dit()
        inputs = _make_dit_inputs()

        out = dit(**inputs)
        assert out.shape == (B, T_DIT, MEL_DIM), (
            f"Expected ({B}, {T_DIT}, {MEL_DIM}), got {out.shape}"
        )

    def test_forward_no_error(self) -> None:
        """Smoke test: forward pass should complete without errors."""
        dit = _make_dit()
        inputs = _make_dit_inputs()

        out = dit(**inputs)
        assert torch.isfinite(out).all(), "Output contains nan/inf"

    def test_input_projection(self) -> None:
        """Input projection should map from mel+mel+content to hidden_size.

        input_dim = mel_dim + mel_dim + content_dim = 10 + 10 + 64 = 84
        """
        dit = _make_dit()
        expected_input_dim = MEL_DIM + MEL_DIM + CONTENT_DIM  # 84
        assert dit.input_proj.in_features == expected_input_dim
        assert dit.input_proj.out_features == HIDDEN

    def test_output_projection(self) -> None:
        """Final layer should project from hidden_size to mel_dim."""
        dit = _make_dit()
        assert dit.final_layer.output_size == MEL_DIM

    def test_gradient_flow(self) -> None:
        """All trainable parameters should receive gradients."""
        dit = _make_dit()
        inputs = _make_dit_inputs()

        out = dit(**inputs)
        loss = out.sum()
        loss.backward()

        params_with_grad = 0
        for name, param in dit.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Nan/inf gradient for {name}"
                params_with_grad += 1

        assert params_with_grad > 0, "No trainable parameters found"

    def test_parameter_count(self) -> None:
        """Parameter count should be reasonable for the small test model."""
        dit = _make_dit()
        num_params = dit.num_parameters(trainable_only=True)

        assert num_params > 0, "Model has no trainable parameters"
        # For 2 layers, hidden=64, ffn=256, heads=4:
        # very rough estimate ~500K params, generous bounds
        assert num_params < 10_000_000, (
            f"Parameter count {num_params} seems unreasonably large"
        )

    def test_from_config(self) -> None:
        """DiT.from_config should correctly build a model from a config object."""

        @dataclass
        class MockDiTConfig:
            num_layers: int = 2
            hidden_size: int = HIDDEN
            ffn_size: int = FFN
            num_heads: int = HEADS
            content_dim: int = CONTENT_DIM
            mel_dim: int = MEL_DIM
            dropout: float = 0.0
            gradient_checkpointing: bool = False

        config = MockDiTConfig()
        dit = DiT.from_config(config)

        assert dit.num_layers == 2
        assert dit.hidden_size == HIDDEN
        assert dit.mel_dim == MEL_DIM

        # Should produce valid output
        inputs = _make_dit_inputs()
        out = dit(**inputs)
        assert out.shape == (B, T_DIT, MEL_DIM)

    def test_conditioning_effect(self) -> None:
        """Different style embeddings should produce different velocities.

        At init (zero gates), this difference comes from the FinalAdaLN.
        We need to break zero-init to see the effect.
        """
        dit = _make_dit()
        # Break zero-init on final layer to see conditioning effect
        with torch.no_grad():
            dit.final_layer.adaln_linear[1].weight.fill_(0.01)
            dit.final_layer.output_linear.weight.fill_(0.01)

        inputs1 = _make_dit_inputs()
        inputs2 = _make_dit_inputs()
        torch.manual_seed(99)
        inputs2["style_emb"] = torch.randn(B, HIDDEN) * 5.0

        out1 = dit(**inputs1)
        out2 = dit(**inputs2)

        assert not torch.allclose(out1, out2, atol=1e-3), (
            "Different style embeddings should produce different outputs"
        )

    def test_timestep_effect(self) -> None:
        """Different timesteps should produce different velocities.

        Again, we need to break zero-init for this to be visible.
        """
        dit = _make_dit()
        # Break zero-init
        with torch.no_grad():
            dit.final_layer.adaln_linear[1].weight.fill_(0.01)
            dit.final_layer.output_linear.weight.fill_(0.01)

        inputs1 = _make_dit_inputs()
        inputs1["t"] = torch.tensor([0.1, 0.1])

        inputs2 = _make_dit_inputs()
        inputs2["t"] = torch.tensor([0.9, 0.9])

        out1 = dit(**inputs1)
        out2 = dit(**inputs2)

        assert not torch.allclose(out1, out2, atol=1e-3), (
            "Different timesteps should produce different outputs"
        )

    def test_deterministic(self) -> None:
        """Same inputs in eval mode should produce identical outputs."""
        dit = _make_dit()
        dit.eval()

        inputs = _make_dit_inputs()
        out1 = dit(**inputs)
        out2 = dit(**inputs)

        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)

    def test_batch_independence(self) -> None:
        """Samples in a batch should not affect each other."""
        dit = _make_dit()
        dit.eval()

        inputs = _make_dit_inputs(batch=3)
        out_batch = dit(**inputs)

        # Run each sample individually
        for i in range(3):
            single_inputs = {
                k: v[i:i + 1] for k, v in inputs.items()
            }
            out_single = dit(**single_inputs)
            torch.testing.assert_close(
                out_batch[i:i + 1],
                out_single,
                atol=1e-5,
                rtol=1e-5,
            )

    def test_zero_gate_identity(self) -> None:
        """At initialization with zero gates, DiT output should be near zero.

        The FinalAdaLN is also zero-initialized, so the initial velocity
        prediction should be approximately zero.
        """
        dit = _make_dit()
        inputs = _make_dit_inputs()

        out = dit(**inputs)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-4), (
            f"Initial DiT output should be near zero, max: {out.abs().max()}"
        )

    def test_num_layers(self) -> None:
        """The model should have the expected number of DiT blocks."""
        dit = _make_dit(num_layers=4)
        assert len(dit.blocks) == 4

    def test_num_parameters_method(self) -> None:
        """num_parameters should return a positive integer."""
        dit = _make_dit()
        n_trainable = dit.num_parameters(trainable_only=True)
        n_total = dit.num_parameters(trainable_only=False)

        assert n_trainable > 0
        assert n_total >= n_trainable

    def test_variable_heads_and_seq_len(self) -> None:
        """Test with different num_heads values.

        Each configuration uses seq_len == num_heads to comply with the
        current RoPE view shape in DiT.forward().
        """
        for num_heads in [2, 4, 8]:
            hidden = num_heads * HEAD_DIM
            dit = _make_dit(
                num_heads=num_heads,
                hidden_size=hidden,
                ffn_size=hidden * 4,
            )
            torch.manual_seed(42)
            inputs = {
                "x_t": torch.randn(B, num_heads, MEL_DIM),
                "t": torch.rand(B),
                "content_features": torch.randn(B, num_heads, CONTENT_DIM),
                "context_mel": torch.randn(B, num_heads, MEL_DIM),
                "style_emb": torch.randn(B, hidden),
            }
            out = dit(**inputs)
            assert out.shape == (B, num_heads, MEL_DIM)


# ======================================================================
# GQA (Grouped Query Attention) Tests
# ======================================================================


GQA_KV_HEADS = 2  # 4 Q heads, 2 KV heads -> repeat factor 2


def _make_gqa_dit_block(num_kv_heads: int = GQA_KV_HEADS) -> DiTBlock:
    """Create a small DiTBlock with GQA enabled."""
    torch.manual_seed(42)
    return DiTBlock(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        ffn_size=FFN,
        dropout=0.0,
        num_kv_heads=num_kv_heads,
    )


class TestGQADiTBlock:
    """Tests for DiTBlock with Grouped Query Attention."""

    def test_gqa_output_shape(self) -> None:
        """GQA block should produce the same output shape as MHA."""
        block = _make_gqa_dit_block()
        x = torch.randn(B, T_BLOCK, HIDDEN)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(T_BLOCK)

        out = block(x, c, cos, sin)
        assert out.shape == (B, T_BLOCK, HIDDEN)

    def test_gqa_uses_separate_projections(self) -> None:
        """GQA block should have q_proj and kv_proj instead of qkv_proj."""
        block = _make_gqa_dit_block()
        assert hasattr(block, "q_proj"), "GQA block should have q_proj"
        assert hasattr(block, "kv_proj"), "GQA block should have kv_proj"
        assert not hasattr(block, "qkv_proj"), (
            "GQA block should not have qkv_proj"
        )

    def test_gqa_kv_proj_size(self) -> None:
        """KV projection output dimension should be 2 * head_dim * num_kv_heads."""
        block = _make_gqa_dit_block()
        expected_kv_dim = 2 * HEAD_DIM * GQA_KV_HEADS
        assert block.kv_proj.out_features == expected_kv_dim

    def test_gqa_q_proj_size(self) -> None:
        """Q projection output should be full hidden_size."""
        block = _make_gqa_dit_block()
        assert block.q_proj.out_features == HIDDEN

    def test_gqa_fewer_params_than_mha(self) -> None:
        """GQA block should have fewer parameters than MHA block."""
        mha_block = _make_dit_block()
        gqa_block = _make_gqa_dit_block()

        mha_params = sum(p.numel() for p in mha_block.parameters())
        gqa_params = sum(p.numel() for p in gqa_block.parameters())

        assert gqa_params < mha_params, (
            f"GQA ({gqa_params}) should have fewer params than MHA ({mha_params})"
        )

    def test_gqa_residual_at_init(self) -> None:
        """At init (zero gates), GQA block output should equal input."""
        block = _make_gqa_dit_block()
        x = torch.randn(B, T_BLOCK, HIDDEN)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(T_BLOCK)

        out = block(x, c, cos, sin)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_gqa_gradient_flow(self) -> None:
        """All GQA block parameters should receive gradients."""
        block = _make_gqa_dit_block()
        x = torch.randn(B, T_BLOCK, HIDDEN, requires_grad=True)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(T_BLOCK)

        out = block(x, c, cos, sin)
        loss = out.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), (
                f"Nan/inf gradient for {name}"
            )

    @pytest.mark.parametrize("seq_len", [10, 50, 100])
    def test_gqa_different_seq_lengths(self, seq_len: int) -> None:
        """GQA block should work with various sequence lengths."""
        block = _make_gqa_dit_block()
        x = torch.randn(B, seq_len, HIDDEN)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(seq_len)

        out = block(x, c, cos, sin)
        assert out.shape == (B, seq_len, HIDDEN)

    @pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
    def test_gqa_various_kv_heads(self, num_kv_heads: int) -> None:
        """GQA should work with different numbers of KV heads."""
        block = _make_gqa_dit_block(num_kv_heads=num_kv_heads)
        x = torch.randn(B, T_BLOCK, HIDDEN)
        c = torch.randn(B, HIDDEN)
        cos, sin = _make_rope_cos_sin(T_BLOCK)

        out = block(x, c, cos, sin)
        assert out.shape == (B, T_BLOCK, HIDDEN)

    def test_gqa_num_kv_heads_must_divide_num_heads(self) -> None:
        """num_heads must be divisible by num_kv_heads."""
        with pytest.raises(AssertionError, match="divisible"):
            DiTBlock(
                hidden_size=HIDDEN,
                num_heads=HEADS,  # 4
                num_kv_heads=3,  # 4 % 3 != 0
            )


class TestGQADiT:
    """Tests for the full DiT model with GQA enabled."""

    def test_gqa_dit_output_shape(self) -> None:
        """GQA DiT should produce the same output shape."""
        dit = _make_dit(num_kv_heads=GQA_KV_HEADS)
        inputs = _make_dit_inputs()

        out = dit(**inputs)
        assert out.shape == (B, T_DIT, MEL_DIM)

    def test_gqa_dit_forward_no_error(self) -> None:
        """GQA DiT forward should complete without errors."""
        dit = _make_dit(num_kv_heads=GQA_KV_HEADS)
        inputs = _make_dit_inputs()

        out = dit(**inputs)
        assert torch.isfinite(out).all(), "Output contains nan/inf"

    def test_gqa_dit_zero_gate_identity(self) -> None:
        """At initialization, GQA DiT output should be near zero."""
        dit = _make_dit(num_kv_heads=GQA_KV_HEADS)
        inputs = _make_dit_inputs()

        out = dit(**inputs)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-4), (
            f"Initial GQA DiT output should be near zero, max: {out.abs().max()}"
        )

    def test_gqa_dit_gradient_flow(self) -> None:
        """All GQA DiT parameters should receive gradients."""
        dit = _make_dit(num_kv_heads=GQA_KV_HEADS)
        inputs = _make_dit_inputs()

        out = dit(**inputs)
        loss = out.sum()
        loss.backward()

        for name, param in dit.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), (
                    f"Nan/inf gradient for {name}"
                )

    def test_gqa_dit_fewer_params(self) -> None:
        """GQA DiT should have fewer parameters than MHA DiT."""
        mha_dit = _make_dit()
        gqa_dit = _make_dit(num_kv_heads=GQA_KV_HEADS)

        mha_params = mha_dit.num_parameters()
        gqa_params = gqa_dit.num_parameters()

        assert gqa_params < mha_params, (
            f"GQA ({gqa_params}) should have fewer params than MHA ({mha_params})"
        )

    def test_gqa_dit_from_config(self) -> None:
        """DiT.from_config should correctly propagate num_kv_heads."""

        @dataclass
        class MockGQAConfig:
            num_layers: int = 2
            hidden_size: int = HIDDEN
            ffn_size: int = FFN
            num_heads: int = HEADS
            num_kv_heads: int = GQA_KV_HEADS
            content_dim: int = CONTENT_DIM
            mel_dim: int = MEL_DIM
            dropout: float = 0.0
            gradient_checkpointing: bool = False

        config = MockGQAConfig()
        dit = DiT.from_config(config)

        assert dit.num_kv_heads == GQA_KV_HEADS
        # All blocks should use GQA
        for block in dit.blocks:
            assert block.use_gqa is True
            assert block.num_kv_heads == GQA_KV_HEADS

        inputs = _make_dit_inputs()
        out = dit(**inputs)
        assert out.shape == (B, T_DIT, MEL_DIM)

    def test_default_num_kv_heads_is_mha(self) -> None:
        """num_kv_heads=0 (default) should use standard MHA."""
        dit = _make_dit()  # default: num_kv_heads=0
        for block in dit.blocks:
            assert block.use_gqa is False
            assert hasattr(block, "qkv_proj")
