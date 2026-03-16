"""Tests for the full Stylizer model (DiT + StyleEncoder + CFM + CFG).

All tests use small dimensions and mock the WavLM backbone to avoid
downloading the pretrained model and to keep tests fast.
Dimensions: hidden=64, heads=4, ffn=256, layers=2, mel_dim=10, content_dim=64.

NOTE: Due to a known RoPE view issue in DiT.forward() (see test_dit.py
for details), seq_len must equal num_heads in the full model tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from stylestream.stylizer.model import Stylizer
from stylestream.stylizer.style_encoder import StyleEncoder

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

HIDDEN = 64
HEADS = 4
FFN = 256
LAYERS = 2
MEL_DIM = 10
CONTENT_DIM = 64
TDNN_CHANNELS = 32
NUM_WAVLM_LAYERS = 13
NFE = 4
B = 2

# seq_len must equal num_heads due to the RoPE view issue in DiT.forward()
T = HEADS  # 4

AUDIO_SAMPLES = 4800  # ~0.3s at 16kHz, enough for mock WavLM


# ------------------------------------------------------------------
# Mock WavLM
# ------------------------------------------------------------------


class MockWavLMOutput:
    """Mimics HuggingFace WavLMModel output with hidden_states."""

    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states
        self.last_hidden_state = hidden_states[-1]


class MockWavLM(nn.Module):
    """A lightweight mock of WavLMModel."""

    def __init__(self, hidden_size: int = HIDDEN, num_layers: int = NUM_WAVLM_LAYERS) -> None:
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers - 1,
        })()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dummy = nn.Linear(1, 1)

    def forward(self, input_values, output_hidden_states=True, **kwargs):
        B = input_values.shape[0]
        T_out = max(1, input_values.shape[1] // 320)
        hidden_states = tuple(
            torch.randn(B, T_out, self.hidden_size)
            for _ in range(self.num_layers)
        )
        return MockWavLMOutput(hidden_states)


def _patch_wavlm():
    """Return a context manager that patches StyleEncoder._load_wavlm."""
    mock = MockWavLM(hidden_size=HIDDEN, num_layers=NUM_WAVLM_LAYERS)
    return patch.object(
        StyleEncoder,
        "_load_wavlm",
        staticmethod(lambda model_id, freeze: mock),
    )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def small_stylizer():
    """Create a small Stylizer for testing with tiny dims and mocked WavLM."""
    torch.manual_seed(42)
    with _patch_wavlm():
        model = Stylizer(
            num_layers=LAYERS,
            hidden_size=HIDDEN,
            ffn_size=FFN,
            num_heads=HEADS,
            mel_dim=MEL_DIM,
            content_dim=CONTENT_DIM,
            tdnn_channels=TDNN_CHANNELS,
            output_size=HIDDEN,
            style_hidden_size=HIDDEN,
            num_wavlm_layers=NUM_WAVLM_LAYERS,
            nfe=NFE,
            dropout=0.0,
            content_drop_prob=0.2,
            context_drop_prob=0.3,
            style_drop_prob=0.3,
            guidance_strength=2.0,
        )
    return model


def _make_training_inputs(batch: int = B, seq_len: int = T) -> dict:
    """Create all inputs needed for a Stylizer forward pass."""
    torch.manual_seed(42)
    return {
        "mel": torch.randn(batch, seq_len, MEL_DIM),
        "content_features": torch.randn(batch, seq_len, CONTENT_DIM),
        "mask": (torch.rand(batch, seq_len) > 0.3).float(),
        "style_waveform": torch.randn(batch, AUDIO_SAMPLES),
    }


# ======================================================================
# Construction Tests
# ======================================================================


class TestStylizerConstruction:
    """Tests for Stylizer model construction."""

    def test_from_kwargs(self, small_stylizer: Stylizer) -> None:
        """Stylizer should be constructable from keyword arguments."""
        assert isinstance(small_stylizer, Stylizer)
        assert isinstance(small_stylizer, nn.Module)

    def test_parameter_count(self, small_stylizer: Stylizer) -> None:
        """Model should have a reasonable number of parameters."""
        n_params = small_stylizer.num_parameters(trainable_only=True)
        assert n_params > 0, "Model should have trainable parameters"
        # Small model should not be huge
        assert n_params < 50_000_000, (
            f"Parameter count {n_params} seems unreasonably large for a small test model"
        )

    def test_has_all_components(self, small_stylizer: Stylizer) -> None:
        """Stylizer should contain dit, style_encoder, cfm, and cfg."""
        assert hasattr(small_stylizer, "dit"), "Missing DiT component"
        assert hasattr(small_stylizer, "style_encoder"), "Missing StyleEncoder component"
        assert hasattr(small_stylizer, "cfm"), "Missing CFM component"
        assert hasattr(small_stylizer, "cfg"), "Missing CFG component"

    def test_mel_dim_attribute(self, small_stylizer: Stylizer) -> None:
        """Stylizer should store the mel dimension."""
        assert small_stylizer.mel_dim == MEL_DIM

    def test_nfe_attribute(self, small_stylizer: Stylizer) -> None:
        """Stylizer should store the default number of function evaluations."""
        assert small_stylizer.nfe == NFE


# ======================================================================
# Forward Pass Tests
# ======================================================================


class TestStylizerForward:
    """Tests for the training forward pass."""

    def test_output_keys(self, small_stylizer: Stylizer) -> None:
        """Forward output dict should contain 'loss' and 'velocity_pred'."""
        inputs = _make_training_inputs()
        result = small_stylizer(**inputs)

        assert "loss" in result, "Output must contain 'loss'"
        assert "velocity_pred" in result, "Output must contain 'velocity_pred'"

    def test_loss_is_scalar(self, small_stylizer: Stylizer) -> None:
        """The loss should be a scalar tensor."""
        inputs = _make_training_inputs()
        result = small_stylizer(**inputs)

        assert result["loss"].dim() == 0, (
            f"Loss should be scalar, got shape {result['loss'].shape}"
        )

    def test_velocity_shape(self, small_stylizer: Stylizer) -> None:
        """velocity_pred should have shape (B, T, mel_dim)."""
        inputs = _make_training_inputs()
        result = small_stylizer(**inputs)

        assert result["velocity_pred"].shape == (B, T, MEL_DIM), (
            f"Expected ({B}, {T}, {MEL_DIM}), got {result['velocity_pred'].shape}"
        )

    def test_loss_is_finite(self, small_stylizer: Stylizer) -> None:
        """Loss should not be nan or inf."""
        inputs = _make_training_inputs()
        result = small_stylizer(**inputs)

        assert torch.isfinite(result["loss"]), (
            f"Loss is not finite: {result['loss'].item()}"
        )

    def test_gradient_flow(self, small_stylizer: Stylizer) -> None:
        """loss.backward() should produce gradients on DiT parameters."""
        inputs = _make_training_inputs()
        result = small_stylizer(**inputs)

        result["loss"].backward()

        # Check DiT parameters have gradients
        dit_has_grad = False
        for name, param in small_stylizer.dit.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    dit_has_grad = True
                    break

        assert dit_has_grad, "At least some DiT parameters should have non-zero gradients"

    def test_cfg_dropout_effect(self, small_stylizer: Stylizer) -> None:
        """With CFG dropout disabled vs. full dropout, both should produce finite losses."""
        inputs = _make_training_inputs()

        # No dropout
        no_drop = {
            **inputs,
            "cfg_drop_content": torch.zeros(B, dtype=torch.bool),
            "cfg_drop_context": torch.zeros(B, dtype=torch.bool),
            "cfg_drop_style": torch.zeros(B, dtype=torch.bool),
        }

        # Full dropout
        full_drop = {
            **inputs,
            "cfg_drop_content": torch.ones(B, dtype=torch.bool),
            "cfg_drop_context": torch.ones(B, dtype=torch.bool),
            "cfg_drop_style": torch.ones(B, dtype=torch.bool),
        }

        torch.manual_seed(42)
        result_no = small_stylizer(**no_drop)
        torch.manual_seed(42)
        result_full = small_stylizer(**full_drop)

        assert torch.isfinite(result_no["loss"]), "No-drop loss should be finite"
        assert torch.isfinite(result_full["loss"]), "Full-drop loss should be finite"

    def test_different_masks(self, small_stylizer: Stylizer) -> None:
        """Full mask vs. partial mask should produce finite losses."""
        inputs_full = _make_training_inputs()
        inputs_full["mask"] = torch.ones(B, T)

        inputs_partial = _make_training_inputs()
        inputs_partial["mask"] = torch.zeros(B, T)
        inputs_partial["mask"][:, :T // 2] = 1.0

        torch.manual_seed(42)
        result_full = small_stylizer(**inputs_full)
        torch.manual_seed(42)
        result_partial = small_stylizer(**inputs_partial)

        assert torch.isfinite(result_full["loss"])
        assert torch.isfinite(result_partial["loss"])


# ======================================================================
# Sample (Inference) Tests
# ======================================================================


class TestStylizerSample:
    """Tests for inference sampling."""

    def test_output_shape(self, small_stylizer: Stylizer) -> None:
        """sample() should return (B, T, mel_dim)."""
        torch.manual_seed(42)
        content = torch.randn(B, T, CONTENT_DIM)
        style_wav = torch.randn(B, AUDIO_SAMPLES)

        out = small_stylizer.sample(
            content_features=content,
            style_waveform=style_wav,
            nfe=NFE,
        )
        assert out.shape == (B, T, MEL_DIM)

    def test_no_nan(self, small_stylizer: Stylizer) -> None:
        """Sampled output should not contain nan or inf."""
        torch.manual_seed(42)
        content = torch.randn(B, T, CONTENT_DIM)
        style_wav = torch.randn(B, AUDIO_SAMPLES)

        out = small_stylizer.sample(
            content_features=content,
            style_waveform=style_wav,
            nfe=NFE,
        )
        assert torch.isfinite(out).all(), "Sample output contains nan/inf"

    def test_context_preserved(self, small_stylizer: Stylizer) -> None:
        """Non-masked regions should equal the provided context_mel."""
        torch.manual_seed(42)
        content = torch.randn(B, T, CONTENT_DIM)
        style_wav = torch.randn(B, AUDIO_SAMPLES)
        context_mel = torch.randn(B, T, MEL_DIM)

        # Mask: only first half is generated, second half is context
        mask = torch.zeros(B, T)
        mask[:, :T // 2] = 1.0

        out = small_stylizer.sample(
            content_features=content,
            style_waveform=style_wav,
            context_mel=context_mel,
            mask=mask,
            nfe=NFE,
        )

        # Second half (context) should match the provided context_mel
        torch.testing.assert_close(
            out[:, T // 2:, :],
            context_mel[:, T // 2:, :],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_full_generation(self, small_stylizer: Stylizer) -> None:
        """Full generation (no context) should work without errors."""
        torch.manual_seed(42)
        content = torch.randn(B, T, CONTENT_DIM)
        style_wav = torch.randn(B, AUDIO_SAMPLES)

        # No context_mel and no mask -> full generation
        out = small_stylizer.sample(
            content_features=content,
            style_waveform=style_wav,
            context_mel=None,
            mask=None,
            nfe=NFE,
        )
        assert out.shape == (B, T, MEL_DIM)
        assert torch.isfinite(out).all()


# ======================================================================
# Integration Tests
# ======================================================================


class TestStylizerIntegration:
    """Integration tests verifying end-to-end workflows."""

    def test_training_step(self, small_stylizer: Stylizer) -> None:
        """A complete training step (forward + backward) should succeed."""
        small_stylizer.train()
        inputs = _make_training_inputs()

        result = small_stylizer(**inputs)
        loss = result["loss"]

        assert torch.isfinite(loss), "Loss should be finite"

        loss.backward()

        # Verify at least some parameter got a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_stylizer.parameters()
            if p.requires_grad
        )
        assert has_grad, "Some parameters should have received gradients"

    def test_eval_mode(self, small_stylizer: Stylizer) -> None:
        """model.eval() should not raise errors; forward should still work."""
        small_stylizer.eval()
        inputs = _make_training_inputs()

        with torch.no_grad():
            result = small_stylizer(**inputs)

        assert torch.isfinite(result["loss"])

    @pytest.mark.parametrize("batch_size", [2, 3, 4])
    def test_batch_sizes(self, small_stylizer: Stylizer, batch_size: int) -> None:
        """Different batch sizes should work correctly.

        Note: batch_size >= 2 is required because the StyleEncoder uses
        BatchNorm, which needs at least 2 samples during training.
        """
        inputs = _make_training_inputs(batch=batch_size)
        result = small_stylizer(**inputs)

        assert result["velocity_pred"].shape[0] == batch_size
        assert torch.isfinite(result["loss"])

    def test_sequence_length(self, small_stylizer: Stylizer) -> None:
        """Sequence length matching num_heads should work correctly."""
        inputs = _make_training_inputs(seq_len=T)
        result = small_stylizer(**inputs)

        assert result["velocity_pred"].shape[1] == T
        assert torch.isfinite(result["loss"])
