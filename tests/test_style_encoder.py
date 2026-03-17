"""Tests for the WavLM-TDNN style encoder.

All tests mock the WavLM backbone to avoid downloading the pretrained model.
Small dimensions are used for speed: hidden_size=64, tdnn_channels=32.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from stylestream.stylizer.style_encoder import (
    AttentiveStatisticsPooling,
    StyleEncoder,
    TDNNBlock,
)

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

B = 2
HIDDEN = 64          # WavLM hidden size (mocked)
TDNN_CHANNELS = 32
OUTPUT_SIZE = 64
NUM_WAVLM_LAYERS = 13
ATTN_SIZE = 16


# ------------------------------------------------------------------
# Mock WavLM
# ------------------------------------------------------------------


class MockWavLMOutput:
    """Mimics HuggingFace WavLMModel output with hidden_states."""

    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states
        self.last_hidden_state = hidden_states[-1]


class MockWavLMConfig:
    """Mimics WavLM config object."""

    def __init__(self, hidden_size: int = HIDDEN, num_layers: int = 12) -> None:
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_layers


class MockWavLM(nn.Module):
    """A lightweight mock of WavLMModel that returns fake hidden states."""

    def __init__(
        self,
        hidden_size: int = HIDDEN,
        num_layers: int = NUM_WAVLM_LAYERS,
    ) -> None:
        super().__init__()
        self.config = MockWavLMConfig(hidden_size, num_layers - 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # A dummy parameter so the model is non-empty
        self.dummy = nn.Linear(1, 1)

    def forward(
        self,
        input_values: torch.Tensor,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> MockWavLMOutput:
        """Return fake hidden states based on waveform length."""
        B = input_values.shape[0]
        # Approximate WavLM frame rate: ~320 samples per frame
        T = max(1, input_values.shape[1] // 320)
        hidden_states = tuple(
            torch.randn(B, T, self.hidden_size)
            for _ in range(self.num_layers)
        )
        return MockWavLMOutput(hidden_states)


@pytest.fixture
def mock_wavlm():
    """Create a MockWavLM instance."""
    return MockWavLM(hidden_size=HIDDEN, num_layers=NUM_WAVLM_LAYERS)


def _patch_wavlm_loading():
    """Return a context manager that patches StyleEncoder._load_wavlm to return MockWavLM.

    The mock respects the ``freeze`` flag, freezing all parameters when True.
    """
    def mock_load(model_id, freeze):
        mock = MockWavLM(hidden_size=HIDDEN, num_layers=NUM_WAVLM_LAYERS)
        if freeze:
            for param in mock.parameters():
                param.requires_grad_(False)
        return mock

    return patch.object(
        StyleEncoder,
        "_load_wavlm",
        staticmethod(mock_load),
    )


def _make_style_encoder() -> StyleEncoder:
    """Create a StyleEncoder with mocked WavLM."""
    torch.manual_seed(42)
    with _patch_wavlm_loading():
        return StyleEncoder(
            wavlm_model_id="mock/wavlm",
            hidden_size=HIDDEN,
            num_wavlm_layers=NUM_WAVLM_LAYERS,
            tdnn_channels=TDNN_CHANNELS,
            output_size=OUTPUT_SIZE,
            freeze_wavlm=True,
        )


# ======================================================================
# TDNNBlock Tests
# ======================================================================


class TestTDNNBlock:
    """Tests for the TDNN block (Conv1d + ReLU + BatchNorm)."""

    def test_output_shape(self) -> None:
        """Output should have the correct number of channels."""
        tdnn = TDNNBlock(in_channels=HIDDEN, out_channels=TDNN_CHANNELS, kernel_size=5, dilation=1)
        x = torch.randn(B, HIDDEN, 50)  # (B, C, T) channels-first
        out = tdnn(x)
        assert out.shape == (B, TDNN_CHANNELS, 50)

    def test_sequence_length_preserved(self) -> None:
        """Same-padding should preserve the temporal dimension."""
        for kernel, dilation in [(5, 1), (3, 2), (3, 3), (1, 1)]:
            tdnn = TDNNBlock(
                in_channels=HIDDEN,
                out_channels=TDNN_CHANNELS,
                kernel_size=kernel,
                dilation=dilation,
            )
            x = torch.randn(B, HIDDEN, 40)  # (B, C, T) channels-first
            out = tdnn(x)
            assert out.shape[2] == 40, (
                f"k={kernel}, d={dilation}: expected T=40, got T={out.shape[2]}"
            )

    def test_gradient_flow(self) -> None:
        """Gradients should propagate through the TDNN block."""
        tdnn = TDNNBlock(in_channels=HIDDEN, out_channels=TDNN_CHANNELS)
        x = torch.randn(B, HIDDEN, 30, requires_grad=True)  # (B, C, T) channels-first
        out = tdnn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input should receive gradients"
        for name, param in tdnn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ======================================================================
# AttentiveStatisticsPooling Tests
# ======================================================================


class TestAttentiveStatisticsPooling:
    """Tests for attentive statistics pooling."""

    def test_output_shape(self) -> None:
        """Output should be (B, output_size) from (B, T, input_size) input."""
        pool = AttentiveStatisticsPooling(
            input_size=TDNN_CHANNELS,
            attention_size=ATTN_SIZE,
            output_size=OUTPUT_SIZE,
        )
        x = torch.randn(B, 50, TDNN_CHANNELS)
        out = pool(x)
        assert out.shape == (B, OUTPUT_SIZE)

    @pytest.mark.parametrize("seq_len", [10, 50, 100, 200])
    def test_variable_length(self, seq_len: int) -> None:
        """Pooling should handle different temporal dimensions."""
        pool = AttentiveStatisticsPooling(
            input_size=TDNN_CHANNELS,
            attention_size=ATTN_SIZE,
            output_size=OUTPUT_SIZE,
        )
        x = torch.randn(B, seq_len, TDNN_CHANNELS)
        out = pool(x)
        assert out.shape == (B, OUTPUT_SIZE)

    def test_attention_weights_sum_to_one(self) -> None:
        """Softmax attention weights should sum to 1 over the time dimension."""
        pool = AttentiveStatisticsPooling(
            input_size=TDNN_CHANNELS,
            attention_size=ATTN_SIZE,
            output_size=OUTPUT_SIZE,
        )
        x = torch.randn(B, 50, TDNN_CHANNELS)

        # Extract attention weights manually
        attn_scores = pool.attention(x)  # (B, T, 1)
        alpha = torch.nn.functional.softmax(attn_scores, dim=1)

        # Check sum over time is ~1.0
        sums = alpha.sum(dim=1).squeeze(-1)  # (B,)
        torch.testing.assert_close(
            sums,
            torch.ones(B),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_gradient_flow(self) -> None:
        """Gradients should propagate through the pooling module."""
        pool = AttentiveStatisticsPooling(
            input_size=TDNN_CHANNELS,
            attention_size=ATTN_SIZE,
            output_size=OUTPUT_SIZE,
        )
        x = torch.randn(B, 50, TDNN_CHANNELS, requires_grad=True)
        out = pool(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input should receive gradients"

    def test_length_masking(self) -> None:
        """When lengths are provided, padded positions should be masked."""
        pool = AttentiveStatisticsPooling(
            input_size=TDNN_CHANNELS,
            attention_size=ATTN_SIZE,
            output_size=OUTPUT_SIZE,
        )
        pool.eval()

        # Create input with padding
        x = torch.randn(2, 50, TDNN_CHANNELS)
        lengths = torch.tensor([30, 40])

        out_masked = pool(x, lengths=lengths)
        assert out_masked.shape == (2, OUTPUT_SIZE)
        assert torch.isfinite(out_masked).all()


# ======================================================================
# StyleEncoder Tests
# ======================================================================


class TestStyleEncoder:
    """Tests for the full StyleEncoder module with mocked WavLM."""

    def test_output_shape(self) -> None:
        """Output should be (B, output_size) from waveform input."""
        encoder = _make_style_encoder()
        # Waveform at 16kHz, ~1 second
        waveform = torch.randn(B, 16000)
        out = encoder(waveform)
        assert out.shape == (B, OUTPUT_SIZE)

    def test_layer_weights_learnable(self) -> None:
        """The layer_weights parameter should be learnable (requires_grad=True)."""
        encoder = _make_style_encoder()
        assert encoder.layer_weights.requires_grad, (
            "layer_weights should be a learnable parameter"
        )
        assert encoder.layer_weights.shape == (NUM_WAVLM_LAYERS,)

    def test_wavlm_frozen(self) -> None:
        """WavLM parameters should be frozen (requires_grad=False)."""
        encoder = _make_style_encoder()
        for name, param in encoder.wavlm.named_parameters():
            assert not param.requires_grad, (
                f"WavLM param {name} should be frozen"
            )

    @pytest.mark.parametrize("num_samples", [8000, 16000, 32000])
    def test_different_audio_lengths(self, num_samples: int) -> None:
        """Encoder should handle variable-length audio input."""
        encoder = _make_style_encoder()
        waveform = torch.randn(B, num_samples)
        out = encoder(waveform)
        assert out.shape == (B, OUTPUT_SIZE)
        assert torch.isfinite(out).all()

    def test_layer_aggregation_weights_normalized(self) -> None:
        """After softmax, layer aggregation weights should sum to 1."""
        encoder = _make_style_encoder()
        weights = torch.nn.functional.softmax(encoder.layer_weights, dim=0)
        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_trainable_parameter_count(self) -> None:
        """Trainable params should include layer_weights, TDNN, and pooling, but not WavLM."""
        encoder = _make_style_encoder()
        n_trainable = encoder.num_parameters(trainable_only=True)
        n_total = encoder.num_parameters(trainable_only=False)

        assert n_trainable > 0, "Should have some trainable parameters"
        assert n_trainable < n_total, (
            "Total params should exceed trainable (WavLM is frozen)"
        )

    def test_output_finite(self) -> None:
        """All output values should be finite (no nan/inf)."""
        encoder = _make_style_encoder()
        waveform = torch.randn(B, 16000)
        out = encoder(waveform)
        assert torch.isfinite(out).all(), "Output contains nan/inf"
