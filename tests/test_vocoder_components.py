"""Tests for StyleStream Causal Vocos vocoder components."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from stylestream.vocoder.causal_conv import CausalConv1d
from stylestream.vocoder.convnext import ConvNeXtBlock
from stylestream.vocoder.backbone import VocosBackbone
from stylestream.vocoder.istft_head import ISTFTHead

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

B = 2
C_IN = 16
C_OUT = 32
T = 50
N_MELS = 100
HIDDEN = 64
INTERMEDIATE = 128
LAYERS = 2


# ======================================================================
# CausalConv1d Tests
# ======================================================================


class TestCausalConv1dOutputShape:
    """CausalConv1d should preserve time dimension for stride=1."""

    def test_output_shape_basic(self) -> None:
        """Output should be (B, C_out, T) when stride=1."""
        torch.manual_seed(42)
        conv = CausalConv1d(C_IN, C_OUT, kernel_size=3)
        x = torch.randn(B, C_IN, T)
        y = conv(x)
        assert y.shape == (B, C_OUT, T), f"Expected ({B}, {C_OUT}, {T}), got {y.shape}"

    @pytest.mark.parametrize("kernel_size", [1, 3, 7, 15])
    def test_kernel_size_variants(self, kernel_size: int) -> None:
        """Output time dimension should equal input for various kernel sizes."""
        torch.manual_seed(42)
        conv = CausalConv1d(C_IN, C_OUT, kernel_size=kernel_size)
        x = torch.randn(B, C_IN, T)
        y = conv(x)
        assert y.shape == (B, C_OUT, T), (
            f"kernel_size={kernel_size}: expected T={T}, got T={y.shape[-1]}"
        )

    def test_dilation(self) -> None:
        """Dilated causal conv should preserve time dimension and causality."""
        torch.manual_seed(42)
        conv = CausalConv1d(C_IN, C_OUT, kernel_size=3, dilation=2)
        x = torch.randn(B, C_IN, T)
        y = conv(x)
        assert y.shape == (B, C_OUT, T), (
            f"Dilation=2: expected T={T}, got T={y.shape[-1]}"
        )

    def test_groups_depthwise(self) -> None:
        """Depthwise (groups=channels) causal conv should produce correct shape."""
        torch.manual_seed(42)
        dim = 16
        conv = CausalConv1d(dim, dim, kernel_size=7, groups=dim)
        x = torch.randn(B, dim, T)
        y = conv(x)
        assert y.shape == (B, dim, T)


class TestCausalConv1dCausality:
    """CausalConv1d: future input frames must not affect past output frames."""

    def test_causality_kernel3(self) -> None:
        """Changing future input should not change past output."""
        torch.manual_seed(42)
        conv = CausalConv1d(C_IN, C_OUT, kernel_size=3)
        conv.eval()

        x1 = torch.randn(1, C_IN, T)
        x2 = x1.clone()

        # Modify the last 10 frames of x2
        t_split = T - 10
        x2[:, :, t_split:] = torch.randn(1, C_IN, 10)

        with torch.no_grad():
            y1 = conv(x1)
            y2 = conv(x2)

        # Output up to t_split should be identical
        torch.testing.assert_close(
            y1[:, :, :t_split],
            y2[:, :, :t_split],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_causality_with_dilation(self) -> None:
        """Causality should hold with dilation=2."""
        torch.manual_seed(42)
        conv = CausalConv1d(C_IN, C_OUT, kernel_size=3, dilation=2)
        conv.eval()

        x1 = torch.randn(1, C_IN, T)
        x2 = x1.clone()

        t_split = T - 10
        x2[:, :, t_split:] = torch.randn(1, C_IN, 10)

        with torch.no_grad():
            y1 = conv(x1)
            y2 = conv(x2)

        torch.testing.assert_close(
            y1[:, :, :t_split],
            y2[:, :, :t_split],
            atol=1e-6,
            rtol=1e-6,
        )


class TestCausalConv1dGradient:
    """CausalConv1d should support gradient flow."""

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the causal convolution."""
        torch.manual_seed(42)
        conv = CausalConv1d(C_IN, C_OUT, kernel_size=3)
        x = torch.randn(B, C_IN, T, requires_grad=True)
        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input should receive gradients"
        assert x.grad.abs().sum() > 0, "Input gradients should be non-zero"

        for name, param in conv.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"


# ======================================================================
# ConvNeXtBlock Tests
# ======================================================================


class TestConvNeXtBlockShape:
    """ConvNeXtBlock should preserve input shape."""

    def test_output_shape(self) -> None:
        """Output shape should match input: (B, dim, T)."""
        torch.manual_seed(42)
        block = ConvNeXtBlock(dim=HIDDEN, intermediate_dim=INTERMEDIATE, kernel_size=7)
        x = torch.randn(B, HIDDEN, T)
        y = block(x)
        assert y.shape == (B, HIDDEN, T), f"Expected ({B}, {HIDDEN}, {T}), got {y.shape}"

    def test_non_causal_shape(self) -> None:
        """Non-causal ConvNeXtBlock should also preserve shape."""
        torch.manual_seed(42)
        block = ConvNeXtBlock(dim=HIDDEN, intermediate_dim=INTERMEDIATE, causal=False)
        x = torch.randn(B, HIDDEN, T)
        y = block(x)
        assert y.shape == (B, HIDDEN, T)


class TestConvNeXtBlockResidual:
    """ConvNeXtBlock should have a functional residual connection."""

    def test_residual_connection(self) -> None:
        """With zero-weight final projection, output should approximate input."""
        torch.manual_seed(42)
        block = ConvNeXtBlock(dim=HIDDEN, intermediate_dim=INTERMEDIATE)

        # Zero out the down-projection so the branch output is zero
        nn.init.zeros_(block.pwconv2.weight)
        nn.init.zeros_(block.pwconv2.bias)

        x = torch.randn(B, HIDDEN, T)
        with torch.no_grad():
            y = block(x)

        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


class TestConvNeXtBlockCausality:
    """ConvNeXtBlock in causal mode must not leak future information."""

    def test_causal_no_future_leak(self) -> None:
        """Future frames should not affect past output in causal mode."""
        torch.manual_seed(42)
        block = ConvNeXtBlock(dim=HIDDEN, intermediate_dim=INTERMEDIATE, causal=True)
        block.eval()

        x1 = torch.randn(1, HIDDEN, T)
        x2 = x1.clone()

        t_split = T - 10
        x2[:, :, t_split:] = torch.randn(1, HIDDEN, T - t_split)

        with torch.no_grad():
            y1 = block(x1)
            y2 = block(x2)

        torch.testing.assert_close(
            y1[:, :, :t_split],
            y2[:, :, :t_split],
            atol=1e-6,
            rtol=1e-6,
        )


class TestConvNeXtBlockGradient:
    """ConvNeXtBlock should support gradient flow."""

    def test_gradient_flow(self) -> None:
        """All parameters should receive gradients."""
        torch.manual_seed(42)
        block = ConvNeXtBlock(dim=HIDDEN, intermediate_dim=INTERMEDIATE)
        x = torch.randn(B, HIDDEN, T)
        y = block(x)
        loss = y.sum()
        loss.backward()

        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"


# ======================================================================
# VocosBackbone Tests
# ======================================================================


class TestVocosBackboneShape:
    """VocosBackbone should map mel to hidden features."""

    def test_output_shape(self) -> None:
        """Output should be (B, hidden_size, T) from (B, n_mels, T)."""
        torch.manual_seed(42)
        backbone = VocosBackbone(
            n_mels=N_MELS, hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE, num_layers=LAYERS,
        )
        mel = torch.randn(B, N_MELS, T)
        out = backbone(mel)
        assert out.shape == (B, HIDDEN, T), (
            f"Expected ({B}, {HIDDEN}, {T}), got {out.shape}"
        )

    def test_default_config_construction(self) -> None:
        """Default config (512/1536/8 layers) should construct without errors."""
        backbone = VocosBackbone(
            n_mels=100, hidden_size=512,
            intermediate_size=1536, num_layers=8,
        )
        assert isinstance(backbone, nn.Module)
        assert len(backbone.layers) == 8

    def test_small_config(self) -> None:
        """Small config for faster testing should work correctly."""
        torch.manual_seed(42)
        backbone = VocosBackbone(
            n_mels=N_MELS, hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE, num_layers=LAYERS,
        )
        mel = torch.randn(B, N_MELS, 10)
        out = backbone(mel)
        assert out.shape == (B, HIDDEN, 10)

    @pytest.mark.parametrize("seq_len", [10, 50, 100])
    def test_variable_sequence_length(self, seq_len: int) -> None:
        """Backbone should handle different sequence lengths."""
        torch.manual_seed(42)
        backbone = VocosBackbone(
            n_mels=N_MELS, hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE, num_layers=LAYERS,
        )
        mel = torch.randn(B, N_MELS, seq_len)
        out = backbone(mel)
        assert out.shape == (B, HIDDEN, seq_len), (
            f"seq_len={seq_len}: expected T={seq_len}, got T={out.shape[-1]}"
        )


class TestVocosBackboneCausality:
    """VocosBackbone in causal mode must not leak future information."""

    def test_causal_backbone(self) -> None:
        """Future mel frames should not affect past hidden features."""
        torch.manual_seed(42)
        backbone = VocosBackbone(
            n_mels=N_MELS, hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE, num_layers=LAYERS, causal=True,
        )
        backbone.eval()

        mel1 = torch.randn(1, N_MELS, T)
        mel2 = mel1.clone()

        t_split = T - 10
        mel2[:, :, t_split:] = torch.randn(1, N_MELS, T - t_split)

        with torch.no_grad():
            out1 = backbone(mel1)
            out2 = backbone(mel2)

        torch.testing.assert_close(
            out1[:, :, :t_split],
            out2[:, :, :t_split],
            atol=1e-5,
            rtol=1e-5,
        )


# ======================================================================
# ISTFTHead Tests
# ======================================================================


class TestISTFTHeadShape:
    """ISTFTHead should produce waveform from backbone features."""

    def test_output_shape(self) -> None:
        """Output waveform length should be approximately T * hop_length."""
        torch.manual_seed(42)
        head = ISTFTHead(hidden_size=HIDDEN, n_fft=1024, hop_length=320)
        features = torch.randn(B, HIDDEN, T)
        waveform = head(features)

        assert waveform.dim() == 2, (
            f"Waveform should be 2D (B, T_samples), got {waveform.dim()}D"
        )
        assert waveform.shape[0] == B

        # The output length should be within n_fft tolerance of T * hop_length
        expected = T * 320
        actual = waveform.shape[1]
        assert abs(actual - expected) <= 1024, (
            f"Output length {actual} too far from expected {expected}"
        )

    def test_output_is_1d_per_sample(self) -> None:
        """Output shape should be (B, T_samples) with no extra dimensions."""
        torch.manual_seed(42)
        head = ISTFTHead(hidden_size=HIDDEN)
        features = torch.randn(B, HIDDEN, 20)
        waveform = head(features)
        assert waveform.dim() == 2, f"Expected 2D output, got {waveform.dim()}D"

    @pytest.mark.parametrize("seq_len", [10, 50, 100])
    def test_different_sequence_lengths(self, seq_len: int) -> None:
        """ISTFTHead should handle different input sequence lengths."""
        torch.manual_seed(42)
        head = ISTFTHead(hidden_size=HIDDEN)
        features = torch.randn(B, HIDDEN, seq_len)
        waveform = head(features)

        assert waveform.dim() == 2
        assert waveform.shape[0] == B
        # Just verify it produces a reasonable length
        assert waveform.shape[1] > 0


class TestISTFTHeadGradient:
    """ISTFTHead should support gradient flow."""

    def test_gradient_flow(self) -> None:
        """Gradients should flow from waveform back to input features."""
        torch.manual_seed(42)
        head = ISTFTHead(hidden_size=HIDDEN)
        features = torch.randn(B, HIDDEN, T, requires_grad=True)
        waveform = head(features)
        loss = waveform.sum()
        loss.backward()

        assert features.grad is not None, "Features should receive gradients"
        assert features.grad.abs().sum() > 0, "Feature gradients should be non-zero"
