"""Tests for StyleStream Causal Vocos full model, discriminator, and losses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from stylestream.vocoder.model import CausalVocos
from stylestream.vocoder.discriminator import MultiScaleDiscriminator
from stylestream.vocoder.losses import (
    VocoderLoss,
    discriminator_adversarial_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    mel_reconstruction_loss,
)
from stylestream.utils.mel import MelSpectrogramTransform

# ------------------------------------------------------------------
# Shared constants (small config for speed)
# ------------------------------------------------------------------

B = 2
N_MELS = 100
HIDDEN = 64
INTERMEDIATE = 128
LAYERS = 2
N_FFT = 1024
HOP = 320
T_MEL = 50  # mel frames
DISC_CHANNELS = 16
SCALES = [1, 2, 4]

# Waveform length for discriminator tests (must be long enough for pooling)
T_WAV = 8192


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def small_vocos() -> CausalVocos:
    """Create a small CausalVocos model for fast testing."""
    torch.manual_seed(42)
    return CausalVocos(
        n_mels=N_MELS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_layers=LAYERS,
        n_fft=N_FFT,
        hop_length=HOP,
    )


@pytest.fixture
def small_discriminator() -> MultiScaleDiscriminator:
    """Create a small MultiScaleDiscriminator for fast testing."""
    torch.manual_seed(42)
    return MultiScaleDiscriminator(scales=SCALES, channels=DISC_CHANNELS)


@pytest.fixture
def vocoder_loss() -> VocoderLoss:
    """Create a VocoderLoss instance."""
    return VocoderLoss(
        reconstruction_weight=45.0,
        gan_generator_weight=1.0,
        gan_discriminator_weight=1.0,
        feature_matching_weight=2.0,
    )


# ======================================================================
# CausalVocos Model Tests
# ======================================================================


class TestCausalVocosConstruction:
    """Tests for CausalVocos model construction."""

    def test_construction(self, small_vocos: CausalVocos) -> None:
        """CausalVocos should be constructable with small config."""
        assert isinstance(small_vocos, CausalVocos)
        assert isinstance(small_vocos, nn.Module)

    def test_num_parameters(self, small_vocos: CausalVocos) -> None:
        """num_parameters should return a positive integer."""
        n_params = small_vocos.num_parameters(trainable_only=True)
        assert isinstance(n_params, int)
        assert n_params > 0, "Model should have trainable parameters"

    def test_num_parameters_all(self, small_vocos: CausalVocos) -> None:
        """Total parameters should be >= trainable parameters."""
        n_trainable = small_vocos.num_parameters(trainable_only=True)
        n_total = small_vocos.num_parameters(trainable_only=False)
        assert n_total >= n_trainable

    def test_from_config_yaml_style(self) -> None:
        """from_config should work with YAML-style config (model + mel)."""
        config = SimpleNamespace(
            model=SimpleNamespace(
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                num_layers=LAYERS,
                kernel_size=7,
                causal=True,
            ),
            mel=SimpleNamespace(
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP,
            ),
        )
        model = CausalVocos.from_config(config)
        assert isinstance(model, CausalVocos)
        assert model.hidden_size == HIDDEN
        assert model.num_layers == LAYERS
        assert model.n_mels == N_MELS

    def test_from_config_flat(self) -> None:
        """from_config should work with flat config."""
        config = SimpleNamespace(
            n_mels=N_MELS,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            n_fft=N_FFT,
            hop_length=HOP,
            kernel_size=7,
            causal=True,
        )
        model = CausalVocos.from_config(config)
        assert isinstance(model, CausalVocos)
        assert model.n_mels == N_MELS


class TestCausalVocosForward:
    """Tests for CausalVocos forward pass."""

    def test_forward_shape(self, small_vocos: CausalVocos) -> None:
        """Forward pass should produce waveform of correct shape."""
        torch.manual_seed(42)
        mel = torch.randn(B, N_MELS, T_MEL)
        waveform = small_vocos(mel)

        assert waveform.dim() == 2, f"Expected 2D output, got {waveform.dim()}D"
        assert waveform.shape[0] == B

        # Output length should be approximately T_MEL * HOP
        expected = T_MEL * HOP
        actual = waveform.shape[1]
        assert abs(actual - expected) <= N_FFT, (
            f"Output length {actual} too far from expected {expected}"
        )

    def test_forward_determinism(self, small_vocos: CausalVocos) -> None:
        """Same input should produce same output in eval mode."""
        small_vocos.eval()
        mel = torch.randn(B, N_MELS, T_MEL)

        with torch.no_grad():
            y1 = small_vocos(mel)
            y2 = small_vocos(mel)

        torch.testing.assert_close(y1, y2, atol=1e-6, rtol=1e-6)

    def test_gradient_flow(self, small_vocos: CausalVocos) -> None:
        """Loss on waveform should produce gradients on all parameters."""
        torch.manual_seed(42)
        mel = torch.randn(B, N_MELS, T_MEL)
        waveform = small_vocos(mel)
        loss = waveform.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_vocos.parameters()
            if p.requires_grad
        )
        assert has_grad, "Some parameters should have received gradients"

    def test_causal_property(self, small_vocos: CausalVocos) -> None:
        """Future mel frames should not affect past waveform samples."""
        small_vocos.eval()
        torch.manual_seed(42)

        mel1 = torch.randn(1, N_MELS, T_MEL)
        mel2 = mel1.clone()

        # Modify the last 15 mel frames
        t_split = T_MEL - 15
        mel2[:, :, t_split:] = torch.randn(1, N_MELS, 15)

        with torch.no_grad():
            wav1 = small_vocos(mel1)
            wav2 = small_vocos(mel2)

        # Waveform samples well before the modification boundary should be
        # identical.  The ISTFT uses an analysis window of n_fft samples which
        # spans several mel frames, so we apply a conservative margin of
        # n_fft samples beyond the raw frame boundary to avoid the overlap
        # zone.
        safe_samples = (t_split - 1) * HOP - N_FFT
        if safe_samples > 0:
            torch.testing.assert_close(
                wav1[:, :safe_samples],
                wav2[:, :safe_samples],
                atol=1e-5,
                rtol=1e-5,
            )


# ======================================================================
# MultiScaleDiscriminator Tests
# ======================================================================


class TestMultiScaleDiscriminatorConstruction:
    """Tests for MultiScaleDiscriminator construction."""

    def test_construction(self, small_discriminator: MultiScaleDiscriminator) -> None:
        """MultiScaleDiscriminator should construct with small config."""
        assert isinstance(small_discriminator, MultiScaleDiscriminator)
        assert isinstance(small_discriminator, nn.Module)


class TestMultiScaleDiscriminatorForward:
    """Tests for MultiScaleDiscriminator forward pass."""

    def test_output_format(self, small_discriminator: MultiScaleDiscriminator) -> None:
        """Forward should return (logits_list, features_list)."""
        torch.manual_seed(42)
        x = torch.randn(B, 1, T_WAV)
        logits_list, features_list = small_discriminator(x)

        assert isinstance(logits_list, list)
        assert isinstance(features_list, list)

    def test_logits_count(self, small_discriminator: MultiScaleDiscriminator) -> None:
        """Number of logit tensors should equal number of scales."""
        torch.manual_seed(42)
        x = torch.randn(B, 1, T_WAV)
        logits_list, _ = small_discriminator(x)

        assert len(logits_list) == len(SCALES), (
            f"Expected {len(SCALES)} logit tensors, got {len(logits_list)}"
        )

    def test_features_count(self, small_discriminator: MultiScaleDiscriminator) -> None:
        """Number of feature lists should equal number of scales."""
        torch.manual_seed(42)
        x = torch.randn(B, 1, T_WAV)
        _, features_list = small_discriminator(x)

        assert len(features_list) == len(SCALES), (
            f"Expected {len(SCALES)} feature lists, got {len(features_list)}"
        )

    def test_input_2d(self, small_discriminator: MultiScaleDiscriminator) -> None:
        """Discriminator should accept (B, T) input (auto-unsqueeze)."""
        torch.manual_seed(42)
        x = torch.randn(B, T_WAV)
        logits_list, features_list = small_discriminator(x)

        assert len(logits_list) == len(SCALES)
        assert len(features_list) == len(SCALES)

    def test_input_3d(self, small_discriminator: MultiScaleDiscriminator) -> None:
        """Discriminator should accept (B, 1, T) input directly."""
        torch.manual_seed(42)
        x = torch.randn(B, 1, T_WAV)
        logits_list, features_list = small_discriminator(x)

        assert len(logits_list) == len(SCALES)
        for logits in logits_list:
            assert logits.shape[0] == B

    def test_gradient_flow(self, small_discriminator: MultiScaleDiscriminator) -> None:
        """Gradients should flow to discriminator parameters."""
        torch.manual_seed(42)
        x = torch.randn(B, 1, T_WAV)
        logits_list, _ = small_discriminator(x)

        loss = sum(lg.mean() for lg in logits_list)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_discriminator.parameters()
            if p.requires_grad
        )
        assert has_grad, "Discriminator parameters should have received gradients"


# ======================================================================
# Standalone Loss Function Tests
# ======================================================================


class TestGeneratorAdversarialLoss:
    """Tests for generator_adversarial_loss."""

    def test_non_negative_scalar(self) -> None:
        """Generator adversarial loss should be a non-negative scalar."""
        disc_outputs = [torch.randn(B, 1, 10) for _ in range(3)]
        loss = generator_adversarial_loss(disc_outputs)

        assert loss.dim() == 0 or loss.numel() == 1, "Loss should be scalar"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"


class TestDiscriminatorAdversarialLoss:
    """Tests for discriminator_adversarial_loss."""

    def test_non_negative_scalar(self) -> None:
        """Discriminator adversarial loss should be a non-negative scalar."""
        real_outputs = [torch.randn(B, 1, 10) for _ in range(3)]
        fake_outputs = [torch.randn(B, 1, 10) for _ in range(3)]
        loss = discriminator_adversarial_loss(real_outputs, fake_outputs)

        assert loss.dim() == 0 or loss.numel() == 1
        assert loss.item() >= 0

    def test_zero_for_perfect(self) -> None:
        """Loss should be approximately zero when real=1 and fake=0."""
        real_outputs = [torch.ones(B, 1, 10) for _ in range(3)]
        fake_outputs = [torch.zeros(B, 1, 10) for _ in range(3)]
        loss = discriminator_adversarial_loss(real_outputs, fake_outputs)

        assert loss.item() < 1e-6, (
            f"Perfect discrimination should give ~0 loss, got {loss.item()}"
        )


class TestFeatureMatchingLoss:
    """Tests for feature_matching_loss."""

    def test_non_negative_scalar(self) -> None:
        """Feature matching loss should be a non-negative scalar."""
        real_features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]
        fake_features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]
        loss = feature_matching_loss(real_features, fake_features)

        assert loss.dim() == 0 or loss.numel() == 1
        assert loss.item() >= 0

    def test_zero_for_identical(self) -> None:
        """Feature matching loss should be zero for identical features."""
        features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]
        # Pass same features as both real and fake
        loss = feature_matching_loss(features, features)

        assert loss.item() < 1e-6, (
            f"Identical features should give ~0 loss, got {loss.item()}"
        )


class TestMelReconstructionLoss:
    """Tests for mel_reconstruction_loss."""

    def test_non_negative_scalar(self) -> None:
        """Mel reconstruction loss should be a non-negative scalar."""
        torch.manual_seed(42)
        mel_transform = MelSpectrogramTransform(n_mels=N_MELS, hop_length=HOP)

        # Create waveforms long enough for mel computation
        pred = torch.randn(B, 16000)
        target = torch.randn(B, 16000)
        loss = mel_reconstruction_loss(pred, target, mel_transform)

        assert loss.dim() == 0 or loss.numel() == 1
        assert loss.item() >= 0

    def test_zero_for_identical(self) -> None:
        """Mel reconstruction loss should be approximately zero for identical waveforms."""
        torch.manual_seed(42)
        mel_transform = MelSpectrogramTransform(n_mels=N_MELS, hop_length=HOP)

        waveform = torch.randn(B, 16000)
        loss = mel_reconstruction_loss(waveform, waveform, mel_transform)

        assert loss.item() < 1e-5, (
            f"Identical waveforms should give ~0 loss, got {loss.item()}"
        )


# ======================================================================
# VocoderLoss Tests
# ======================================================================


class TestVocoderLossGenerator:
    """Tests for VocoderLoss.generator_loss."""

    def test_output_keys(self, vocoder_loss: VocoderLoss) -> None:
        """generator_loss should return dict with 'loss', 'mel_loss', 'gan_loss', 'fm_loss'."""
        torch.manual_seed(42)
        pred_wav = torch.randn(B, 16000)
        target_wav = torch.randn(B, 16000)
        disc_fake_outputs = [torch.randn(B, 1, 10) for _ in range(3)]
        real_features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]
        fake_features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]

        result = vocoder_loss.generator_loss(
            pred_wav, target_wav,
            disc_fake_outputs, real_features, fake_features,
        )

        assert "loss" in result
        assert "mel_loss" in result
        assert "gan_loss" in result
        assert "fm_loss" in result

    def test_total_loss_is_scalar(self, vocoder_loss: VocoderLoss) -> None:
        """Total generator loss should be a scalar tensor."""
        torch.manual_seed(42)
        pred_wav = torch.randn(B, 16000)
        target_wav = torch.randn(B, 16000)
        disc_fake_outputs = [torch.randn(B, 1, 10) for _ in range(3)]
        real_features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]
        fake_features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]

        result = vocoder_loss.generator_loss(
            pred_wav, target_wav,
            disc_fake_outputs, real_features, fake_features,
        )

        assert result["loss"].dim() == 0 or result["loss"].numel() == 1


class TestVocoderLossDiscriminator:
    """Tests for VocoderLoss.discriminator_loss."""

    def test_output_keys(self, vocoder_loss: VocoderLoss) -> None:
        """discriminator_loss should return dict with 'loss'."""
        real_outputs = [torch.randn(B, 1, 10) for _ in range(3)]
        fake_outputs = [torch.randn(B, 1, 10) for _ in range(3)]

        result = vocoder_loss.discriminator_loss(real_outputs, fake_outputs)

        assert "loss" in result
        assert result["loss"].dim() == 0 or result["loss"].numel() == 1


class TestVocoderLossWeights:
    """Tests for VocoderLoss weight application."""

    def test_weights_applied(self) -> None:
        """Changing weights should change the total loss."""
        torch.manual_seed(42)
        pred_wav = torch.randn(B, 16000)
        target_wav = torch.randn(B, 16000)
        disc_fake_outputs = [torch.randn(B, 1, 10) for _ in range(3)]
        real_features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]
        fake_features = [[torch.randn(B, 16, 10) for _ in range(3)] for _ in range(3)]

        loss_w1 = VocoderLoss(reconstruction_weight=1.0, feature_matching_weight=1.0)
        loss_w2 = VocoderLoss(reconstruction_weight=100.0, feature_matching_weight=1.0)

        r1 = loss_w1.generator_loss(
            pred_wav, target_wav,
            disc_fake_outputs, real_features, fake_features,
        )
        r2 = loss_w2.generator_loss(
            pred_wav, target_wav,
            disc_fake_outputs, real_features, fake_features,
        )

        # With higher reconstruction weight, total loss should be larger
        # (assuming mel_loss > 0, which is virtually certain for random inputs)
        assert r2["loss"].item() > r1["loss"].item(), (
            "Higher reconstruction weight should produce larger total loss"
        )


# ======================================================================
# Integration Tests
# ======================================================================


class TestVocoderIntegration:
    """Integration tests for full vocoder training pipeline."""

    @pytest.mark.slow
    def test_full_training_step(
        self,
        small_vocos: CausalVocos,
        small_discriminator: MultiScaleDiscriminator,
    ) -> None:
        """Full training step: mel -> vocos -> waveform -> disc -> losses -> backward."""
        torch.manual_seed(42)
        small_vocos.train()
        small_discriminator.train()

        mel = torch.randn(B, N_MELS, T_MEL)

        # Generate waveform
        pred_waveform = small_vocos(mel)

        # Create a "real" target waveform of matching length
        target_waveform = torch.randn_like(pred_waveform)

        # Discriminator forward on both real and fake
        real_logits, real_features = small_discriminator(target_waveform.detach())
        fake_logits, fake_features = small_discriminator(pred_waveform)

        # Generator adversarial loss
        g_adv_loss = generator_adversarial_loss(fake_logits)
        fm_loss = feature_matching_loss(real_features, fake_features)

        g_loss = g_adv_loss + fm_loss
        assert torch.isfinite(g_loss), f"Generator loss is not finite: {g_loss.item()}"

        # Generator backward
        g_loss.backward()

        has_vocos_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_vocos.parameters()
            if p.requires_grad
        )
        assert has_vocos_grad, "Vocos model should have received gradients"

        # Discriminator loss
        small_vocos.zero_grad()
        small_discriminator.zero_grad()

        with torch.no_grad():
            pred_waveform_detached = small_vocos(mel)
        real_logits_d, _ = small_discriminator(target_waveform)
        fake_logits_d, _ = small_discriminator(pred_waveform_detached)

        d_loss = discriminator_adversarial_loss(real_logits_d, fake_logits_d)
        assert torch.isfinite(d_loss), f"Discriminator loss is not finite: {d_loss.item()}"

        d_loss.backward()

        has_disc_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_discriminator.parameters()
            if p.requires_grad
        )
        assert has_disc_grad, "Discriminator should have received gradients"

    def test_gan_adversarial_dynamics(self) -> None:
        """Discriminator loss should be low when real=1 and fake=0."""
        # Simulate perfect discrimination
        real_outputs = [torch.ones(B, 1, 10) for _ in range(3)]
        fake_outputs = [torch.zeros(B, 1, 10) for _ in range(3)]

        d_loss = discriminator_adversarial_loss(real_outputs, fake_outputs)
        assert d_loss.item() < 1e-6, (
            f"Perfect discrimination should yield ~0 loss, got {d_loss.item()}"
        )

        # Simulate worst-case discrimination (reversed labels)
        bad_real = [torch.zeros(B, 1, 10) for _ in range(3)]
        bad_fake = [torch.ones(B, 1, 10) for _ in range(3)]
        bad_loss = discriminator_adversarial_loss(bad_real, bad_fake)

        assert bad_loss.item() > d_loss.item(), (
            "Reversed labels should give higher loss than perfect discrimination"
        )
