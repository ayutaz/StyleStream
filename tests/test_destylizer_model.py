"""Comprehensive integration tests for the full Destylizer model.

Tests cover model construction, forward pass, content feature extraction,
gradient flow, training integration, and the ContentFeatureExtractor wrapper.

All tests use small dimensions for speed:
    hidden_size=64, ffn_size=256, num_heads=4, conformer layers=2,
    kernel_size=7, FSQ levels=[5,3,3], ASR layers=1, vocab_size=10.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from stylestream.destylizer.model import Destylizer
from stylestream.destylizer.feature_extractor import ContentFeatureExtractor
from stylestream.destylizer.conformer import ConformerEncoder
from stylestream.destylizer.fsq import FSQ
from stylestream.destylizer.asr_head import ASRHead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 2       # batch size
T = 20      # sequence length (frames)
S = 10      # target sequence length (characters)
HIDDEN = 64
VOCAB = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    """Minimal config for fast testing."""
    return SimpleNamespace(
        conformer=SimpleNamespace(
            num_layers=2,
            hidden_size=HIDDEN,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            dropout=0.0,
        ),
        fsq=SimpleNamespace(levels=[5, 3, 3], down_dim=3),
        asr_decoder=SimpleNamespace(
            num_layers=1,
            hidden_size=HIDDEN,
            ffn_size=256,
            num_heads=4,
            vocab_size=VOCAB,
            dropout=0.0,
            loss_type="ctc",
            label_smoothing=0.0,
        ),
    )


@pytest.fixture
def small_model(small_config):
    """Build a Destylizer with small config for fast testing."""
    torch.manual_seed(42)
    model = Destylizer(
        hidden_size=HIDDEN,
        num_layers=small_config.conformer.num_layers,
        ffn_size=small_config.conformer.ffn_size,
        num_heads=small_config.conformer.num_heads,
        kernel_size=small_config.conformer.kernel_size,
        dropout=small_config.conformer.dropout,
        fsq_levels=list(small_config.fsq.levels),
        fsq_hidden_size=HIDDEN,
        asr_loss_type=small_config.asr_decoder.loss_type,
        asr_num_layers=small_config.asr_decoder.num_layers,
        asr_ffn_size=small_config.asr_decoder.ffn_size,
        asr_num_heads=small_config.asr_decoder.num_heads,
        vocab_size=small_config.asr_decoder.vocab_size,
        asr_dropout=small_config.asr_decoder.dropout,
        label_smoothing=small_config.asr_decoder.label_smoothing,
    )
    model.eval()
    return model


# =========================================================================
# Model Construction
# =========================================================================


class TestDestylizerConstruction:
    """Tests for Destylizer model construction."""

    def test_build_with_defaults(self):
        """Destylizer() builds with small kwargs without errors."""
        model = Destylizer(
            hidden_size=HIDDEN,
            num_layers=2,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            vocab_size=VOCAB,
            fsq_hidden_size=HIDDEN,
            asr_num_layers=1,
            asr_ffn_size=256,
            asr_num_heads=4,
        )
        assert isinstance(model, nn.Module)

    def test_build_from_config(self, small_config):
        """from_config creates a valid model from a config namespace."""
        model = Destylizer.from_config(small_config)
        assert isinstance(model, Destylizer)
        assert isinstance(model, nn.Module)

    def test_has_conformer(self, small_model):
        """Model has a conformer (ConformerEncoder) attribute."""
        assert hasattr(small_model, "conformer")
        assert isinstance(small_model.conformer, ConformerEncoder)

    def test_has_fsq(self, small_model):
        """Model has an fsq (FSQ) attribute."""
        assert hasattr(small_model, "fsq")
        assert isinstance(small_model.fsq, FSQ)

    def test_has_asr_head(self, small_model):
        """Model has an asr_head (ASRHead) attribute."""
        assert hasattr(small_model, "asr_head")
        assert isinstance(small_model.asr_head, ASRHead)

    def test_has_input_norm(self, small_model):
        """Model has an input LayerNorm."""
        assert hasattr(small_model, "input_norm")
        assert isinstance(small_model.input_norm, nn.LayerNorm)


# =========================================================================
# Forward Pass
# =========================================================================


class TestDestylizerForward:
    """Tests for the full forward pass."""

    def test_forward_shape_bt768(self, small_model):
        """(B, T, hidden) input produces the expected output dict."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        out = small_model(x)

        assert isinstance(out, dict)
        assert "content_features" in out
        assert "logits" in out
        assert "fsq_info" in out

    def test_forward_shape_b768t(self, small_model):
        """(B, 768, T) input is auto-transposed and produces valid output.

        We use the real HuBERT dim (768) so the auto-transpose heuristic
        triggers. T must differ from 768 for detection.
        """
        # Build a model with hidden_size=768 but minimal layers for this test.
        torch.manual_seed(42)
        model = Destylizer(
            hidden_size=768,
            num_layers=1,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            vocab_size=VOCAB,
            fsq_hidden_size=768,
            asr_num_layers=1,
            asr_ffn_size=256,
            asr_num_heads=4,
        )
        model.eval()

        # Channels-first layout: (B, 768, T) where T != 768
        x = torch.randn(B, 768, T)
        out = model(x)

        assert out["content_features"].shape == (B, T, 768)
        assert out["logits"].shape[0] == B
        assert out["logits"].shape[1] == T

    def test_content_features_shape(self, small_model):
        """content_features has shape (B, T, hidden_size)."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        out = small_model(x)
        assert out["content_features"].shape == (B, T, HIDDEN)

    def test_logits_shape_ctc(self, small_model):
        """logits shape is (B, T, vocab_size) in CTC mode."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        out = small_model(x)
        assert out["logits"].shape == (B, T, VOCAB)

    def test_fsq_info_present(self, small_model):
        """fsq_info dict has the expected diagnostic keys."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        out = small_model(x)

        info = out["fsq_info"]
        assert "indices" in info
        assert "codebook_usage" in info
        assert "perplexity" in info
        assert "pre_quant" in info

        # indices should be integer codes with shape (B, T)
        assert info["indices"].shape == (B, T)
        # codebook_usage is a float in [0, 1]
        assert 0.0 <= info["codebook_usage"] <= 1.0
        # perplexity is a positive float
        assert info["perplexity"] > 0.0

    def test_with_padding_mask(self, small_model):
        """Padding mask does not crash and produces valid output."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        # Mask the last 5 frames of each sequence
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -5:] = True

        out = small_model(x, padding_mask=mask)

        assert out["content_features"].shape == (B, T, HIDDEN)
        assert out["logits"].shape == (B, T, VOCAB)
        # Output should be finite
        assert torch.isfinite(out["content_features"]).all()

    def test_with_target_ids(self):
        """Providing target_ids works for seq2seq mode."""
        torch.manual_seed(42)
        model = Destylizer(
            hidden_size=HIDDEN,
            num_layers=1,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            vocab_size=VOCAB,
            fsq_hidden_size=HIDDEN,
            asr_loss_type="seq2seq_ce",
            asr_num_layers=1,
            asr_ffn_size=256,
            asr_num_heads=4,
        )
        model.eval()

        x = torch.randn(B, T, HIDDEN)
        # target_ids: (B, S) with valid token IDs
        target_ids = torch.randint(0, VOCAB, (B, S))

        out = model(x, target_ids=target_ids)

        assert out["content_features"].shape == (B, T, HIDDEN)
        # seq2seq logits shape: (B, S-1, vocab_size)
        assert out["logits"].shape == (B, S - 1, VOCAB)

    def test_without_target_ids(self, small_model):
        """CTC mode works without target_ids."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        out = small_model(x)
        assert out["logits"].shape == (B, T, VOCAB)


# =========================================================================
# Content Feature Extraction
# =========================================================================


class TestContentFeatures:
    """Tests for extract_content_features (inference path)."""

    def test_extract_shape(self, small_model):
        """extract_content_features returns (B, T, hidden)."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        fc = small_model.extract_content_features(x)
        assert fc.shape == (B, T, HIDDEN)

    def test_extract_no_fsq(self, small_model):
        """Content features are pre-FSQ (continuous, not quantized).

        FSQ projects to a 3-dim space with levels [5,3,3], producing
        discrete values. The content features from extract_content_features
        should be 64-dim (hidden_size) continuous vectors, not the quantized
        output.
        """
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        fc = small_model.extract_content_features(x)

        # Content features should have hidden_size dims (not FSQ dims)
        assert fc.shape[-1] == HIDDEN

        # They should generally be continuous (non-integer) values
        # Rounding to nearest int should change the values
        is_integer = torch.allclose(fc, fc.round(), atol=1e-6)
        assert not is_integer, "Content features appear quantized (all integer values)"

    def test_extract_matches_forward(self, small_model):
        """fc from extract_content_features matches fc from full forward."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)

        small_model.eval()
        # Full forward returns content_features
        with torch.no_grad():
            out = small_model(x)
        fc_forward = out["content_features"]

        # extract_content_features returns the same thing
        fc_extract = small_model.extract_content_features(x)

        assert torch.allclose(fc_forward, fc_extract, atol=1e-6), (
            "extract_content_features should produce the same result as "
            "content_features from forward"
        )

    def test_extract_deterministic(self, small_model):
        """Same input produces same output in eval mode."""
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)

        small_model.eval()
        fc1 = small_model.extract_content_features(x)
        fc2 = small_model.extract_content_features(x)

        assert torch.allclose(fc1, fc2, atol=1e-7), (
            "extract_content_features should be deterministic in eval mode"
        )

    def test_extract_no_grad(self, small_model):
        """extract_content_features does not require gradient computation.

        The method is decorated with @torch.no_grad, so the output
        should not have grad_fn.
        """
        x = torch.randn(B, T, HIDDEN, requires_grad=True)
        fc = small_model.extract_content_features(x)
        assert fc.grad_fn is None, (
            "extract_content_features output should not track gradients"
        )


# =========================================================================
# Gradient Flow
# =========================================================================


class TestGradientFlow:
    """Tests verifying gradients flow correctly through the model."""

    def _make_trainable_model(self):
        """Build a small model in training mode."""
        torch.manual_seed(42)
        model = Destylizer(
            hidden_size=HIDDEN,
            num_layers=2,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            dropout=0.0,
            vocab_size=VOCAB,
            fsq_hidden_size=HIDDEN,
            asr_num_layers=1,
            asr_ffn_size=256,
            asr_num_heads=4,
            asr_dropout=0.0,
            label_smoothing=0.0,
        )
        model.train()
        return model

    def test_gradient_through_fsq_ste(self):
        """STE allows gradients to flow through FSQ back to Conformer."""
        model = self._make_trainable_model()
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)

        out = model(x)
        # Sum logits as a scalar loss proxy
        loss = out["logits"].sum()
        loss.backward()

        # Check that conformer parameters received gradients
        # (they would not if FSQ blocked the gradient flow)
        for name, param in model.conformer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Conformer param {name} has no gradient -- FSQ STE may be broken"
                )
                break  # one check is sufficient to confirm flow

    def test_conformer_params_have_grad(self):
        """After backward, Conformer parameters have non-None gradients."""
        model = self._make_trainable_model()
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)

        out = model(x)
        loss = out["logits"].sum()
        loss.backward()

        grads_found = 0
        for name, param in model.conformer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Conformer param '{name}' has no gradient after backward"
                )
                grads_found += 1

        assert grads_found > 0, "No trainable Conformer parameters found"

    def test_fsq_projection_has_grad(self):
        """FSQ down/up projections have gradients after backward."""
        model = self._make_trainable_model()
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)

        out = model(x)
        loss = out["logits"].sum()
        loss.backward()

        assert model.fsq.down_proj.weight.grad is not None, (
            "FSQ down_proj.weight has no gradient"
        )
        assert model.fsq.up_proj.weight.grad is not None, (
            "FSQ up_proj.weight has no gradient"
        )

    def test_asr_head_has_grad(self):
        """ASR head parameters have gradients after backward."""
        model = self._make_trainable_model()
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)

        out = model(x)
        loss = out["logits"].sum()
        loss.backward()

        grads_found = 0
        for name, param in model.asr_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"ASR head param '{name}' has no gradient after backward"
                )
                grads_found += 1

        assert grads_found > 0, "No trainable ASR head parameters found"


# =========================================================================
# Integration
# =========================================================================


class TestDestylizerIntegration:
    """End-to-end integration tests."""

    def test_training_step_decreases_loss(self):
        """5 steps of training reduce the CTC loss."""
        torch.manual_seed(42)
        model = Destylizer(
            hidden_size=HIDDEN,
            num_layers=2,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            dropout=0.0,
            vocab_size=VOCAB,
            fsq_hidden_size=HIDDEN,
            asr_num_layers=1,
            asr_ffn_size=256,
            asr_num_heads=4,
            asr_dropout=0.0,
            label_smoothing=0.0,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed input and targets for reproducibility
        x = torch.randn(B, T, HIDDEN)
        # CTC targets: random token IDs in [1, VOCAB-1] (avoid blank=0)
        target_lengths = torch.tensor([5, 4])
        targets = torch.randint(1, VOCAB, (B, max(target_lengths)))
        encoder_lengths = torch.full((B,), T, dtype=torch.long)

        losses = []
        for _step in range(5):
            optimizer.zero_grad()
            out = model(x)
            logits = out["logits"]
            loss = model.asr_head.compute_loss(
                logits, targets, encoder_lengths, target_lengths
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (last < first)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease over 5 steps: {losses}"
        )

    def test_batch_independence(self, small_model):
        """Each batch item is processed independently.

        Processing items individually vs. batched should produce the same
        content features (within floating point tolerance).
        """
        torch.manual_seed(42)
        small_model.eval()

        x = torch.randn(B, T, HIDDEN)

        # Batched forward
        with torch.no_grad():
            fc_batched = small_model(x)["content_features"]

        # Individual forwards
        individual = []
        for i in range(B):
            with torch.no_grad():
                fc_i = small_model(x[i : i + 1])["content_features"]
            individual.append(fc_i)
        fc_individual = torch.cat(individual, dim=0)

        assert torch.allclose(fc_batched, fc_individual, atol=1e-5), (
            "Batched and individual processing should give the same result"
        )

    def test_variable_sequence_length(self):
        """Different T values in the same batch work with padding masks."""
        torch.manual_seed(42)
        model = Destylizer(
            hidden_size=HIDDEN,
            num_layers=2,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            dropout=0.0,
            vocab_size=VOCAB,
            fsq_hidden_size=HIDDEN,
            asr_num_layers=1,
            asr_ffn_size=256,
            asr_num_heads=4,
            asr_dropout=0.0,
        )
        model.eval()

        # Two sequences: lengths 20 and 12, padded to 20
        T_max = 20
        lengths = [20, 12]

        x = torch.randn(B, T_max, HIDDEN)
        # Zero out padded positions
        x[1, lengths[1]:] = 0.0

        mask = torch.zeros(B, T_max, dtype=torch.bool)
        mask[1, lengths[1]:] = True

        out = model(x, padding_mask=mask)

        assert out["content_features"].shape == (B, T_max, HIDDEN)
        assert out["logits"].shape == (B, T_max, VOCAB)
        # Output should be finite
        assert torch.isfinite(out["content_features"]).all()
        assert torch.isfinite(out["logits"]).all()

    def test_content_features_continuous(self, small_model):
        """Content features are continuous, not discrete.

        The Conformer output (fc) should be smooth continuous values,
        not restricted to a small set of integer-like values.
        """
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)

        small_model.eval()
        fc = small_model.extract_content_features(x)

        # Count unique values -- continuous features should have many unique values
        unique_vals = torch.unique(fc)
        total_elements = fc.numel()

        # With B*T*HIDDEN = 2*20*64 = 2560 elements, continuous features
        # should have a large number of unique values (close to total_elements)
        assert len(unique_vals) > total_elements * 0.5, (
            f"Expected many unique values in content features, "
            f"got {len(unique_vals)} / {total_elements}"
        )


# =========================================================================
# ContentFeatureExtractor
# =========================================================================


class TestContentFeatureExtractor:
    """Tests for the ContentFeatureExtractor inference wrapper."""

    def _make_extractor(self):
        """Build a ContentFeatureExtractor with a small Destylizer."""
        torch.manual_seed(42)
        model = Destylizer(
            hidden_size=HIDDEN,
            num_layers=2,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            dropout=0.0,
            vocab_size=VOCAB,
            fsq_hidden_size=HIDDEN,
            asr_num_layers=1,
            asr_ffn_size=256,
            asr_num_heads=4,
            asr_dropout=0.0,
        )
        return ContentFeatureExtractor(
            destylizer=model,
            device="cpu",
            hubert_layer=18,
            max_audio_sec=30.0,
        )

    def test_extract_from_hubert(self):
        """extract_from_hubert_features works with pre-computed features."""
        extractor = self._make_extractor()
        torch.manual_seed(42)
        x = torch.randn(B, T, HIDDEN)
        fc = extractor.extract_from_hubert_features(x)

        assert fc.shape == (B, T, HIDDEN)
        assert torch.isfinite(fc).all()

    def test_feature_dim(self):
        """feature_dim property returns 768 (HuBERT dimension)."""
        extractor = self._make_extractor()
        assert extractor.feature_dim == 768

    def test_frame_rate(self):
        """frame_rate property returns 50 Hz."""
        extractor = self._make_extractor()
        assert extractor.frame_rate == 50
