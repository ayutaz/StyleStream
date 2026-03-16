"""Tests for streaming models, ring buffer, and inference pipeline.

Covers:
    - RingBuffer: FIFO feature accumulation for streaming
    - StreamingContext: target + source state management
    - StreamingDestylizer: causal Destylizer (mocked HuBERT)
    - StreamingDiTBlock / StreamingDiT / StreamingStylizer
    - StreamingInferencePipeline: end-to-end orchestration

All tests use small configs (hidden=64, layers=2, heads=4) and mock
WavLM/HuBERT to avoid downloading pretrained models.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

from stylestream.streaming.ring_buffer import RingBuffer, StreamingContext
from stylestream.streaming.destylizer import StreamingDestylizer
from stylestream.streaming.stylizer import (
    StreamingDiTBlock,
    StreamingDiT,
    StreamingStylizer,
)
from stylestream.streaming.pipeline import StreamingInferencePipeline
from stylestream.stylizer.style_encoder import StyleEncoder

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS  # 16
FFN = 256
LAYERS = 2
MEL_DIM = 10
CONTENT_DIM = 64
CHUNK_SIZE = 10
B = 1  # streaming is batch-size 1
FEATURE_DIM = 768  # default content feature dim

TDNN_CHANNELS = 32
NUM_WAVLM_LAYERS = 13
AUDIO_SAMPLES = 4800  # ~0.3s at 16kHz


# ------------------------------------------------------------------
# Mocks
# ------------------------------------------------------------------


class MockStreamingHuBERT(nn.Module):
    """Lightweight mock of StreamingHuBERT that skips model download."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 1024)  # dummy parameter

    def forward(self, waveform: torch.Tensor, attention_mask=None) -> torch.Tensor:
        B = waveform.shape[0]
        T = waveform.shape[-1] // 320  # simulate 50Hz
        return torch.randn(B, max(T, 1), 1024)


class MockWavLMOutput:
    """Mimics HuggingFace WavLMModel output with hidden_states."""

    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states
        self.last_hidden_state = hidden_states[-1]


class MockWavLM(nn.Module):
    """A lightweight mock of WavLMModel."""

    def __init__(
        self,
        hidden_size: int = HIDDEN,
        num_layers: int = NUM_WAVLM_LAYERS,
    ) -> None:
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


# ======================================================================
# RingBuffer Tests
# ======================================================================


class TestRingBuffer:
    """Tests for RingBuffer."""

    def test_empty_buffer(self) -> None:
        """New buffer should be empty with get()=None and length=0."""
        buf = RingBuffer(max_frames=250, feature_dim=FEATURE_DIM)
        assert buf.is_empty is True
        assert buf.get() is None
        assert buf.length == 0

    def test_append_and_get(self) -> None:
        """Appending (1, 30, 768) should be retrievable with correct shape."""
        torch.manual_seed(42)
        buf = RingBuffer(max_frames=250, feature_dim=FEATURE_DIM)

        features = torch.randn(1, 30, FEATURE_DIM)
        buf.append(features)

        result = buf.get()
        assert result is not None
        assert result.shape == (1, 30, FEATURE_DIM)
        assert buf.length == 30
        assert buf.is_empty is False

    def test_multiple_appends(self) -> None:
        """Appending 30 + 30 frames should give length=60."""
        torch.manual_seed(42)
        buf = RingBuffer(max_frames=250, feature_dim=FEATURE_DIM)

        buf.append(torch.randn(1, 30, FEATURE_DIM))
        buf.append(torch.randn(1, 30, FEATURE_DIM))

        assert buf.length == 60
        result = buf.get()
        assert result is not None
        assert result.shape == (1, 60, FEATURE_DIM)

    def test_fifo_eviction(self) -> None:
        """max_frames=50, append 30+30 should give length=50."""
        torch.manual_seed(42)
        buf = RingBuffer(max_frames=50, feature_dim=FEATURE_DIM)

        buf.append(torch.randn(1, 30, FEATURE_DIM))
        buf.append(torch.randn(1, 30, FEATURE_DIM))

        # 30 + 30 = 60 > 50 -> trimmed to newest 50
        assert buf.length == 50
        result = buf.get()
        assert result is not None
        assert result.shape == (1, 50, FEATURE_DIM)

    def test_reset(self) -> None:
        """Reset should clear the buffer."""
        torch.manual_seed(42)
        buf = RingBuffer(max_frames=250, feature_dim=FEATURE_DIM)

        buf.append(torch.randn(1, 30, FEATURE_DIM))
        assert buf.length == 30

        buf.reset()
        assert buf.is_empty is True
        assert buf.length == 0
        assert buf.get() is None


# ======================================================================
# StreamingContext Tests
# ======================================================================


class TestStreamingContext:
    """Tests for StreamingContext."""

    @pytest.fixture
    def ctx(self) -> StreamingContext:
        """Build a StreamingContext with small target tensors."""
        torch.manual_seed(42)
        target_mel = torch.randn(1, 50, MEL_DIM)
        target_content = torch.randn(1, 50, FEATURE_DIM)
        style_emb = torch.randn(1, FEATURE_DIM)
        return StreamingContext(
            target_mel=target_mel,
            target_content=target_content,
            style_embedding=style_emb,
            max_source_frames=250,
        )

    def test_init_stores_targets(self, ctx: StreamingContext) -> None:
        """Context should store target mel, content, and style."""
        assert ctx.target_mel.shape == (1, 50, MEL_DIM)
        assert ctx.target_content.shape == (1, 50, FEATURE_DIM)
        assert ctx.style_embedding.shape == (1, FEATURE_DIM)

    def test_add_source_chunk(self, ctx: StreamingContext) -> None:
        """Adding a source chunk should increase the source buffer length."""
        content = torch.randn(1, 30, FEATURE_DIM)
        ctx.add_source_chunk(content)
        assert ctx.source_content_buffer.length == 30
        assert ctx.num_chunks_processed == 1

    def test_build_input_returns_dict(self, ctx: StreamingContext) -> None:
        """build_stylizer_input should return a dict with expected keys."""
        content = torch.randn(1, 30, FEATURE_DIM)
        ctx.add_source_chunk(content)

        result = ctx.build_stylizer_input()
        expected_keys = {
            "content_features",
            "context_mel",
            "mask",
            "style_embedding",
            "source_start_idx",
            "source_length",
        }
        assert set(result.keys()) == expected_keys

    def test_mask_target_false_source_true(self, ctx: StreamingContext) -> None:
        """Mask should be False for target and True for source."""
        content = torch.randn(1, 30, FEATURE_DIM)
        ctx.add_source_chunk(content)

        result = ctx.build_stylizer_input()
        mask = result["mask"]

        # Target region (first 50 frames): False (unmasked)
        assert not mask[0, :50].any(), "Target region should be unmasked (False)"
        # Source region (frames 50-79): True (masked / to generate)
        assert mask[0, 50:].all(), "Source region should be masked (True)"

    def test_content_concat(self, ctx: StreamingContext) -> None:
        """Content features should be target + source concatenated."""
        content = torch.randn(1, 30, FEATURE_DIM)
        ctx.add_source_chunk(content)

        result = ctx.build_stylizer_input()
        total_T = 50 + 30  # target + source
        assert result["content_features"].shape == (1, total_T, FEATURE_DIM)
        assert result["context_mel"].shape == (1, total_T, MEL_DIM)
        assert result["source_start_idx"] == 50
        assert result["source_length"] == 30


# ======================================================================
# StreamingDestylizer Tests
# ======================================================================


class TestStreamingDestylizer:
    """Tests for StreamingDestylizer with mocked HuBERT."""

    def _make_model(self) -> StreamingDestylizer:
        """Build a StreamingDestylizer with mocked HuBERT."""
        torch.manual_seed(42)

        # Construct model, but replace hubert with mock
        # We patch StreamingHuBERT.__init__ to avoid loading the real model
        model = StreamingDestylizer.__new__(StreamingDestylizer)
        nn.Module.__init__(model)

        model.chunk_size = 30
        model.hidden_size = 768
        model.num_layers = 2

        # Use mock HuBERT
        model.hubert = MockStreamingHuBERT()
        model.hubert_proj = nn.Linear(1024, 768)
        model.input_norm = nn.LayerNorm(768)

        # Use a small Conformer (only 2 layers for speed)
        from stylestream.destylizer.conformer import ConformerEncoder

        model.conformer = ConformerEncoder(
            num_layers=2,
            hidden_size=768,
            ffn_size=256,
            num_heads=4,
            kernel_size=7,
            dropout=0.0,
            causal=True,
        )
        return model

    def test_construction(self) -> None:
        """StreamingDestylizer should be constructable with mocked HuBERT."""
        model = self._make_model()
        assert isinstance(model, nn.Module)
        assert hasattr(model, "hubert")
        assert hasattr(model, "conformer")

    def test_forward_output_keys(self) -> None:
        """Forward should return dict with 'content_features'."""
        model = self._make_model()
        model.eval()

        waveform = torch.randn(1, 16000)  # 1 second
        with torch.no_grad():
            result = model(waveform)

        assert isinstance(result, dict)
        assert "content_features" in result

    def test_content_features_shape(self) -> None:
        """Content features should have shape (B, T, 768)."""
        model = self._make_model()
        model.eval()

        waveform = torch.randn(1, 16000)  # 1 second -> 50 frames
        with torch.no_grad():
            result = model(waveform)

        cf = result["content_features"]
        assert cf.ndim == 3
        assert cf.shape[0] == 1
        assert cf.shape[2] == 768
        assert torch.isfinite(cf).all(), "Content features have nan/inf"


# ======================================================================
# StreamingDiTBlock Tests
# ======================================================================


class TestStreamingDiTBlock:
    """Tests for StreamingDiTBlock with small config."""

    @pytest.fixture
    def block(self) -> StreamingDiTBlock:
        """Build a small StreamingDiTBlock."""
        torch.manual_seed(42)
        return StreamingDiTBlock(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            ffn_size=FFN,
            dropout=0.0,
            chunk_size=CHUNK_SIZE,
        )

    def test_output_shape(self, block: StreamingDiTBlock) -> None:
        """Output should be (B, T, hidden_size)."""
        torch.manual_seed(42)
        from stylestream.stylizer.rope import RotaryPositionEmbedding

        B, T = 2, 20
        x = torch.randn(B, T, HIDDEN)
        c = torch.randn(B, HIDDEN)

        rope = RotaryPositionEmbedding(dim=HEAD_DIM)
        rope_input = x.view(B, T, HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        rope_cos, rope_sin = rope(rope_input)

        out = block(x, c, rope_cos, rope_sin)
        assert out.shape == (B, T, HIDDEN)

    def test_with_chunk_mask(self, block: StreamingDiTBlock) -> None:
        """Output should be valid with chunked causal attention mask."""
        torch.manual_seed(42)
        from stylestream.stylizer.rope import RotaryPositionEmbedding
        from stylestream.streaming.attention_mask import (
            build_chunked_causal_mask,
            chunked_causal_mask_to_attn_bias,
        )

        B, T = 2, 20
        x = torch.randn(B, T, HIDDEN)
        c = torch.randn(B, HIDDEN)

        rope = RotaryPositionEmbedding(dim=HEAD_DIM)
        rope_input = x.view(B, T, HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        rope_cos, rope_sin = rope(rope_input)

        mask = build_chunked_causal_mask(seq_len=T, chunk_size=CHUNK_SIZE)
        attn_mask = chunked_causal_mask_to_attn_bias(mask)

        out = block(x, c, rope_cos, rope_sin, attn_mask=attn_mask)
        assert out.shape == (B, T, HIDDEN)
        assert torch.isfinite(out).all(), "Output contains nan/inf with chunk mask"

    def test_gradient_flow(self, block: StreamingDiTBlock) -> None:
        """Gradients should flow through the block."""
        torch.manual_seed(42)
        from stylestream.stylizer.rope import RotaryPositionEmbedding

        B, T = 2, 20
        x = torch.randn(B, T, HIDDEN)
        c = torch.randn(B, HIDDEN)

        rope = RotaryPositionEmbedding(dim=HEAD_DIM)
        rope_input = x.view(B, T, HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        rope_cos, rope_sin = rope(rope_input)

        out = block(x, c, rope_cos, rope_sin)
        loss = out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in block.parameters()
            if p.requires_grad
        )
        assert has_grad, "Some parameters should have non-zero gradients"


# ======================================================================
# StreamingDiT Tests
# ======================================================================


class TestStreamingDiT:
    """Tests for StreamingDiT with small config."""

    @pytest.fixture
    def small_dit(self) -> StreamingDiT:
        """Build a small StreamingDiT."""
        torch.manual_seed(42)
        return StreamingDiT(
            num_layers=LAYERS,
            hidden_size=HIDDEN,
            ffn_size=FFN,
            num_heads=HEADS,
            mel_dim=MEL_DIM,
            content_dim=CONTENT_DIM,
            dropout=0.0,
            chunk_size=CHUNK_SIZE,
            max_cache_frames=250,
        )

    def test_construction(self, small_dit: StreamingDiT) -> None:
        """StreamingDiT should be constructable from small config."""
        assert isinstance(small_dit, nn.Module)
        assert small_dit.num_layers == LAYERS
        assert small_dit.hidden_size == HIDDEN
        assert small_dit.mel_dim == MEL_DIM
        assert len(small_dit.blocks) == LAYERS

    def test_forward_shape(self, small_dit: StreamingDiT) -> None:
        """Forward should return (B, T, mel_dim) velocity."""
        torch.manual_seed(42)
        B_local, T = 2, HEADS  # T = num_heads to avoid RoPE view issues

        x_t = torch.randn(B_local, T, MEL_DIM)
        t = torch.rand(B_local)
        content = torch.randn(B_local, T, CONTENT_DIM)
        context_mel = torch.randn(B_local, T, MEL_DIM)
        style_emb = torch.randn(B_local, HIDDEN)

        velocity = small_dit(x_t, t, content, context_mel, style_emb)
        assert velocity.shape == (B_local, T, MEL_DIM)
        assert torch.isfinite(velocity).all(), "Velocity contains nan/inf"


# ======================================================================
# StreamingStylizer Tests
# ======================================================================


class TestStreamingStylizer:
    """Tests for StreamingStylizer with mocked WavLM."""

    def test_construction(self) -> None:
        """StreamingStylizer should be constructable with mocked WavLM."""
        torch.manual_seed(42)
        with _patch_wavlm():
            model = StreamingStylizer(
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
                nfe=4,
                dropout=0.0,
                chunk_size=CHUNK_SIZE,
            )
        assert isinstance(model, nn.Module)
        assert hasattr(model, "dit")
        assert hasattr(model, "style_encoder")
        assert hasattr(model, "cfm")
        assert hasattr(model, "cfg")


# ======================================================================
# StreamingInferencePipeline Tests
# ======================================================================


class TestStreamingInferencePipeline:
    """Tests for StreamingInferencePipeline with all models mocked."""

    def _make_mock_destylizer(self) -> nn.Module:
        """Create a mock destylizer that returns dummy content features."""
        mock = MagicMock(spec=nn.Module)
        mock.extract_content_features = MagicMock(
            side_effect=lambda wav, **kw: torch.randn(
                1, max(wav.shape[-1] // 320, 1), FEATURE_DIM
            )
        )
        return mock

    def _make_mock_stylizer(self) -> nn.Module:
        """Create a mock stylizer with style_encoder, dit, cfm, cfg."""
        mock = MagicMock(spec=nn.Module)

        # Style encoder mock
        mock.style_encoder = MagicMock(
            side_effect=lambda wav: torch.randn(1, FEATURE_DIM)
        )

        # CFM mock (euler_sample returns random mel)
        mock.cfm = MagicMock()
        mock.cfm.euler_sample = MagicMock(
            side_effect=lambda velocity_fn, shape, nfe, device, dtype: torch.randn(
                *shape
            )
        )

        # CFG mock
        mock.cfg = MagicMock()
        mock.cfg.guided_velocity = MagicMock(
            side_effect=lambda velocity_fn, x_t, t, content_features,
            context_mel, style_emb, guidance_strength: torch.randn_like(x_t)
        )

        # DiT mock (not directly called by pipeline, but may be referenced)
        mock.dit = MagicMock()

        return mock

    def _make_mock_vocoder(self) -> nn.Module:
        """Create a mock vocoder that returns dummy waveform."""
        mock = MagicMock(spec=nn.Module)
        # Vocoder receives (1, mel_dim, T) and returns (1, T * 320)
        mock.__call__ = MagicMock(
            side_effect=lambda mel: torch.randn(1, mel.shape[-1] * 320)
        )
        mock.side_effect = mock.__call__.side_effect
        return mock

    @pytest.fixture
    def pipeline(self) -> StreamingInferencePipeline:
        """Build a pipeline with all mocked models."""
        destylizer = self._make_mock_destylizer()
        stylizer = self._make_mock_stylizer()
        vocoder = self._make_mock_vocoder()

        return StreamingInferencePipeline(
            destylizer=destylizer,
            stylizer=stylizer,
            vocoder=vocoder,
            chunk_size_ms=600,
            sample_rate=16000,
            nfe=4,
            cfg_strength=2.0,
            max_source_seconds=5.0,
            device="cpu",
        )

    def test_construction(self, pipeline: StreamingInferencePipeline) -> None:
        """Pipeline should be constructable with mock models."""
        assert isinstance(pipeline, StreamingInferencePipeline)
        assert pipeline.chunk_size_ms == 600
        assert pipeline.chunk_samples == 9600
        assert pipeline.chunk_frames == 30
        assert pipeline.max_source_frames == 250

    def test_initialize_target(self, pipeline: StreamingInferencePipeline) -> None:
        """initialize_target should set _initialized flag."""
        assert pipeline.is_initialized is False

        target_wav = torch.randn(1, 80000)  # 5 seconds
        pipeline.initialize_target(target_wav)

        assert pipeline.is_initialized is True

    def test_convert_file_returns_waveform_and_stats(
        self, pipeline: StreamingInferencePipeline
    ) -> None:
        """convert_file should return a waveform tensor and stats dict."""
        source_wav = torch.randn(32000)  # 2 seconds
        target_wav = torch.randn(80000)  # 5 seconds

        converted, stats = pipeline.convert_file(source_wav, target_wav)

        assert isinstance(converted, torch.Tensor)
        assert converted.ndim == 1 or converted.ndim == 2

    def test_stats_keys(self, pipeline: StreamingInferencePipeline) -> None:
        """Stats dict should contain expected keys."""
        source_wav = torch.randn(32000)  # 2 seconds
        target_wav = torch.randn(80000)  # 5 seconds

        _, stats = pipeline.convert_file(source_wav, target_wav)

        expected_keys = {"total_time_ms", "chunk_times_ms", "rtf", "num_chunks"}
        assert expected_keys.issubset(set(stats.keys())), (
            f"Missing stats keys: {expected_keys - set(stats.keys())}"
        )
        assert stats["num_chunks"] > 0
        assert stats["total_time_ms"] >= 0
        assert isinstance(stats["chunk_times_ms"], list)
        assert stats["rtf"] >= 0
