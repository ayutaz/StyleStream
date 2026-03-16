"""Tests for stylestream.utils.audio."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from stylestream.utils.audio import (
    get_duration,
    load_audio,
    pad_or_trim,
    resample,
    save_audio,
    segment_audio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000


def _make_sine(sr: int = SAMPLE_RATE, duration_sec: float = 1.0, freq: float = 440.0) -> torch.Tensor:
    """Generate a 1-D sine-wave tensor."""
    t = torch.arange(0, int(sr * duration_sec), dtype=torch.float32) / sr
    return torch.sin(2 * torch.pi * freq * t)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveAndLoadRoundtrip:
    """save_audio -> load_audio roundtrip."""

    def test_roundtrip_preserves_shape_and_values(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "roundtrip.wav"
        original = torch.randn(SAMPLE_RATE * 2)  # 2 seconds of audio

        save_audio(wav_path, original, sr=SAMPLE_RATE)
        loaded = load_audio(wav_path, sr=SAMPLE_RATE)

        assert loaded.ndim == 1
        assert loaded.shape[0] == original.shape[0]
        # WAV 16-bit introduces small quantization noise, so use a tolerance.
        torch.testing.assert_close(loaded, original, atol=1e-4, rtol=1e-4)

    def test_roundtrip_2d_input(self, tmp_path: Path) -> None:
        """save_audio should accept (1, samples) input."""
        wav_path = tmp_path / "roundtrip_2d.wav"
        original = torch.randn(1, SAMPLE_RATE)

        save_audio(wav_path, original, sr=SAMPLE_RATE)
        loaded = load_audio(wav_path, sr=SAMPLE_RATE)

        assert loaded.ndim == 1
        assert loaded.shape[0] == SAMPLE_RATE


class TestLoadAudio:
    """load_audio specifics."""

    def test_returns_1d_tensor(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "mono.wav"
        save_audio(wav_path, torch.randn(SAMPLE_RATE), sr=SAMPLE_RATE)

        result = load_audio(wav_path, sr=SAMPLE_RATE)
        assert result.ndim == 1

    def test_resamples_to_target_sr(self, tmp_path: Path) -> None:
        """Save at 24 kHz, load at 16 kHz — length should reflect new rate."""
        orig_sr = 24_000
        duration_sec = 2.0
        wav_path = tmp_path / "24k.wav"

        audio_24k = torch.randn(int(orig_sr * duration_sec))
        save_audio(wav_path, audio_24k, sr=orig_sr)

        loaded = load_audio(wav_path, sr=SAMPLE_RATE)
        expected_len = int(SAMPLE_RATE * duration_sec)
        # Allow a small tolerance for resampling boundary effects.
        assert abs(loaded.shape[0] - expected_len) <= 2

    def test_stereo_to_mono(self, tmp_path: Path) -> None:
        """A stereo file should be averaged to mono on load."""
        import soundfile as sf_test

        wav_path = tmp_path / "stereo.wav"
        stereo = torch.randn(2, SAMPLE_RATE)  # 2-channel
        # soundfile expects (samples, channels)
        sf_test.write(str(wav_path), stereo.T.numpy(), SAMPLE_RATE, subtype="FLOAT")

        loaded = load_audio(wav_path, sr=SAMPLE_RATE)
        assert loaded.ndim == 1
        assert loaded.shape[0] == SAMPLE_RATE


class TestResample:
    """resample function."""

    def test_24k_to_16k(self) -> None:
        orig_sr = 24_000
        target_sr = 16_000
        duration_sec = 1.0

        audio = torch.randn(int(orig_sr * duration_sec))
        result = resample(audio, orig_sr, target_sr)

        expected_len = int(target_sr * duration_sec)
        assert result.ndim == 1
        assert abs(result.shape[0] - expected_len) <= 2

    def test_noop_when_same_sr(self) -> None:
        audio = torch.randn(SAMPLE_RATE)
        result = resample(audio, SAMPLE_RATE, SAMPLE_RATE)
        assert result is audio  # should return the exact same object


class TestSegmentAudio:
    """segment_audio function."""

    def test_3_5s_into_1s_segments(self) -> None:
        duration_sec = 3.5
        segment_sec = 1.0
        audio = torch.randn(int(SAMPLE_RATE * duration_sec))

        segments = segment_audio(audio, SAMPLE_RATE, segment_sec)

        segment_len = int(SAMPLE_RATE * segment_sec)

        # 3.5 s / 1 s = 4 segments (last one padded)
        assert len(segments) == 4

        for seg in segments:
            assert seg.shape == (segment_len,)

        # The first 3 segments should contain only original data (no padding).
        for i in range(3):
            start = i * segment_len
            torch.testing.assert_close(segments[i], audio[start : start + segment_len])

        # Last segment: first half is original data, second half is zero-padding.
        last_real = audio[3 * segment_len :]
        assert last_real.shape[0] == int(0.5 * SAMPLE_RATE)
        torch.testing.assert_close(segments[3][: last_real.shape[0]], last_real)
        assert torch.all(segments[3][last_real.shape[0] :] == 0.0)

    def test_exact_multiple_no_extra_segment(self) -> None:
        """If the audio length is an exact multiple, no extra padded segment."""
        audio = torch.randn(SAMPLE_RATE * 2)  # exactly 2 seconds
        segments = segment_audio(audio, SAMPLE_RATE, 1.0)
        assert len(segments) == 2


class TestPadOrTrim:
    """pad_or_trim function."""

    def test_pad_shorter(self) -> None:
        audio = torch.randn(100)
        result = pad_or_trim(audio, 200)

        assert result.shape == (200,)
        torch.testing.assert_close(result[:100], audio)
        assert torch.all(result[100:] == 0.0)

    def test_trim_longer(self) -> None:
        audio = torch.randn(200)
        result = pad_or_trim(audio, 100)

        assert result.shape == (100,)
        torch.testing.assert_close(result, audio[:100])

    def test_exact_length_noop(self) -> None:
        audio = torch.randn(100)
        result = pad_or_trim(audio, 100)

        assert result.shape == (100,)
        torch.testing.assert_close(result, audio)


class TestGetDuration:
    """get_duration function."""

    def test_correct_duration(self, tmp_path: Path) -> None:
        duration_sec = 2.5
        wav_path = tmp_path / "duration.wav"
        audio = torch.randn(int(SAMPLE_RATE * duration_sec))

        save_audio(wav_path, audio, sr=SAMPLE_RATE)
        result = get_duration(wav_path)

        assert abs(result - duration_sec) < 0.01

    def test_different_sample_rate(self, tmp_path: Path) -> None:
        """Duration should be correct regardless of sample rate."""
        sr = 44_100
        duration_sec = 1.0
        wav_path = tmp_path / "44k.wav"
        audio = torch.randn(int(sr * duration_sec))

        save_audio(wav_path, audio, sr=sr)
        result = get_duration(wav_path)

        assert abs(result - duration_sec) < 0.01
