"""Tests for MelSpectrogramTransform.

All tests are self-contained and use synthetic waveforms (torch.randn)
so no audio files are required.
"""

from __future__ import annotations

import math

import pytest
import torch

from stylestream.utils.mel import MelSpectrogramTransform

SAMPLE_RATE = 16000
HOP_LENGTH = 320
N_MELS = 100


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_waveform(batch: int, duration_s: float) -> torch.Tensor:
    """Return a random waveform tensor of shape (batch, samples)."""
    num_samples = int(SAMPLE_RATE * duration_s)
    return torch.randn(batch, num_samples)


# ------------------------------------------------------------------
# 1. Output shape
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration_s, batch",
    [
        (1.0, 1),
        (0.5, 2),
        (6.0, 4),
        (0.01, 1),   # very short: 160 samples
        (3.7, 3),    # non-round duration
    ],
)
def test_output_shape(duration_s: float, batch: int) -> None:
    """Output must be (B, 100, T) with T = ceil(samples / 320)."""
    transform = MelSpectrogramTransform()
    waveform = _make_waveform(batch, duration_s)
    mel = transform(waveform)

    num_samples = waveform.shape[-1]
    expected_t = math.ceil(num_samples / HOP_LENGTH)

    assert mel.shape == (batch, N_MELS, expected_t), (
        f"Expected ({batch}, {N_MELS}, {expected_t}), got {mel.shape}"
    )


# ------------------------------------------------------------------
# 2. 50 Hz frame rate
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration_s, expected_frames",
    [
        (1.0, 50),    # 16000 / 320 = 50
        (6.0, 300),   # 96000 / 320 = 300
        (0.1, 5),     # 1600 / 320 = 5
        (2.0, 100),   # 32000 / 320 = 100
    ],
)
def test_50hz_frame_rate(duration_s: float, expected_frames: int) -> None:
    """Verify that common audio lengths produce exact frame counts at 50 Hz."""
    transform = MelSpectrogramTransform()
    waveform = _make_waveform(1, duration_s)
    mel = transform(waveform)

    assert mel.shape[-1] == expected_frames, (
        f"For {duration_s}s expected {expected_frames} frames, got {mel.shape[-1]}"
    )


def test_frame_rate_property() -> None:
    """The frame_rate property should report 50.0 Hz."""
    transform = MelSpectrogramTransform()
    assert transform.frame_rate == pytest.approx(50.0)


# ------------------------------------------------------------------
# 3. Batch processing consistency
# ------------------------------------------------------------------


def test_batch_consistency() -> None:
    """Each item in a batch should produce the same result as processing it alone."""
    transform = MelSpectrogramTransform()
    torch.manual_seed(42)
    waveform = _make_waveform(4, 1.0)

    # Full-batch forward
    mel_batch = transform(waveform)

    # Per-sample forward
    for i in range(waveform.shape[0]):
        mel_single = transform(waveform[i : i + 1])
        torch.testing.assert_close(
            mel_batch[i : i + 1],
            mel_single,
            atol=1e-6,
            rtol=1e-5,
            msg=f"Mismatch at batch index {i}",
        )


# ------------------------------------------------------------------
# 4. Log-mel values are finite (no NaN / Inf)
# ------------------------------------------------------------------


def test_log_mel_values_finite() -> None:
    """Output should contain no NaN or Inf values."""
    transform = MelSpectrogramTransform()
    waveform = _make_waveform(2, 2.0)
    mel = transform(waveform)

    assert torch.isfinite(mel).all(), "Log-mel spectrogram contains NaN or Inf"


def test_log_mel_with_silence() -> None:
    """Even all-zero input (silence) should produce finite values thanks to epsilon."""
    transform = MelSpectrogramTransform()
    waveform = torch.zeros(1, SAMPLE_RATE)  # 1 second of silence
    mel = transform(waveform)

    assert torch.isfinite(mel).all(), "Silence produced NaN or Inf"


# ------------------------------------------------------------------
# 5. Single sample vs. batch produce same results
# ------------------------------------------------------------------


def test_single_vs_batch() -> None:
    """A single waveform passed as (1, T) should match batch result."""
    transform = MelSpectrogramTransform()
    torch.manual_seed(123)
    waveform = torch.randn(1, SAMPLE_RATE * 3)  # 3 seconds

    mel_from_batch = transform(waveform)
    mel_from_single = transform(waveform.squeeze(0))  # (T,) input

    assert mel_from_batch.shape == mel_from_single.shape
    torch.testing.assert_close(mel_from_batch, mel_from_single, atol=1e-6, rtol=1e-5)


# ------------------------------------------------------------------
# 6. Different audio lengths
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_samples",
    [
        1,          # minimal: 1 sample
        160,        # 10 ms
        319,        # just under one hop
        320,        # exactly one hop
        321,        # just over one hop
        4800,       # 0.3 s
        16000,      # 1 s
        96000,      # 6 s (paper training segment)
    ],
)
def test_various_lengths(num_samples: int) -> None:
    """Verify correct T for a range of sample counts."""
    transform = MelSpectrogramTransform()
    waveform = torch.randn(1, num_samples)
    mel = transform(waveform)

    expected_t = math.ceil(num_samples / HOP_LENGTH)
    assert mel.shape == (1, N_MELS, expected_t), (
        f"For {num_samples} samples: expected T={expected_t}, got {mel.shape[-1]}"
    )


# ------------------------------------------------------------------
# 7. Optional flags
# ------------------------------------------------------------------


def test_center_false() -> None:
    """center=False should still produce a valid spectrogram (streaming mode)."""
    transform = MelSpectrogramTransform(center=False)
    waveform = _make_waveform(1, 1.0)
    mel = transform(waveform)

    assert mel.dim() == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == N_MELS
    assert torch.isfinite(mel).all()


def test_log_scale_off() -> None:
    """When log_scale=False, output should be raw mel power (non-negative)."""
    transform = MelSpectrogramTransform(log_scale=False)
    waveform = _make_waveform(1, 1.0)
    mel = transform(waveform)

    assert (mel >= 0).all(), "Raw mel power should be non-negative"
