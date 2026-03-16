"""Audio I/O utilities for StyleStream.

All functions operate on torch.Tensor (not numpy). Waveforms are expected as
1-D ``(samples,)`` tensors unless otherwise noted.  The canonical sample rate
throughout the project is 16 kHz.

File I/O is handled via ``soundfile`` (libsndfile), which supports WAV, FLAC,
and OGG out of the box.  Resampling uses ``torchaudio.transforms.Resample``
with sinc interpolation (pure-torch, no external backend needed).
"""

from __future__ import annotations

import functools
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Resampler cache
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _get_resampler(orig_sr: int, target_sr: int) -> torchaudio.transforms.Resample:
    """Return a cached :class:`torchaudio.transforms.Resample` instance.

    Using ``lru_cache`` avoids rebuilding the sinc-interpolation filter every
    time the same sample-rate pair is requested.
    """
    return torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, sr: int = 16_000) -> torch.Tensor:
    """Load an audio file, convert to mono, and resample to *sr*.

    Parameters
    ----------
    path:
        Path to an audio file (WAV, FLAC, or any format supported by
        ``soundfile`` / libsndfile).
    sr:
        Target sample rate in Hz.

    Returns
    -------
    torch.Tensor
        1-D tensor of shape ``(samples,)``.
    """
    path = Path(path)
    # soundfile returns (samples, channels) float64 ndarray
    data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    # data shape: (samples, channels)
    waveform = torch.from_numpy(data.T)  # (channels, samples)

    # Multi-channel -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if orig_sr != sr:
        waveform = resample(waveform.squeeze(0), orig_sr, sr).unsqueeze(0)

    return waveform.squeeze(0)  # (samples,)


def save_audio(path: str | Path, waveform: torch.Tensor, sr: int = 16_000) -> None:
    """Save a waveform tensor as a WAV file.

    Parameters
    ----------
    path:
        Destination file path.  Parent directories are created automatically.
    waveform:
        1-D ``(samples,)`` or 2-D ``(1, samples)`` tensor.
    sr:
        Sample rate in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)

    # soundfile expects (samples, channels)
    data = waveform.T.cpu().numpy()  # (samples, channels)
    sf.write(str(path), data, sr, subtype="FLOAT")


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample *waveform* from *orig_sr* to *target_sr*.

    Uses :func:`torchaudio.transforms.Resample` with the default sinc
    interpolation filter.  The underlying resampler object is cached so that
    repeated calls with the same sample-rate pair do not rebuild the filter.

    Parameters
    ----------
    waveform:
        1-D ``(samples,)`` tensor.
    orig_sr:
        Original sample rate.
    target_sr:
        Desired sample rate.

    Returns
    -------
    torch.Tensor
        Resampled 1-D tensor.
    """
    if orig_sr == target_sr:
        return waveform

    resampler = _get_resampler(orig_sr, target_sr)
    # Resample expects at least 2-D input: (batch/channels, samples)
    needs_squeeze = waveform.ndim == 1
    if needs_squeeze:
        waveform = waveform.unsqueeze(0)

    waveform = resampler(waveform)

    if needs_squeeze:
        waveform = waveform.squeeze(0)

    return waveform


def segment_audio(
    waveform: torch.Tensor,
    sr: int,
    segment_sec: float,
) -> list[torch.Tensor]:
    """Split *waveform* into fixed-length segments.

    The last segment is zero-padded if it is shorter than *segment_sec*.

    Parameters
    ----------
    waveform:
        1-D ``(samples,)`` tensor.
    sr:
        Sample rate in Hz.
    segment_sec:
        Desired segment length in seconds.

    Returns
    -------
    list[torch.Tensor]
        List of 1-D tensors, each of length ``int(sr * segment_sec)``.
    """
    segment_len = int(sr * segment_sec)
    segments: list[torch.Tensor] = []

    for start in range(0, len(waveform), segment_len):
        chunk = waveform[start : start + segment_len]
        chunk = pad_or_trim(chunk, segment_len)
        segments.append(chunk)

    return segments


def pad_or_trim(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad with zeros or trim *waveform* to exactly *target_length* samples.

    Parameters
    ----------
    waveform:
        1-D ``(samples,)`` tensor.
    target_length:
        Desired number of samples.

    Returns
    -------
    torch.Tensor
        1-D tensor of length *target_length*.
    """
    current = waveform.shape[-1]

    if current >= target_length:
        return waveform[:target_length]

    padding = torch.zeros(
        target_length - current, dtype=waveform.dtype, device=waveform.device
    )
    return torch.cat([waveform, padding], dim=0)


def get_duration(path: str | Path) -> float:
    """Return the duration of an audio file in seconds.

    Uses :func:`soundfile.info` so that the full waveform is **not** loaded
    into memory.

    Parameters
    ----------
    path:
        Path to an audio file.

    Returns
    -------
    float
        Duration in seconds.
    """
    path = Path(path)
    info = sf.info(str(path))
    return info.frames / info.samplerate
