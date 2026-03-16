"""Mel spectrogram transform for StyleStream.

Paper specification:
    - 100 mel bins, hop 320, n_fft 1024, 16 kHz, f_min 0, f_max 8000
    - 50 Hz frame rate (16000 / 320 = 50)
    - Log-mel scaling: log(mel + 1e-5)  (Vocos convention)

Supports both center=True (training, default) and center=False (streaming).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchaudio


class MelSpectrogramTransform(nn.Module):
    """Compute log-mel spectrograms matching the StyleStream paper settings.

    Parameters
    ----------
    n_mels : int
        Number of mel filter-bank channels.  Default 100 (paper).
    hop_length : int
        STFT hop size in samples.  Default 320 (50 Hz at 16 kHz).
    n_fft : int
        FFT window size.  Default 1024.
    sample_rate : int
        Expected input sample rate.  Default 16000.
    f_min : float
        Lowest mel filter-bank edge frequency.  Default 0.
    f_max : float
        Highest mel filter-bank edge frequency.  Default 8000.
    center : bool
        If *True* (default) the STFT is computed with centered padding so
        ``T = ceil(waveform_length / hop_length)``.  Set to *False* for
        causal / streaming inference where no future context is available.
    log_scale : bool
        If *True* (default), apply ``log(mel + 1e-5)`` (Vocos convention).
    """

    LOG_EPSILON: float = 1e-5

    def __init__(
        self,
        n_mels: int = 100,
        hop_length: int = 320,
        n_fft: int = 1024,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        center: bool = True,
        log_scale: bool = True,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.center = center
        self.log_scale = log_scale

        # Build the torchaudio transform.  It is registered as a sub-module
        # so its buffers (mel filter-bank, window) travel with .to(device).
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=center,
            power=2.0,  # power spectrogram
            norm="slaney",
            mel_scale="slaney",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz."""
        return self.sample_rate / self.hop_length  # 50.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert a batch of waveforms to log-mel spectrograms.

        Parameters
        ----------
        waveform : Tensor
            Shape ``(batch, waveform_length)`` – raw audio at *sample_rate*.

        Returns
        -------
        Tensor
            Shape ``(batch, n_mels, T)`` where
            ``T = ceil(waveform_length / hop_length)`` when *center=True*.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        orig_length = waveform.shape[-1]

        # When center=True, torch.stft applies reflect padding of n_fft//2.
        # Reflect padding requires input length >= pad + 1.  For short
        # waveforms we zero-pad to the minimum required length and trim
        # the output frames to match the *original* length.
        if self.center:
            min_length = self.n_fft // 2 + 1
            if orig_length < min_length:
                waveform = torch.nn.functional.pad(
                    waveform, (0, min_length - orig_length)
                )

        # torchaudio MelSpectrogram: (B, time) -> (B, n_mels, T')
        mel = self.mel_spec(waveform)

        # When center=True, torchaudio pads symmetrically.  The resulting
        # frame count is  1 + floor(padded_length / hop_length)  which
        # may exceed  ceil(orig_length / hop_length)  when the input
        # length is an exact multiple of hop_length or was zero-padded.
        # We trim to the expected frame count so downstream modules can
        # rely on the exact 50 Hz relationship.
        if self.center:
            expected_frames = math.ceil(orig_length / self.hop_length)
            mel = mel[:, :, :expected_frames]

        if self.log_scale:
            mel = torch.log(mel + self.LOG_EPSILON)

        return mel
