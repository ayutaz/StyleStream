"""ISTFT-based waveform generation head for the StyleStream Causal Vocos vocoder.

Converts backbone hidden features into waveform by predicting STFT magnitude
and phase, then applying inverse STFT. This is the core innovation of Vocos ---
operating in the frequency domain rather than the time domain for efficient
and high-quality waveform synthesis.

StyleStream spec:
    - n_fft: 1024, hop_length: 320
    - Frequency bins: 513 (n_fft // 2 + 1)
    - Sample rate: 16 kHz
    - Output: raw waveform aligned with 50 Hz mel frames
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ISTFTHead(nn.Module):
    """ISTFT-based waveform generation head for Vocos.

    Projects backbone features to STFT magnitude and phase predictions,
    then synthesizes waveform via inverse STFT.

    Parameters
    ----------
    hidden_size : int
        Input feature dimension from backbone (512).
    n_fft : int
        FFT size (1024).
    hop_length : int
        STFT hop length (320).
    """

    def __init__(
        self,
        hidden_size: int,
        n_fft: int = 1024,
        hop_length: int = 320,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        n_freq = n_fft // 2 + 1  # 513

        # Project backbone features to magnitude + phase (2 * 513 = 1026 channels)
        self.proj = nn.Conv1d(hidden_size, 2 * n_freq, 1)

        # Hann window for ISTFT
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, features: Tensor) -> Tensor:
        """Convert backbone features to waveform.

        Parameters
        ----------
        features : Tensor
            Shape ``(B, hidden_size, T)`` from VocosBackbone.

        Returns
        -------
        Tensor
            Waveform of shape ``(B, T_samples)`` where
            ``T_samples ~ T * hop_length``.
        """
        # Project to magnitude and phase
        x = self.proj(features)  # (B, n_fft + 2, T)

        # Split into magnitude and phase components
        n_freq = self.n_fft // 2 + 1
        mag = x[:, :n_freq, :]   # (B, 513, T)
        phase = x[:, n_freq:, :]  # (B, 513, T)

        # Ensure positive magnitude via exp
        mag = torch.exp(mag)

        # Construct complex STFT: S = mag * exp(j * phase)
        # Using real/imag form to avoid complex exp
        S = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

        # Inverse STFT to produce waveform
        waveform = torch.istft(
            S,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            onesided=True,
        )

        return waveform

    def output_length(self, mel_frames: int) -> int:
        """Compute expected output waveform length for given mel frames.

        Parameters
        ----------
        mel_frames : int
            Number of input mel spectrogram frames.

        Returns
        -------
        int
            Expected number of waveform samples.
        """
        return mel_frames * self.hop_length
