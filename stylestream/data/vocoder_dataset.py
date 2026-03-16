"""Vocoder (Causal Vocos) dataset and dataloader.

Each training example is a random 2-second crop of a LibriTTS utterance,
yielding a perfectly aligned mel / waveform pair.

Paper specification:
    - Segment length: 2 seconds (100 frames at 50 Hz, 32000 samples at 16 kHz)
    - Mel spectrogram: (100, 100) — 100 mel bins x 100 time frames
    - Waveform: (32000,) — raw audio aligned with the mel frames
    - Batch size: 64 (per GPU)
    - Silence-only crops are skipped (up to 3 retries per sample)

Mel spectrograms may be loaded from pre-computed ``.pt`` files for speed or
computed on-the-fly using :class:`~stylestream.utils.mel.MelSpectrogramTransform`.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from stylestream.data.manifest import Manifest
from stylestream.utils.audio import load_audio
from stylestream.utils.mel import MelSpectrogramTransform

logger = logging.getLogger(__name__)


class VocoderDataset(Dataset):
    """Dataset for Causal Vocos vocoder training.

    Each item returns a random 2-second crop with:
    - mel: (100, 100) mel spectrogram
    - waveform: (32000,) aligned waveform
    """

    _MAX_SILENCE_RETRIES: int = 3

    def __init__(
        self,
        manifest: Manifest,
        audio_dir: str | Path,
        mel_dir: str | Path | None = None,
        sample_rate: int = 16_000,
        segment_sec: float = 2.0,
        hop_length: int = 320,
        n_mels: int = 100,
        min_energy: float = 1e-6,
        use_precomputed_mel: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        manifest:
            Utterance manifest.  Only utterances whose ``duration >= segment_sec``
            are kept so that every crop is guaranteed to succeed without padding
            under normal circumstances.
        audio_dir:
            Root directory containing resampled 16 kHz audio files.  Audio is
            located at ``audio_dir / utterance.audio_path``.
        mel_dir:
            Root directory with pre-computed mel ``.pt`` files arranged as
            ``mel_dir/{dataset}/{subset}/{stem}.pt``.  Ignored when
            *use_precomputed_mel* is ``False``.
        sample_rate:
            Expected sample rate of the audio (16 000).
        segment_sec:
            Segment length in seconds (2.0).
        hop_length:
            Mel hop length in samples (320).
        n_mels:
            Number of mel frequency bins (100).
        min_energy:
            Mean-squared energy threshold below which a crop is treated as
            silence and re-drawn.
        use_precomputed_mel:
            If ``True`` (default), load pre-computed mel tensors from
            *mel_dir*.  If ``False``, compute mel spectrograms on-the-fly.
        """
        self.audio_dir = Path(audio_dir)
        self.mel_dir = Path(mel_dir) if mel_dir is not None else None
        self.sample_rate = sample_rate
        self.segment_sec = segment_sec
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.min_energy = min_energy
        self.use_precomputed_mel = use_precomputed_mel

        # Derived constants
        self.segment_frames = int(segment_sec * sample_rate / hop_length)  # 100
        self.segment_samples = int(segment_sec * sample_rate)  # 32000

        # Filter manifest: keep only utterances long enough for a full crop
        self.utterances = [u for u in manifest if u.duration >= segment_sec]
        n_dropped = len(manifest) - len(self.utterances)
        if n_dropped > 0:
            logger.info(
                "VocoderDataset: dropped %d / %d utterances shorter than %.1fs",
                n_dropped,
                len(manifest),
                segment_sec,
            )
        logger.info(
            "VocoderDataset: %d utterances (%.1f hours)",
            len(self.utterances),
            sum(u.duration for u in self.utterances) / 3600.0,
        )

        # Lazy-initialised mel transform for on-the-fly computation.
        # Built on first use so that the torchaudio buffers are created in the
        # worker process (avoids pickling issues with DataLoader workers).
        self._mel_transform: MelSpectrogramTransform | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def mel_transform(self) -> MelSpectrogramTransform:
        """Lazily create the :class:`MelSpectrogramTransform` on first access.

        This avoids pickling CUDA buffers when DataLoader spawns workers.
        """
        if self._mel_transform is None:
            self._mel_transform = MelSpectrogramTransform(
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
        return self._mel_transform

    def _mel_path(self, utt) -> Path:
        """Resolve the path to a pre-computed mel tensor.

        Layout: ``mel_dir / {dataset} / {subset} / {stem}.pt``
        """
        assert self.mel_dir is not None
        return self.mel_dir / utt.dataset / utt.subset / f"{utt.stem}.pt"

    def _load_audio(self, utt) -> torch.Tensor:
        """Load waveform for *utt* as a 1-D float tensor at *sample_rate*."""
        path = self.audio_dir / utt.audio_path
        return load_audio(path, sr=self.sample_rate)

    def _load_mel(self, utt) -> torch.Tensor:
        """Load pre-computed mel spectrogram for *utt*.

        Returns a tensor of shape ``(n_mels, T)``.
        """
        mel = torch.load(self._mel_path(utt), weights_only=True)
        # Accept both (n_mels, T) and (1, n_mels, T)
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        return mel

    def _compute_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram on-the-fly.

        Parameters
        ----------
        waveform:
            1-D tensor of shape ``(samples,)``.

        Returns
        -------
        Tensor
            Shape ``(n_mels, T)``.
        """
        mel = self.mel_transform(waveform)  # (1, n_mels, T)
        return mel.squeeze(0)  # (n_mels, T)

    # ------------------------------------------------------------------
    # Crop & silence
    # ------------------------------------------------------------------

    def _random_crop_aligned(
        self,
        waveform: torch.Tensor,
        mel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly crop aligned waveform and mel to the segment length.

        The crop is anchored on a mel frame index so that perfect alignment is
        maintained:

        - ``mel_start`` = random integer in ``[0, T - segment_frames]``
        - ``wave_start`` = ``mel_start * hop_length``
        - ``mel_crop``: ``(n_mels, segment_frames)``
        - ``wave_crop``: ``(segment_samples,)``

        If the waveform or mel is shorter than expected (e.g. rounding at file
        boundaries), zero-padding is applied.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(waveform_crop, mel_crop)`` with shapes ``(segment_samples,)``
            and ``(n_mels, segment_frames)`` respectively.
        """
        mel_frames = mel.shape[-1]

        if mel_frames <= self.segment_frames:
            # Utterance is (just barely) long enough — take the whole thing
            # and pad if a few frames short.
            mel_start = 0
        else:
            mel_start = random.randint(0, mel_frames - self.segment_frames)

        wave_start = mel_start * self.hop_length

        # Crop mel
        mel_crop = mel[:, mel_start : mel_start + self.segment_frames]
        if mel_crop.shape[-1] < self.segment_frames:
            pad = self.segment_frames - mel_crop.shape[-1]
            mel_crop = torch.nn.functional.pad(mel_crop, (0, pad))

        # Crop waveform
        wave_crop = waveform[wave_start : wave_start + self.segment_samples]
        if wave_crop.shape[-1] < self.segment_samples:
            pad = self.segment_samples - wave_crop.shape[-1]
            wave_crop = torch.nn.functional.pad(wave_crop, (0, pad))

        return wave_crop, mel_crop

    def _is_silence(self, waveform: torch.Tensor) -> bool:
        """Return ``True`` if *waveform* is predominantly silence.

        Silence is detected by comparing the mean squared energy against
        :attr:`min_energy`.
        """
        energy = waveform.pow(2).mean()
        return energy.item() < self.min_energy

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single training example.

        Returns
        -------
        dict
            ``mel``  — ``(n_mels, segment_frames)`` i.e. ``(100, 100)``
            ``waveform`` — ``(segment_samples,)`` i.e. ``(32000,)``

        If the initial random crop lands on a silent region the crop is
        re-drawn up to :attr:`_MAX_SILENCE_RETRIES` times.  On the final
        attempt the crop is accepted regardless.
        """
        utt = self.utterances[idx]

        # --- Load waveform ---------------------------------------------------
        waveform = self._load_audio(utt)

        # --- Load or compute mel ---------------------------------------------
        if self.use_precomputed_mel and self.mel_dir is not None:
            mel = self._load_mel(utt)
        else:
            mel = self._compute_mel(waveform)

        # --- Random crop with silence retry ----------------------------------
        for attempt in range(self._MAX_SILENCE_RETRIES + 1):
            wave_crop, mel_crop = self._random_crop_aligned(waveform, mel)
            if attempt == self._MAX_SILENCE_RETRIES or not self._is_silence(wave_crop):
                break

        return {
            "mel": mel_crop,       # (100, 100)
            "waveform": wave_crop,  # (32000,)
        }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------


def build_vocoder_dataloader(
    manifest: Manifest,
    audio_dir: str | Path,
    mel_dir: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    **kwargs,
) -> DataLoader:
    """Build a :class:`DataLoader` for vocoder training.

    All crops have identical shape so the default collate function works
    without a custom ``collate_fn``.

    Parameters
    ----------
    manifest:
        Utterance manifest (will be filtered to ``duration >= 2s``).
    audio_dir:
        Root directory with resampled 16 kHz audio.
    mel_dir:
        Root directory with pre-computed mel ``.pt`` files (optional).
    batch_size:
        Samples per batch (default 64, matching the paper).
    num_workers:
        DataLoader worker processes.
    shuffle:
        Shuffle samples each epoch (default ``True``).
    pin_memory:
        Pin host memory for faster GPU transfer (default ``True``).
    drop_last:
        Drop the last incomplete batch (default ``True``).
    **kwargs:
        Forwarded to :class:`VocoderDataset`.

    Returns
    -------
    DataLoader
        Ready-to-iterate dataloader yielding dicts with ``mel`` and
        ``waveform`` tensors.
    """
    dataset = VocoderDataset(
        manifest=manifest,
        audio_dir=audio_dir,
        mel_dir=mel_dir,
        **kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
