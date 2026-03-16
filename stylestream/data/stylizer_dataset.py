"""Stylizer (DiT + CFM) dataset and dataloader.

The Stylizer trains on mel spectrograms using spectrogram inpainting as the
objective.  Each training sample is a fixed 6-second segment (300 frames at
50 Hz) drawn from the Emilia-EN corpus (~50k hours).

Data pipeline per sample
------------------------
1. Pick a random utterance that is >= 6 seconds.
2. Randomly crop a 300-frame segment from the pre-computed mel spectrogram
   and the corresponding content features.
3. Sample a contiguous mask spanning 70--100 % of the segment.  Unmasked
   frames become the *context*; masked frames are the generation target.
4. Select a *style reference* from a different utterance of the same speaker
   (for zero-shot training).  If the speaker has only one utterance, fall
   back to the non-masked portion of the current segment.
5. Independently decide CFG dropout for content, context, and style.

Paper references
~~~~~~~~~~~~~~~~
* Mel spectrogram: 100 bins, hop 320, n_fft 1024, 16 kHz, log-mel.
* Content features: (768, T) -- Destylizer output or HuBERT L18.
* Mask ratio: U[0.7, 1.0] contiguous region.
* CFG dropout: content 20 %, context 30 %, style 30 % (independent).
* Batch size: 64.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from stylestream.data.manifest import Manifest, Utterance
from stylestream.utils.audio import load_audio, pad_or_trim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRAME_RATE = 50  # Hz  (16 kHz / hop 320)
_CONTENT_DIM = 768
_MEL_BINS = 100
_DEFAULT_SR = 16_000
_STYLE_SAMPLES = 80_000  # 5 s at 16 kHz


# ======================================================================
# Dataset
# ======================================================================


class StylizerDataset(Dataset):
    """Dataset for Stylizer (DiT + CFM) training.

    Each item returns a fixed 6-second segment with:

    * ``mel``:              (100, 300)  mel spectrogram
    * ``content_features``: (768, 300)  content features (Destylizer output)
    * ``mask``:             (300,)      binary mask (1 = masked / to generate,
                                        0 = context)
    * ``context_mel``:      (100, 300)  = ``mel * (1 - mask)`` -- masked
                                        regions zeroed
    * ``style_waveform``:   (samples,)  style reference audio (~5 s at 16 kHz)
    * ``cfg_drop_content``: bool        whether to zero out content features
    * ``cfg_drop_context``: bool        whether to zero out context mel
    * ``cfg_drop_style``:   bool        whether to zero out style embedding
    """

    def __init__(
        self,
        manifest: Manifest,
        mel_dir: str | Path,
        content_features_dir: str | Path | None = None,
        sample_rate: int = _DEFAULT_SR,
        segment_frames: int = 300,  # 6 seconds at 50 Hz
        mask_ratio_min: float = 0.7,
        mask_ratio_max: float = 1.0,
        cfg_content_drop: float = 0.2,
        cfg_context_drop: float = 0.3,
        cfg_style_drop: float = 0.3,
        use_precomputed_mel: bool = True,
        use_precomputed_content: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        manifest :
            Utterances manifest.  Only utterances whose duration is at least
            ``segment_frames / frame_rate`` seconds are kept.
        mel_dir :
            Directory containing pre-computed mel ``.pt`` files.  Each file
            should store a tensor of shape ``(100, T)`` and be named
            ``{utterance.stem}.pt``.
        content_features_dir :
            Directory with content-feature ``.pt`` files ``(768, T)``.  Pass
            ``None`` to return zero placeholders (useful before the Destylizer
            is trained).
        sample_rate :
            Expected audio sample rate (for loading style references).
        segment_frames :
            Number of frames per training segment.  Default 300 (= 6 s).
        mask_ratio_min / mask_ratio_max :
            Range for the uniform-random contiguous mask ratio.
        cfg_content_drop / cfg_context_drop / cfg_style_drop :
            CFG dropout probabilities (independent Bernoulli per sample).
        use_precomputed_mel :
            If ``True`` (default), load mel spectrograms from *mel_dir*.
            Set to ``False`` to compute them on-the-fly (not recommended
            for large-scale training).
        use_precomputed_content :
            If ``True`` (default), load content features from
            *content_features_dir*.  Ignored when *content_features_dir* is
            ``None``.
        """
        self.mel_dir = Path(mel_dir)
        self.content_features_dir = (
            Path(content_features_dir) if content_features_dir is not None else None
        )
        self.sample_rate = sample_rate
        self.segment_frames = segment_frames
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.cfg_content_drop = cfg_content_drop
        self.cfg_context_drop = cfg_context_drop
        self.cfg_style_drop = cfg_style_drop
        self.use_precomputed_mel = use_precomputed_mel
        self.use_precomputed_content = use_precomputed_content

        # Minimum utterance duration in seconds
        min_duration = segment_frames / _FRAME_RATE

        # Filter utterances shorter than the segment length
        self.utterances: list[Utterance] = [
            u for u in manifest.utterances if u.duration >= min_duration
        ]
        n_dropped = len(manifest) - len(self.utterances)
        if n_dropped > 0:
            logger.info(
                "Filtered out %d utterances shorter than %.1f s (kept %d)",
                n_dropped,
                min_duration,
                len(self.utterances),
            )

        if len(self.utterances) == 0:
            raise ValueError(
                f"No utterances remaining after filtering for >= {min_duration} s. "
                "Check your manifest and duration values."
            )

        # Build speaker -> utterance index list for style-reference lookup
        self._speaker_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, utt in enumerate(self.utterances):
            self._speaker_to_indices[utt.speaker_id].append(idx)

        # On-the-fly mel transform (lazy import to avoid hard torchaudio dep
        # when using pre-computed features)
        self._mel_transform = None

    # ------------------------------------------------------------------
    # Length / getitem
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample as a dictionary."""
        utt = self.utterances[idx]

        # --- mel spectrogram (100, T) ------------------------------------
        mel = self._load_mel(utt)

        # --- content features (768, T) -----------------------------------
        content = self._load_content(utt, expected_frames=mel.shape[-1])

        # --- random crop to segment_frames frames ------------------------
        mel, start = self._random_crop(mel, self.segment_frames)
        content = content[:, start : start + self.segment_frames]
        # Guarantee exact size (pad if minor rounding differences)
        if content.shape[-1] < self.segment_frames:
            pad_len = self.segment_frames - content.shape[-1]
            content = torch.nn.functional.pad(content, (0, pad_len))

        # --- contiguous mask (segment_frames,) ---------------------------
        mask_ratio = random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        mask = self._generate_contiguous_mask(self.segment_frames, mask_ratio)

        # --- context mel: unmasked portion, masked frames zeroed ----------
        # mask shape (T,) -> (1, T) for broadcasting over mel bins
        context_mel = mel * (1.0 - mask.unsqueeze(0))

        # --- style reference waveform ------------------------------------
        style_waveform = self._sample_style_waveform(idx, mask)

        # --- CFG dropout (independent Bernoulli draws) -------------------
        cfg_drop_content = random.random() < self.cfg_content_drop
        cfg_drop_context = random.random() < self.cfg_context_drop
        cfg_drop_style = random.random() < self.cfg_style_drop

        return {
            "mel": mel,  # (100, 300)
            "content_features": content,  # (768, 300)
            "mask": mask,  # (300,)
            "context_mel": context_mel,  # (100, 300)
            "style_waveform": style_waveform,  # (samples,)
            "cfg_drop_content": cfg_drop_content,
            "cfg_drop_context": cfg_drop_context,
            "cfg_drop_style": cfg_drop_style,
        }

    # ------------------------------------------------------------------
    # Mask generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_contiguous_mask(
        num_frames: int, mask_ratio: float
    ) -> torch.Tensor:
        """Generate a contiguous binary mask.

        A contiguous block of ``int(round(mask_ratio * num_frames))`` frames
        is placed at a uniformly random position within the sequence.

        Parameters
        ----------
        num_frames :
            Total number of frames (e.g. 300).
        mask_ratio :
            Fraction of frames to mask (0.0--1.0).

        Returns
        -------
        torch.Tensor
            Shape ``(num_frames,)`` with ``1`` = masked, ``0`` = unmasked.
        """
        mask_len = int(round(mask_ratio * num_frames))
        mask_len = max(1, min(mask_len, num_frames))

        # Random start position for the contiguous block
        max_start = num_frames - mask_len
        start = random.randint(0, max_start) if max_start > 0 else 0

        mask = torch.zeros(num_frames, dtype=torch.float32)
        mask[start : start + mask_len] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Random crop
    # ------------------------------------------------------------------

    @staticmethod
    def _random_crop(
        tensor: torch.Tensor, target_frames: int
    ) -> tuple[torch.Tensor, int]:
        """Randomly crop a ``(C, T)`` tensor to ``(C, target_frames)``.

        If ``T < target_frames`` the tensor is zero-padded on the right.

        Returns
        -------
        tuple[torch.Tensor, int]
            ``(cropped, start_frame_idx)``
        """
        total_frames = tensor.shape[-1]

        if total_frames <= target_frames:
            # Pad to target length
            pad_len = target_frames - total_frames
            padded = torch.nn.functional.pad(tensor, (0, pad_len))
            return padded, 0

        max_start = total_frames - target_frames
        start = random.randint(0, max_start)
        return tensor[:, start : start + target_frames], start

    # ------------------------------------------------------------------
    # Feature loading helpers
    # ------------------------------------------------------------------

    def _load_mel(self, utt: Utterance) -> torch.Tensor:
        """Load or compute the mel spectrogram for *utt*.

        Returns a ``(100, T)`` tensor.
        """
        if self.use_precomputed_mel:
            mel_path = self.mel_dir / f"{utt.stem}.pt"
            mel = torch.load(mel_path, map_location="cpu", weights_only=True)
            # Accept both (100, T) and (1, 100, T)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            return mel

        # On-the-fly computation (slow, for debugging / small experiments)
        if self._mel_transform is None:
            from stylestream.utils.mel import MelSpectrogramTransform

            self._mel_transform = MelSpectrogramTransform()
        waveform = load_audio(utt.audio_path, sr=self.sample_rate)
        mel = self._mel_transform(waveform.unsqueeze(0)).squeeze(0)  # (100, T)
        return mel

    def _load_content(
        self, utt: Utterance, expected_frames: int
    ) -> torch.Tensor:
        """Load content features for *utt*.

        Returns a ``(768, T)`` tensor.  If the content-features directory is
        not set, returns zeros as a placeholder.
        """
        if self.content_features_dir is not None and self.use_precomputed_content:
            feat_path = self.content_features_dir / f"{utt.stem}.pt"
            if feat_path.exists():
                feat = torch.load(
                    feat_path, map_location="cpu", weights_only=True
                )
                if feat.dim() == 3:
                    feat = feat.squeeze(0)
                return feat

        # Placeholder zeros
        return torch.zeros(_CONTENT_DIM, expected_frames, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Style reference sampling
    # ------------------------------------------------------------------

    def _sample_style_waveform(
        self, current_idx: int, mask: torch.Tensor
    ) -> torch.Tensor:
        """Return a style-reference waveform for the given sample.

        Strategy:
          1. Prefer a *different* utterance from the same speaker (zero-shot).
          2. If the speaker has only one utterance, extract the non-masked
             portion of the current utterance.

        The returned waveform is exactly ``_STYLE_SAMPLES`` (80 000 = 5 s)
        long, padded or trimmed as needed.
        """
        utt = self.utterances[current_idx]
        speaker_indices = self._speaker_to_indices[utt.speaker_id]

        # Try to pick a different utterance from the same speaker
        if len(speaker_indices) > 1:
            candidates = [i for i in speaker_indices if i != current_idx]
            ref_idx = random.choice(candidates)
            ref_utt = self.utterances[ref_idx]
            waveform = load_audio(ref_utt.audio_path, sr=self.sample_rate)

            # Take a random 5-second crop from the reference utterance
            if waveform.shape[0] > _STYLE_SAMPLES:
                start = random.randint(0, waveform.shape[0] - _STYLE_SAMPLES)
                waveform = waveform[start : start + _STYLE_SAMPLES]
            else:
                waveform = pad_or_trim(waveform, _STYLE_SAMPLES)
            return waveform

        # Fallback: use the non-masked portion of the current utterance
        waveform = load_audio(utt.audio_path, sr=self.sample_rate)

        # Convert frame-level mask to sample-level indices
        hop = self.sample_rate // _FRAME_RATE  # 320
        unmasked_indices = (mask == 0.0).nonzero(as_tuple=True)[0]

        if len(unmasked_indices) > 0:
            # Extract unmasked samples
            first_frame = unmasked_indices[0].item()
            last_frame = unmasked_indices[-1].item()
            sample_start = first_frame * hop
            sample_end = min((last_frame + 1) * hop, waveform.shape[0])
            style_region = waveform[sample_start:sample_end]
        else:
            # Fully masked -- use the whole utterance as style
            style_region = waveform

        return pad_or_trim(style_region, _STYLE_SAMPLES)


# ======================================================================
# Collator
# ======================================================================


class StylizerCollator:
    """Collate fixed-size Stylizer batches.

    All mel spectrograms, content features, and masks share the same frame
    count (300) so they are simply stacked.  Style waveforms may differ in
    length (though :class:`StylizerDataset` already pads them to a fixed
    size) -- the collator pads to the batch maximum as a safety net.
    """

    def __call__(self, batch: list[dict]) -> dict:
        """
        Returns
        -------
        dict
            * ``mel``:              (B, 100, 300)
            * ``content_features``: (B, 768, 300)
            * ``mask``:             (B, 300)
            * ``context_mel``:      (B, 100, 300)
            * ``style_waveform``:   (B, max_style_samples)  padded
            * ``cfg_drop_content``: (B,)  bool tensor
            * ``cfg_drop_context``: (B,)  bool tensor
            * ``cfg_drop_style``:   (B,)  bool tensor
        """
        mel = torch.stack([s["mel"] for s in batch])
        content = torch.stack([s["content_features"] for s in batch])
        mask = torch.stack([s["mask"] for s in batch])
        context_mel = torch.stack([s["context_mel"] for s in batch])

        # Style waveforms: pad to the longest in the batch
        style_waveforms = [s["style_waveform"] for s in batch]
        style_waveform = pad_sequence(
            style_waveforms, batch_first=True, padding_value=0.0
        )

        cfg_drop_content = torch.tensor(
            [s["cfg_drop_content"] for s in batch], dtype=torch.bool
        )
        cfg_drop_context = torch.tensor(
            [s["cfg_drop_context"] for s in batch], dtype=torch.bool
        )
        cfg_drop_style = torch.tensor(
            [s["cfg_drop_style"] for s in batch], dtype=torch.bool
        )

        return {
            "mel": mel,
            "content_features": content,
            "mask": mask,
            "context_mel": context_mel,
            "style_waveform": style_waveform,
            "cfg_drop_content": cfg_drop_content,
            "cfg_drop_context": cfg_drop_context,
            "cfg_drop_style": cfg_drop_style,
        }


# ======================================================================
# Builder
# ======================================================================


def build_stylizer_dataloader(
    manifest: Manifest,
    mel_dir: str | Path,
    content_features_dir: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Convenience function to build a Stylizer :class:`DataLoader`.

    Parameters
    ----------
    manifest :
        Utterances manifest (will be filtered to >= 6 s internally).
    mel_dir :
        Directory with pre-computed mel ``.pt`` files.
    content_features_dir :
        Directory with content-feature ``.pt`` files.  ``None`` to use
        zero placeholders.
    batch_size :
        Per-device batch size (paper: 64).
    num_workers :
        DataLoader worker processes.
    **kwargs :
        Forwarded to :class:`StylizerDataset`.

    Returns
    -------
    DataLoader
        Ready-to-iterate dataloader with :class:`StylizerCollator`.
    """
    dataset = StylizerDataset(
        manifest=manifest,
        mel_dir=mel_dir,
        content_features_dir=content_features_dir,
        **kwargs,
    )
    collator = StylizerCollator()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
