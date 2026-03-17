"""Batch HuBERT-Large layer-18 feature extraction pipeline for StyleStream.

Extracts hidden states from the 18th transformer layer of
``facebook/hubert-large-ls960-ft`` at 50 Hz (hop = 320 at 16 kHz).  Features
are saved to disk as float16 ``.pt`` files for Destylizer training.

Long audio is split into overlapping chunks so that the model stays within
GPU memory and boundary artefacts are avoided.

Usage::

    from stylestream.data.hubert_extractor import HuBERTExtractor

    extractor = HuBERTExtractor(output_dir="data/features")
    extractor.extract_single("audio.wav")          # -> (768, T)
    extractor.run(manifest, skip_existing=True)     # batch over manifest
    stats = extractor.verify(manifest)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import torch
from tqdm import tqdm

from stylestream.utils.audio import load_audio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE: int = 16_000
_HOP_LENGTH: int = 320  # HuBERT CNN encoder stride -> 50 Hz at 16 kHz

# Overlap between consecutive chunks (in samples).  HuBERT-Large's CNN
# feature extractor has a receptive field of ~400 samples.  We use 10 ms
# (160 samples) of overlap on each side, which gives 320 samples total.
# After extraction we discard the first/last overlap frames from interior
# chunks so that the concatenated output has no duplicated frames.
_CHUNK_OVERLAP_SAMPLES: int = _HOP_LENGTH  # 320 samples = 1 frame at 50 Hz


# ---------------------------------------------------------------------------
# Lightweight manifest protocol
# ---------------------------------------------------------------------------
# The ``Manifest`` / ``Utterance`` classes referenced in the Phase-1
# milestone may not exist yet.  We define a structural protocol so that this
# module can work with *any* manifest implementation that exposes the same
# attributes, as well as with our own simple dataclass below.

@runtime_checkable
class UtteranceLike(Protocol):
    """Structural type for a single utterance record."""

    @property
    def audio_path(self) -> str | Path: ...

    @property
    def dataset(self) -> str: ...

    @property
    def subset(self) -> str: ...

    @property
    def stem(self) -> str: ...


@runtime_checkable
class ManifestLike(Protocol):
    """Structural type for a collection of utterances."""

    @property
    def utterances(self) -> Sequence[UtteranceLike]: ...


# Simple concrete implementations for standalone use / testing.

@dataclass
class Utterance:
    """Minimal utterance record."""

    audio_path: str | Path
    dataset: str = "default"
    subset: str = "default"
    stem: str = ""

    def __post_init__(self) -> None:
        if not self.stem:
            self.stem = Path(self.audio_path).stem


@dataclass
class Manifest:
    """Minimal manifest -- a list of :class:`Utterance` objects."""

    utterances: list[Utterance] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _expected_frames(num_samples: int) -> int:
    """Return the number of HuBERT output frames for *num_samples*.

    HuBERT-Large's CNN feature extractor produces one frame per 320 samples
    (same as mel hop-length).  The output length is ``floor(num_samples / 320)``
    for the raw CNN, but HuggingFace's ``HubertModel`` pads internally so the
    effective count is ``ceil(num_samples / 320)``.  We follow the same
    convention used by :class:`stylestream.utils.mel.MelSpectrogramTransform`.
    """
    return math.ceil(num_samples / _HOP_LENGTH)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class HuBERTExtractor:
    """Batch HuBERT layer-18 feature extraction pipeline.

    Extracts features on GPU and saves to disk as ``.pt`` files (float16).
    Handles long audio by chunking with overlap and concatenating.

    Parameters
    ----------
    output_dir:
        Root directory under which features are saved.  A ``hubert_l18/``
        subdirectory is created automatically.
    device:
        PyTorch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    layer:
        HuBERT transformer layer to extract (0-indexed, 18 by default).
    max_audio_sec:
        Audio longer than this (in seconds) is split into overlapping
        chunks to fit in GPU memory.
    batch_size:
        Number of utterances (or chunks) per forward pass.
    """

    def __init__(
        self,
        output_dir: str | Path,
        device: str = "cuda",
        layer: int = 18,
        max_audio_sec: float = 20.0,
        batch_size: int = 8,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.device = device
        self.layer = layer
        self.max_audio_samples = int(max_audio_sec * _SAMPLE_RATE)
        self.batch_size = batch_size

        # Lazily loaded model handles.
        self._model: Any | None = None
        self._extract_fn: Any | None = None

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the HuBERT model on first use."""
        if self._model is not None:
            return
        from stylestream.utils.hub import load_hubert

        logger.info(
            "Loading HuBERT-Large (layer %d) on %s ...", self.layer, self.device
        )
        self._model, self._extract_fn = load_hubert(self.device, self.layer)

    # ------------------------------------------------------------------
    # Single-file extraction
    # ------------------------------------------------------------------

    def extract_single(self, audio_path: str | Path) -> torch.Tensor:
        """Extract HuBERT layer-18 features from a single audio file.

        For audio longer than *max_audio_sec*, the waveform is split into
        overlapping chunks.  Each chunk is run through HuBERT independently,
        and the resulting feature tensors are trimmed at the overlap
        boundaries before concatenation.

        Parameters
        ----------
        audio_path:
            Path to a 16 kHz mono audio file (WAV / FLAC / etc.).

        Returns
        -------
        torch.Tensor
            Shape ``(768, T)`` on CPU, dtype float32.
        """
        self._load_model()
        assert self._extract_fn is not None  # for type checker

        waveform = load_audio(str(audio_path), sr=_SAMPLE_RATE)  # (samples,)
        total_samples = waveform.shape[0]

        if total_samples == 0:
            logger.warning("Empty audio file: %s", audio_path)
            return torch.zeros(768, 0)

        # Short audio -- single forward pass.
        if total_samples <= self.max_audio_samples:
            return self._extract_chunk(waveform)

        # Long audio -- chunk with overlap.
        chunks = self._split_with_overlap(waveform)
        features: list[torch.Tensor] = []

        for idx, chunk in enumerate(chunks):
            feat = self._extract_chunk(chunk)  # (768, T_chunk)

            # Trim overlap frames from interior chunks.
            overlap_frames = _expected_frames(_CHUNK_OVERLAP_SAMPLES)
            if idx > 0:
                # Remove leading overlap frames.
                feat = feat[:, overlap_frames:]
            if idx < len(chunks) - 1:
                # Remove trailing overlap frames.
                feat = feat[:, :-overlap_frames]

            features.append(feat)

        concatenated = torch.cat(features, dim=1)  # (768, T_total)

        # Trim or pad to the expected frame count for the full waveform so
        # that downstream consumers see exactly ceil(samples / 320) frames.
        expected_t = _expected_frames(total_samples)
        current_t = concatenated.shape[1]
        if current_t > expected_t:
            concatenated = concatenated[:, :expected_t]
        elif current_t < expected_t:
            pad = torch.zeros(768, expected_t - current_t, dtype=concatenated.dtype)
            concatenated = torch.cat([concatenated, pad], dim=1)

        return concatenated

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_batch(self, audio_paths: list[str | Path]) -> list[torch.Tensor]:
        """Extract features from a batch of audio files.

        Each file is loaded, and then all waveforms are zero-padded to the
        maximum length in the batch for a single batched forward pass.  If
        any waveform exceeds *max_audio_samples* it is split into
        overlapping chunks which are then batched together for efficient
        GPU utilisation (instead of falling back to sequential processing).

        On CUDA OOM, the batch is automatically halved and retried.

        Parameters
        ----------
        audio_paths:
            List of audio file paths.

        Returns
        -------
        list[torch.Tensor]
            One ``(768, T_i)`` float32 CPU tensor per input file.
        """
        self._load_model()
        assert self._extract_fn is not None

        # Load all waveforms.
        waveforms: list[torch.Tensor] = []
        for p in audio_paths:
            waveforms.append(load_audio(str(p), sr=_SAMPLE_RATE))

        # Separate long and short waveforms.
        short_indices: list[int] = []
        long_indices: list[int] = []
        for i, wav in enumerate(waveforms):
            if wav.shape[0] > self.max_audio_samples:
                long_indices.append(i)
            else:
                short_indices.append(i)

        results: list[torch.Tensor | None] = [None] * len(waveforms)

        # Handle long waveforms by chunking and batching the chunks.
        if long_indices:
            self._extract_long_batched(waveforms, long_indices, results)

        # Handle short waveforms in a padded batch.
        if short_indices:
            short_wavs = [waveforms[i] for i in short_indices]
            short_feats = self._batched_forward(short_wavs)
            for i, feat in zip(short_indices, short_feats):
                results[i] = feat

        return [r for r in results if r is not None]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Run over manifest
    # ------------------------------------------------------------------

    def run(self, manifest: ManifestLike, skip_existing: bool = True) -> None:
        """Extract features for every utterance in *manifest*.

        Each utterance's features are saved as:
        ``<output_dir>/hubert_l18/<dataset>/<subset>/<stem>.pt``

        Parameters
        ----------
        manifest:
            Object with an ``utterances`` attribute (list of objects with
            ``audio_path``, ``dataset``, ``subset``, ``stem`` attributes).
        skip_existing:
            If *True*, skip utterances whose feature file already exists.
        """
        self._load_model()

        utterances = list(manifest.utterances)

        # Filter already-extracted if requested.
        if skip_existing:
            pending = [u for u in utterances if not self.get_feature_path(u).exists()]
            logger.info(
                "Manifest has %d utterances, %d already extracted, %d to process.",
                len(utterances),
                len(utterances) - len(pending),
                len(pending),
            )
        else:
            pending = utterances

        if not pending:
            logger.info("Nothing to extract -- all features already exist.")
            return

        # Process in batches.
        for batch_start in tqdm(
            range(0, len(pending), self.batch_size),
            desc="HuBERT extraction",
            total=math.ceil(len(pending) / self.batch_size),
        ):
            batch_utts = pending[batch_start : batch_start + self.batch_size]
            batch_paths = [u.audio_path for u in batch_utts]

            features = self.extract_batch(batch_paths)

            for utt, feat in zip(batch_utts, features):
                out_path = self.get_feature_path(utt)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat.to(torch.float16), out_path)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_feature_path(self, utterance: UtteranceLike) -> Path:
        """Return the on-disk path for *utterance*'s HuBERT features.

        Layout: ``<output_dir>/hubert_l18/<dataset>/<subset>/<stem>.pt``
        """
        return (
            self.output_dir
            / "hubert_l18"
            / str(utterance.dataset)
            / str(utterance.subset)
            / f"{utterance.stem}.pt"
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, manifest: ManifestLike) -> dict[str, int]:
        """Verify extracted features against *manifest*.

        Checks performed for each utterance:

        1. Feature file exists.
        2. Shape is ``(768, T)``.
        3. ``T`` matches ``ceil(audio_samples / 320)`` (within tolerance of 1).
        4. No NaN or Inf values.
        5. 50 Hz sync with mel spectrogram (if a corresponding mel file
           exists at ``<output_dir>/mel/<dataset>/<subset>/<stem>.pt``).

        Returns
        -------
        dict[str, int]
            Keys: ``total``, ``missing``, ``shape_errors``, ``nan_count``,
            ``frame_mismatch``, ``sync_errors``, ``ok``.
        """
        stats = {
            "total": 0,
            "missing": 0,
            "shape_errors": 0,
            "nan_count": 0,
            "frame_mismatch": 0,
            "sync_errors": 0,
            "ok": 0,
        }

        for utt in tqdm(
            manifest.utterances, desc="Verifying HuBERT features", leave=False
        ):
            stats["total"] += 1
            feat_path = self.get_feature_path(utt)

            # 1. Existence
            if not feat_path.exists():
                stats["missing"] += 1
                continue

            feat = torch.load(feat_path, map_location="cpu", weights_only=True)

            # 2. Shape
            if feat.ndim != 2 or feat.shape[0] != 768:
                stats["shape_errors"] += 1
                logger.warning(
                    "Shape error for %s: expected (768, T), got %s",
                    feat_path,
                    tuple(feat.shape),
                )
                continue

            # Promote to float32 for NaN / frame-count checks.
            feat = feat.float()

            # 3. NaN / Inf
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                stats["nan_count"] += 1
                logger.warning("NaN/Inf detected in %s", feat_path)
                continue

            # 4. Frame count vs audio length
            try:
                wav = load_audio(str(utt.audio_path), sr=_SAMPLE_RATE)
                expected_t = _expected_frames(wav.shape[0])
                actual_t = feat.shape[1]
                if abs(actual_t - expected_t) > 1:
                    stats["frame_mismatch"] += 1
                    logger.warning(
                        "Frame mismatch for %s: expected %d, got %d",
                        feat_path,
                        expected_t,
                        actual_t,
                    )
                    continue
            except Exception as exc:
                # Audio file may have moved; log but don't count as error.
                logger.debug("Could not load audio for %s: %s", utt.audio_path, exc)

            # 5. 50 Hz sync with mel (optional)
            mel_path = (
                self.output_dir
                / "mel"
                / str(utt.dataset)
                / str(utt.subset)
                / f"{utt.stem}.pt"
            )
            if mel_path.exists():
                mel = torch.load(mel_path, map_location="cpu", weights_only=True)
                if mel.ndim == 2:
                    mel_t = mel.shape[1]
                    hubert_t = feat.shape[1]
                    if abs(mel_t - hubert_t) > 1:
                        stats["sync_errors"] += 1
                        logger.warning(
                            "Mel/HuBERT sync error for %s: mel T=%d, hubert T=%d",
                            utt.stem,
                            mel_t,
                            hubert_t,
                        )
                        continue

            stats["ok"] += 1

        logger.info(
            "Verification complete: %d total, %d ok, %d missing, "
            "%d shape errors, %d NaN/Inf, %d frame mismatches, %d sync errors.",
            stats["total"],
            stats["ok"],
            stats["missing"],
            stats["shape_errors"],
            stats["nan_count"],
            stats["frame_mismatch"],
            stats["sync_errors"],
        )
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_long_batched(
        self,
        waveforms: list[torch.Tensor],
        long_indices: list[int],
        results: list[torch.Tensor | None],
    ) -> None:
        """Chunk long waveforms and process all chunks in batches.

        Instead of falling back to sequential ``extract_single`` for each
        long waveform, this method splits every long waveform into
        overlapping chunks, collects all chunks across all long files,
        processes them in batches via :meth:`_batched_forward`, and then
        reassembles the per-file features with proper overlap trimming.

        Parameters
        ----------
        waveforms:
            Full list of loaded waveforms (indexed by *long_indices*).
        long_indices:
            Indices into *waveforms* that exceed *max_audio_samples*.
        results:
            Mutable output list; entries at *long_indices* are filled in.
        """
        # Build a flat list of all chunks and record provenance.
        all_chunks: list[torch.Tensor] = []
        # Each entry: (index_in_long_indices, chunk_index, total_chunks)
        chunk_info: list[tuple[int, int, int]] = []

        for li_pos, wav_idx in enumerate(long_indices):
            wav = waveforms[wav_idx]
            chunks = self._split_with_overlap(wav)
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_info.append((li_pos, c_idx, len(chunks)))

        # Process all chunks in batches (reuses OOM-retry logic).
        all_feats = self._batched_forward(all_chunks)

        # Reassemble per-file features.
        # Group features by their source file.
        per_file_feats: list[list[torch.Tensor]] = [[] for _ in long_indices]
        for feat, (li_pos, c_idx, n_chunks) in zip(all_feats, chunk_info):
            # Trim overlap frames from interior chunks.
            overlap_frames = _expected_frames(_CHUNK_OVERLAP_SAMPLES)
            if c_idx > 0:
                feat = feat[:, overlap_frames:]
            if c_idx < n_chunks - 1:
                feat = feat[:, :-overlap_frames]
            per_file_feats[li_pos].append(feat)

        # Concatenate and trim/pad to expected length.
        for li_pos, wav_idx in enumerate(long_indices):
            concatenated = torch.cat(per_file_feats[li_pos], dim=1)
            total_samples = waveforms[wav_idx].shape[0]
            expected_t = _expected_frames(total_samples)
            current_t = concatenated.shape[1]
            if current_t > expected_t:
                concatenated = concatenated[:, :expected_t]
            elif current_t < expected_t:
                pad = torch.zeros(
                    768, expected_t - current_t, dtype=concatenated.dtype
                )
                concatenated = torch.cat([concatenated, pad], dim=1)
            results[wav_idx] = concatenated

    def _extract_chunk(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run a single waveform through HuBERT and return features on CPU.

        Parameters
        ----------
        waveform:
            1-D ``(samples,)`` tensor (CPU).

        Returns
        -------
        torch.Tensor
            ``(768, T)`` on CPU, float32.
        """
        assert self._extract_fn is not None
        batch = waveform.unsqueeze(0)  # (1, samples)
        with torch.inference_mode():
            feat = self._extract_fn(batch)  # (1, 768, T) on self.device
        return feat.squeeze(0).cpu()  # (768, T)

    def _split_with_overlap(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """Split *waveform* into chunks of at most *max_audio_samples*.

        Adjacent chunks overlap by ``_CHUNK_OVERLAP_SAMPLES`` on each side
        so that HuBERT's CNN feature extractor has valid context at
        boundaries.

        Parameters
        ----------
        waveform:
            1-D ``(samples,)`` tensor.

        Returns
        -------
        list[torch.Tensor]
            Each chunk is a 1-D tensor.
        """
        total = waveform.shape[0]
        step = self.max_audio_samples - _CHUNK_OVERLAP_SAMPLES
        chunks: list[torch.Tensor] = []

        start = 0
        while start < total:
            end = min(start + self.max_audio_samples, total)
            chunks.append(waveform[start:end])

            # Advance by step (max_audio_samples minus one overlap width),
            # so the next chunk begins ``overlap`` samples before this one
            # ended.
            next_start = start + step

            # If the remaining tail after advancing would be shorter than
            # the overlap region, absorb it into the chunk we just appended
            # (i.e. extend the last chunk to the end of the waveform).
            if next_start < total and (total - next_start) < _CHUNK_OVERLAP_SAMPLES:
                chunks[-1] = waveform[start:total]
                break

            start = next_start

        return chunks

    def _batched_forward(
        self, waveforms: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Pad waveforms to equal length, run a batched forward pass, and
        trim each output to its actual frame count.

        On CUDA OOM, recursively halves the batch and retries.

        Parameters
        ----------
        waveforms:
            List of 1-D CPU tensors (varying lengths).

        Returns
        -------
        list[torch.Tensor]
            One ``(768, T_i)`` CPU float32 tensor per input.
        """
        assert self._extract_fn is not None

        if not waveforms:
            return []

        # Try the full batch first; on OOM, split in half.
        try:
            return self._batched_forward_inner(waveforms)
        except torch.cuda.OutOfMemoryError:
            if len(waveforms) == 1:
                # Single item still OOMs -- fall back to chunked extraction.
                logger.warning(
                    "OOM on single utterance (%d samples). "
                    "Falling back to chunked extraction.",
                    waveforms[0].shape[0],
                )
                # Temporarily lower the chunk size and use extract_single logic.
                feat = self._extract_chunk_with_fallback(waveforms[0])
                return [feat]

            mid = len(waveforms) // 2
            logger.warning(
                "CUDA OOM with batch size %d. Splitting to %d + %d.",
                len(waveforms),
                mid,
                len(waveforms) - mid,
            )
            torch.cuda.empty_cache()
            left = self._batched_forward(waveforms[:mid])
            right = self._batched_forward(waveforms[mid:])
            return left + right

    def _batched_forward_inner(
        self, waveforms: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Zero-pad, batch, extract, trim."""
        assert self._extract_fn is not None

        lengths = [w.shape[0] for w in waveforms]
        max_len = max(lengths)

        # Zero-pad to max_len.
        padded = torch.zeros(len(waveforms), max_len)
        for i, w in enumerate(waveforms):
            padded[i, : w.shape[0]] = w

        with torch.inference_mode():
            batch_feat = self._extract_fn(padded)  # (B, 768, T_max)

        batch_feat = batch_feat.cpu()

        # Trim each to its actual frame count.
        results: list[torch.Tensor] = []
        for i, n_samples in enumerate(lengths):
            t = _expected_frames(n_samples)
            results.append(batch_feat[i, :, :t])

        return results

    def _extract_chunk_with_fallback(self, waveform: torch.Tensor) -> torch.Tensor:
        """Last-resort extraction for a single waveform that OOMs even alone.

        Halves the chunk size and retries via :meth:`extract_single` logic.
        """
        # Temporarily halve max_audio_samples.
        original = self.max_audio_samples
        self.max_audio_samples = max(original // 2, _SAMPLE_RATE)  # at least 1 sec
        logger.info(
            "Reducing chunk size from %.1fs to %.1fs for OOM fallback.",
            original / _SAMPLE_RATE,
            self.max_audio_samples / _SAMPLE_RATE,
        )
        try:
            # Re-create a temporary 1-D path to reuse the chunking logic.
            total_samples = waveform.shape[0]
            if total_samples <= self.max_audio_samples:
                torch.cuda.empty_cache()
                return self._extract_chunk(waveform)

            chunks = self._split_with_overlap(waveform)
            features: list[torch.Tensor] = []
            for idx, chunk in enumerate(chunks):
                torch.cuda.empty_cache()
                feat = self._extract_chunk(chunk)
                overlap_frames = _expected_frames(_CHUNK_OVERLAP_SAMPLES)
                if idx > 0:
                    feat = feat[:, overlap_frames:]
                if idx < len(chunks) - 1:
                    feat = feat[:, :-overlap_frames]
                features.append(feat)

            concatenated = torch.cat(features, dim=1)
            expected_t = _expected_frames(total_samples)
            if concatenated.shape[1] > expected_t:
                concatenated = concatenated[:, :expected_t]
            elif concatenated.shape[1] < expected_t:
                pad = torch.zeros(
                    768, expected_t - concatenated.shape[1], dtype=concatenated.dtype
                )
                concatenated = torch.cat([concatenated, pad], dim=1)
            return concatenated
        finally:
            self.max_audio_samples = original
