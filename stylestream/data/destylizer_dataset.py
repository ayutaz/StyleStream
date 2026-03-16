"""Dataset and DataLoader for Destylizer (HuBERT L18 + CTC/ASR) training.

The Destylizer is trained on pre-extracted HuBERT layer-18 features paired
with character-level token IDs.  Training uses full-length utterances (no
segment cropping) with bucket batching to minimise padding overhead.

Feature layout on disk::

    features_dir/
      hubert_l18/
        {dataset}/
          {subset}/
            {stem}.pt          # shape (768, T) float32

Each ``.pt`` file contains a single ``torch.Tensor`` of shape ``(768, T)``
where *T* is the number of 50 Hz frames.

Batch tensors returned by :class:`DestylizerCollator`::

    hubert_features      (B, 768, T_max)   zero-padded
    token_ids            (B, S_max)         padded with -1  (CTC ignores -1)
    feature_lengths      (B,)               actual T per sample
    token_lengths        (B,)               actual S per sample
    feature_padding_mask (B, T_max)         True = padded position
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from stylestream.data.manifest import Manifest, Utterance

logger = logging.getLogger(__name__)

# Feature rate in Hz — all StyleStream components operate at 50 Hz.
_FEATURE_RATE: int = 50

# Token padding value.  ``torch.nn.CTCLoss`` ignores targets padded with -1
# when *blank* is set to 0 and targets never contain -1.
_TOKEN_PAD: int = -1


# ======================================================================
# Dataset
# ======================================================================


class DestylizerDataset(Dataset):
    """Dataset for Destylizer training.

    Each item returns a dict with:

    - ``hubert_features`` — ``(768, T)`` float tensor of pre-extracted
      HuBERT layer-18 features.
    - ``token_ids`` — ``(S,)`` long tensor of character token IDs for
      ASR / CTC loss.
    - ``feature_length`` — int, actual *T* (before padding).
    - ``token_length`` — int, actual *S* (before padding).

    Parameters
    ----------
    manifest :
        :class:`Manifest` whose utterances have the ``text`` field populated.
    features_dir :
        Root directory containing pre-extracted HuBERT ``.pt`` files.
        Expected layout: ``features_dir/hubert_l18/{dataset}/{subset}/{stem}.pt``
    tokenizer :
        A ``CharTokenizer`` (or any object with an ``encode(text) -> list[int]``
        method).  If *None*, text is skipped and ``token_ids`` is an empty
        tensor — useful for feature-only inspection.
    max_frames :
        Utterances whose estimated frame count exceeds this value are
        silently skipped.  Default 3000 (60 s at 50 Hz).
    """

    def __init__(
        self,
        manifest: Manifest,
        features_dir: str | Path,
        tokenizer=None,
        max_frames: int = 3000,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.tokenizer = tokenizer
        self.max_frames = max_frames

        # Pre-filter utterances: drop those exceeding *max_frames* and those
        # with empty text when a tokenizer is provided.
        self.utterances: list[Utterance] = []
        self.feature_paths: list[Path] = []
        n_skipped_length = 0
        n_skipped_text = 0
        n_skipped_missing = 0

        for utt in manifest:
            # Estimate frame count from duration metadata.
            estimated_frames = int(utt.duration * _FEATURE_RATE)
            if estimated_frames > max_frames:
                n_skipped_length += 1
                continue

            if tokenizer is not None and not utt.text.strip():
                n_skipped_text += 1
                continue

            feat_path = self._feature_path(utt)
            if not feat_path.exists():
                n_skipped_missing += 1
                continue

            self.utterances.append(utt)
            self.feature_paths.append(feat_path)

        # Cache estimated lengths for the bucket sampler.
        self._estimated_lengths: list[int] = [
            max(1, int(u.duration * _FEATURE_RATE)) for u in self.utterances
        ]

        logger.info(
            "DestylizerDataset: %d utterances kept "
            "(skipped: %d too long, %d empty text, %d missing features)",
            len(self.utterances),
            n_skipped_length,
            n_skipped_text,
            n_skipped_missing,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _feature_path(self, utt: Utterance) -> Path:
        """Resolve the on-disk path to the HuBERT feature file."""
        return self.features_dir / "hubert_l18" / utt.dataset / utt.subset / f"{utt.stem}.pt"

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample as a dict.

        Keys
        ----
        hubert_features : Tensor (768, T)
        token_ids       : Tensor (S,)   — long; empty if no tokenizer
        feature_length  : int
        token_length    : int
        """
        utt = self.utterances[idx]
        feat_path = self.feature_paths[idx]

        # --- HuBERT features -----------------------------------------------
        features: torch.Tensor = torch.load(feat_path, map_location="cpu", weights_only=True)
        # Expect shape (768, T).  Some extraction pipelines may store (T, 768).
        if features.dim() == 2 and features.shape[0] != 768 and features.shape[1] == 768:
            features = features.t()  # (T, 768) -> (768, T)
        assert features.dim() == 2 and features.shape[0] == 768, (
            f"Expected (768, T) tensor, got {features.shape} for {feat_path}"
        )

        feature_length = features.shape[1]

        # --- Token IDs ------------------------------------------------------
        if self.tokenizer is not None:
            token_ids = torch.tensor(
                self.tokenizer.encode(utt.text), dtype=torch.long
            )
        else:
            token_ids = torch.zeros(0, dtype=torch.long)

        token_length = token_ids.shape[0]

        return {
            "hubert_features": features,
            "token_ids": token_ids,
            "feature_length": feature_length,
            "token_length": token_length,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def estimated_lengths(self) -> list[int]:
        """Per-utterance estimated frame counts (for :class:`BucketBatchSampler`)."""
        return self._estimated_lengths


# ======================================================================
# Collator
# ======================================================================


class DestylizerCollator:
    """Collate variable-length Destylizer samples into a padded batch.

    HuBERT features are zero-padded along the time axis to the longest
    sequence in the batch.  Token sequences are padded with ``-1`` so that
    ``torch.nn.CTCLoss`` can ignore them (its *blank* index is 0).

    Returns
    -------
    dict
        hubert_features      : (B, 768, T_max)
        token_ids            : (B, S_max)        padded with -1
        feature_lengths      : (B,)
        token_lengths        : (B,)
        feature_padding_mask : (B, T_max)        True = padded
    """

    def __call__(self, batch: list[dict]) -> dict:
        feat_lengths = torch.tensor(
            [item["feature_length"] for item in batch], dtype=torch.long
        )
        tok_lengths = torch.tensor(
            [item["token_length"] for item in batch], dtype=torch.long
        )

        t_max = int(feat_lengths.max().item())
        s_max = max(int(tok_lengths.max().item()), 1)  # at least 1 to avoid 0-dim
        bsz = len(batch)

        # --- Pad features ---------------------------------------------------
        padded_features = torch.zeros(bsz, 768, t_max, dtype=torch.float32)
        for i, item in enumerate(batch):
            t = item["feature_length"]
            padded_features[i, :, :t] = item["hubert_features"]

        # --- Pad tokens -----------------------------------------------------
        padded_tokens = torch.full(
            (bsz, s_max), _TOKEN_PAD, dtype=torch.long
        )
        for i, item in enumerate(batch):
            s = item["token_length"]
            if s > 0:
                padded_tokens[i, :s] = item["token_ids"]

        # --- Padding mask ---------------------------------------------------
        # True at positions that are padding (i.e. not real data).
        arange = torch.arange(t_max).unsqueeze(0)  # (1, T_max)
        feature_padding_mask = arange >= feat_lengths.unsqueeze(1)  # (B, T_max)

        return {
            "hubert_features": padded_features,
            "token_ids": padded_tokens,
            "feature_lengths": feat_lengths,
            "token_lengths": tok_lengths,
            "feature_padding_mask": feature_padding_mask,
        }


# ======================================================================
# Bucket batch sampler
# ======================================================================


class BucketBatchSampler(Sampler[list[int]]):
    """Group utterances of similar length into buckets for efficient batching.

    Algorithm:

    1. Sort all utterance indices by their (estimated) frame length.
    2. Partition the sorted indices into consecutive buckets of
       *batch_size* elements.
    3. Optionally shuffle the *order* of buckets each epoch (but keep
       the within-bucket composition fixed so that similar-length
       utterances stay together).

    This dramatically reduces wasted padding compared to fully random
    batching, while still providing inter-epoch diversity.

    Parameters
    ----------
    lengths :
        Per-utterance lengths (in frames).  Typically from
        :pyattr:`DestylizerDataset.estimated_lengths`.
    batch_size :
        Number of utterances per batch.
    drop_last :
        If *True*, the final bucket is dropped when it contains fewer
        than *batch_size* utterances.
    shuffle :
        If *True*, the bucket order is randomised each time
        :meth:`__iter__` is called.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Sort indices by length so consecutive buckets have similar sizes.
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Build buckets (each is a list of indices).
        self.buckets: list[list[int]] = []
        for start in range(0, len(sorted_indices), batch_size):
            bucket = sorted_indices[start : start + batch_size]
            if self.drop_last and len(bucket) < batch_size:
                continue
            self.buckets.append(bucket)

    def __iter__(self) -> Iterator[list[int]]:
        bucket_order = list(range(len(self.buckets)))
        if self.shuffle:
            random.shuffle(bucket_order)
        for idx in bucket_order:
            yield self.buckets[idx]

    def __len__(self) -> int:
        return len(self.buckets)


# ======================================================================
# Builder
# ======================================================================


def build_destylizer_dataloader(
    manifest: Manifest,
    features_dir: str | Path,
    tokenizer=None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    max_frames: int = 3000,
) -> DataLoader:
    """Build a complete :class:`DataLoader` for Destylizer training.

    Uses :class:`BucketBatchSampler` to group similar-length utterances,
    and :class:`DestylizerCollator` to pad each batch.

    Parameters
    ----------
    manifest :
        Utterance manifest (text must be populated for CTC training).
    features_dir :
        Root directory with pre-extracted HuBERT ``.pt`` files.
    tokenizer :
        ``CharTokenizer`` instance (or *None* for feature-only mode).
    batch_size :
        Utterances per batch.
    num_workers :
        DataLoader worker processes.
    shuffle :
        Whether to shuffle bucket order.
    max_frames :
        Skip utterances exceeding this frame count.

    Returns
    -------
    DataLoader
        Yields dicts produced by :class:`DestylizerCollator`.
    """
    dataset = DestylizerDataset(
        manifest=manifest,
        features_dir=features_dir,
        tokenizer=tokenizer,
        max_frames=max_frames,
    )

    sampler = BucketBatchSampler(
        lengths=dataset.estimated_lengths,
        batch_size=batch_size,
        drop_last=False,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=DestylizerCollator(),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
