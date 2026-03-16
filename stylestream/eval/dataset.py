"""StyleStream-Test evaluation dataset and pair generation.

The evaluation dataset consists of 300 source utterances x 10 target
utterances = 3,000 evaluation pairs for comprehensive evaluation.

Source utterances (300 total):
    - 100 from ESD (Emotional Speech Dataset)
    - 100 from GLOBE-test (accented English)
    - 100 from LibriTTS-test-clean (clean read speech)

Target utterances (10 total):
    - 5 emotion targets: happy, angry, sad, fearful, calm
    - 5 accent targets: british, american, indian, arabic, chinese

CSV format:
    source_id,target_id,source_path,target_path,source_text,target_category,target_label
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torchaudio

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class EvalPair:
    """A single source-target evaluation pair."""
    source_id: str
    target_id: str
    source_path: str
    target_path: str
    source_text: str = ""
    target_category: str = ""  # "emotion" or "accent"
    target_label: str = ""  # e.g., "happy", "british"
    source_dataset: str = ""  # "esd", "globe", "libritts"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "source_text": self.source_text,
            "target_category": self.target_category,
            "target_label": self.target_label,
            "source_dataset": self.source_dataset,
        }


@dataclass
class TargetUtterance:
    """A target utterance for style transfer."""
    utterance_id: str
    audio_path: str
    category: str  # "emotion" or "accent"
    label: str  # "happy", "british", etc.


_PAIR_COLUMNS = [
    "source_id", "target_id", "source_path", "target_path",
    "source_text", "target_category", "target_label", "source_dataset",
]

_EMOTION_LABELS = ["happy", "angry", "sad", "fearful", "calm"]
_ACCENT_LABELS = ["british", "american", "indian", "arabic", "chinese"]


# --------------------------------------------------------------------------
# EvalDataset
# --------------------------------------------------------------------------

class EvalDataset:
    """StyleStream-Test evaluation dataset.

    Loads evaluation pairs from a CSV file and provides iteration
    with optional audio loading.

    Parameters
    ----------
    pairs_csv : str or Path
        Path to the pairs CSV file.
    sample_rate : int
        Target sample rate for audio loading. Default 16000.
    """

    def __init__(self, pairs_csv: str | Path, sample_rate: int = 16000) -> None:
        self.pairs_csv = Path(pairs_csv)
        self.sample_rate = sample_rate
        self._pairs: list[EvalPair] = []
        self._load_pairs()

    def _load_pairs(self) -> None:
        """Load pairs from CSV."""
        if not self.pairs_csv.exists():
            raise FileNotFoundError(f"Pairs CSV not found: {self.pairs_csv}")

        with open(self.pairs_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._pairs.append(EvalPair(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    source_path=row["source_path"],
                    target_path=row["target_path"],
                    source_text=row.get("source_text", ""),
                    target_category=row.get("target_category", ""),
                    target_label=row.get("target_label", ""),
                    source_dataset=row.get("source_dataset", ""),
                ))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> EvalPair:
        return self._pairs[idx]

    def __iter__(self) -> Iterator[EvalPair]:
        return iter(self._pairs)

    def filter_by_category(self, category: str) -> list[EvalPair]:
        """Filter pairs by target category ('emotion' or 'accent')."""
        return [p for p in self._pairs if p.target_category == category]

    def filter_by_label(self, label: str) -> list[EvalPair]:
        """Filter pairs by target label (e.g., 'happy', 'british')."""
        return [p for p in self._pairs if p.target_label == label]

    def filter_by_source_dataset(self, dataset: str) -> list[EvalPair]:
        """Filter pairs by source dataset (e.g., 'esd', 'globe', 'libritts')."""
        return [p for p in self._pairs if p.source_dataset == dataset]

    def load_audio(self, path: str) -> torch.Tensor:
        """Load and resample audio to target sample rate.

        Returns
        -------
        Tensor
            Waveform shape (samples,) at self.sample_rate.
        """
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.squeeze(0)

    def load_pair_audio(self, pair: EvalPair) -> tuple[torch.Tensor, torch.Tensor]:
        """Load source and target audio for a pair.

        Returns
        -------
        tuple[Tensor, Tensor]
            (source_waveform, target_waveform), each shape (samples,).
        """
        source = self.load_audio(pair.source_path)
        target = self.load_audio(pair.target_path)
        return source, target

    @property
    def categories(self) -> list[str]:
        """Unique target categories."""
        return sorted(set(p.target_category for p in self._pairs))

    @property
    def labels(self) -> list[str]:
        """Unique target labels."""
        return sorted(set(p.target_label for p in self._pairs))

    @property
    def source_datasets(self) -> list[str]:
        """Unique source datasets."""
        return sorted(set(p.source_dataset for p in self._pairs))


# --------------------------------------------------------------------------
# Pair generation
# --------------------------------------------------------------------------

def build_eval_pairs(
    source_manifest_path: str | Path,
    target_dir: str | Path,
    output_csv: str | Path,
    emotion_labels: list[str] | None = None,
    accent_labels: list[str] | None = None,
) -> list[EvalPair]:
    """Build all source x target evaluation pairs.

    Parameters
    ----------
    source_manifest_path : Path
        CSV manifest of source utterances (Manifest format).
    target_dir : Path
        Directory containing target audio files organized as:
        ``target_dir/emotion/{label}.wav`` and ``target_dir/accent/{label}.wav``.
    output_csv : Path
        Where to save the pairs CSV.
    emotion_labels : list[str] or None
        Emotion target labels. Default: happy, angry, sad, fearful, calm.
    accent_labels : list[str] or None
        Accent target labels. Default: british, american, indian, arabic, chinese.

    Returns
    -------
    list[EvalPair]
        Generated evaluation pairs.
    """
    from stylestream.data.manifest import Manifest

    target_dir = Path(target_dir)
    output_csv = Path(output_csv)
    emotion_labels = emotion_labels or _EMOTION_LABELS
    accent_labels = accent_labels or _ACCENT_LABELS

    # Load source manifest
    manifest = Manifest.load(str(source_manifest_path))
    logger.info("Loaded %d source utterances from %s", len(manifest), source_manifest_path)

    # Build target utterances
    targets: list[TargetUtterance] = []
    for label in emotion_labels:
        audio_path = target_dir / "emotion" / f"{label}.wav"
        if audio_path.exists():
            targets.append(TargetUtterance(
                utterance_id=f"emotion_{label}",
                audio_path=str(audio_path),
                category="emotion",
                label=label,
            ))
        else:
            logger.warning("Target audio not found: %s", audio_path)

    for label in accent_labels:
        audio_path = target_dir / "accent" / f"{label}.wav"
        if audio_path.exists():
            targets.append(TargetUtterance(
                utterance_id=f"accent_{label}",
                audio_path=str(audio_path),
                category="accent",
                label=label,
            ))
        else:
            logger.warning("Target audio not found: %s", audio_path)

    logger.info("Found %d target utterances (%d emotion, %d accent)",
                len(targets),
                sum(1 for t in targets if t.category == "emotion"),
                sum(1 for t in targets if t.category == "accent"))

    # Generate all pairs
    pairs: list[EvalPair] = []
    for utterance in manifest:
        for target in targets:
            pairs.append(EvalPair(
                source_id=utterance.stem,
                target_id=target.utterance_id,
                source_path=utterance.audio_path,
                target_path=target.audio_path,
                source_text=utterance.text,
                target_category=target.category,
                target_label=target.label,
                source_dataset=utterance.dataset,
            ))

    logger.info("Generated %d evaluation pairs", len(pairs))

    # Save to CSV
    save_pairs_csv(pairs, output_csv)
    return pairs


def save_pairs_csv(pairs: list[EvalPair], output_csv: str | Path) -> None:
    """Save evaluation pairs to CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PAIR_COLUMNS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.to_dict())

    logger.info("Saved %d pairs to %s", len(pairs), output_csv)


def load_pairs_csv(csv_path: str | Path) -> list[EvalPair]:
    """Load evaluation pairs from CSV."""
    dataset = EvalDataset(csv_path)
    return list(dataset)
