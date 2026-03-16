"""Dataset manifest management for StyleStream.

A manifest is a CSV file tracking all utterances with metadata (path, speaker,
text, duration, dataset, subset, sample rate).  This module provides:

- :class:`Utterance` -- a single row in a manifest.
- :class:`Manifest` -- ordered collection with CSV I/O, filtering, splitting,
  sampling, merging, and statistics.
- Factory class-methods to build manifests from on-disk dataset layouts:
  LibriTTS, ESD (Emotional Speech Dataset), and GLOBE (accent-labeled).
- Helper functions :func:`build_lmg_manifest` and :func:`build_eval_manifest`
  for the combined training and evaluation sets described in the paper.

CSV format (one row per utterance)::

    audio_path,speaker_id,text,duration,dataset,subset,sample_rate
    train-clean-100/103/1240/103_1240_000000.wav,103,"Hello world",5.12,libritts,train-clean-100,24000

All CSV I/O uses the standard library ``csv`` module.  Heavy dependencies
(soundfile) are imported lazily so that lightweight operations like filtering
or merging do not require them.
"""

from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

# Canonical column order -- also doubles as the CSV header.
_COLUMNS = [
    "audio_path",
    "speaker_id",
    "text",
    "duration",
    "dataset",
    "subset",
    "sample_rate",
]


# ---------------------------------------------------------------------------
# Utterance dataclass
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    """A single utterance (audio file) and its metadata."""

    audio_path: str = ""
    speaker_id: str = ""
    text: str = ""
    duration: float = 0.0
    dataset: str = ""
    subset: str = ""
    sample_rate: int = 16000

    @property
    def stem(self) -> str:
        """Filename without extension."""
        return Path(self.audio_path).stem

    @property
    def filename(self) -> str:
        """Filename with extension."""
        return Path(self.audio_path).name

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Utterance:
        """Create an :class:`Utterance` from a dict, casting types as needed.

        Unknown keys are silently ignored so that CSVs with extra columns
        do not cause errors.
        """
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # csv.DictReader yields strings -- cast numerics explicitly.
        if "duration" in filtered and filtered["duration"] != "":
            filtered["duration"] = float(filtered["duration"])
        elif "duration" in filtered:
            filtered["duration"] = 0.0
        if "sample_rate" in filtered and filtered["sample_rate"] != "":
            filtered["sample_rate"] = int(filtered["sample_rate"])
        elif "sample_rate" in filtered:
            filtered["sample_rate"] = 16000
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Manifest class
# ---------------------------------------------------------------------------

class Manifest:
    """Manages an ordered list of :class:`Utterance` objects.

    Supports CSV I/O, field-based filtering, duration filtering, speaker-based
    train/val splitting, random sampling, merging, and summary statistics.
    """

    def __init__(self, utterances: list[Utterance] | None = None) -> None:
        self.utterances: list[Utterance] = utterances if utterances is not None else []

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.utterances)

    def __iter__(self) -> Iterator[Utterance]:
        return iter(self.utterances)

    def __getitem__(self, idx: int) -> Utterance:
        return self.utterances[idx]

    def __repr__(self) -> str:
        return f"Manifest(n={len(self.utterances)})"

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, utterance: Utterance) -> None:
        """Append a single utterance."""
        self.utterances.append(utterance)

    def extend(self, utterances: Sequence[Utterance]) -> None:
        """Append multiple utterances."""
        self.utterances.extend(utterances)

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path) -> Manifest:
        """Load a manifest from a CSV file.

        The first row must be a header whose values match the canonical column
        names.  Unknown columns are ignored; missing columns get default values.
        """
        path = Path(path)
        utterances: list[Utterance] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                utterances.append(Utterance.from_dict(row))
        logger.info("Loaded %d utterances from %s", len(utterances), path)
        return cls(utterances)

    # Backward-compatible alias
    load = from_csv

    def to_csv(self, path: str | Path) -> None:
        """Save the manifest to a CSV file.

        Creates parent directories automatically.  Uses the canonical column
        order defined in :data:`_COLUMNS`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
            writer.writeheader()
            for utt in self.utterances:
                writer.writerow(utt.to_dict())
        logger.info("Saved %d utterances to %s", len(self.utterances), path)

    def save(self, path: str | Path) -> None:
        """Write the manifest as a CSV file (alias for :meth:`to_csv`)."""
        self.to_csv(path)

    # ------------------------------------------------------------------
    # Dataset-specific factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_libritts(
        cls,
        root_dir: str | Path,
        subsets: list[str] | None = None,
    ) -> Manifest:
        """Build a manifest by scanning a LibriTTS directory tree.

        LibriTTS layout::

            root_dir/
              LibriTTS/                            # optional wrapper
                {subset}/                          # e.g. train-clean-100
                  {speaker_id}/
                    {chapter_id}/
                      {speaker_id}_{chapter_id}_{utt_id}.wav
                      {speaker_id}_{chapter_id}_{utt_id}.normalized.txt

        Parameters
        ----------
        root_dir:
            Path to the directory containing the ``LibriTTS`` folder
            (or directly containing subset folders).
        subsets:
            If given, only scan these subsets.  Otherwise scan all
            found subset directories.

        Returns
        -------
        Manifest
        """
        root = Path(root_dir)

        # Handle both root_dir/LibriTTS/<subset> and root_dir/<subset>
        libritts_dir = root / "LibriTTS"
        base = libritts_dir if libritts_dir.is_dir() else root

        all_subsets = [
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ]

        if subsets is None:
            subsets = [s for s in all_subsets if (base / s).is_dir()]

        utterances: list[Utterance] = []
        for subset in subsets:
            subset_dir = base / subset
            if not subset_dir.is_dir():
                logger.warning("Subset directory not found: %s", subset_dir)
                continue

            wav_files = sorted(subset_dir.rglob("*.wav"))
            for wav in wav_files:
                # Parse speaker_id from path: <subset>/<speaker>/<chapter>/<file>.wav
                parts = wav.relative_to(subset_dir).parts
                speaker_id = parts[0] if len(parts) >= 2 else ""

                # Read normalised transcript if available
                txt_path = wav.with_suffix(".normalized.txt")
                text = ""
                if txt_path.exists():
                    text = txt_path.read_text(encoding="utf-8").strip()

                duration = _get_duration_safe(wav)
                sample_rate = _get_sample_rate_safe(wav, fallback=24000)

                utterances.append(
                    Utterance(
                        audio_path=str(wav),
                        speaker_id=speaker_id,
                        text=text,
                        duration=duration,
                        dataset="libritts",
                        subset=subset,
                        sample_rate=sample_rate,
                    )
                )

        manifest = cls(utterances=utterances)
        logger.info(
            "Built LibriTTS manifest: %d files, %.1f hours",
            len(utterances),
            manifest.total_duration_hours(),
        )
        return manifest

    @classmethod
    def from_esd(cls, root_dir: str | Path) -> Manifest:
        """Build a manifest by scanning an ESD directory tree.

        ESD contains 10 English speakers (0001--0010) and 10 Chinese speakers
        (0011--0020), each with 5 emotions: Angry, Happy, Neutral, Sad,
        Surprise.

        ESD layout::

            root_dir/
              {speaker_id}/  (e.g. "0001" .. "0020")
                {emotion}/   (Angry, Happy, Neutral, Sad, Surprise)
                  train/
                    {speaker}_{emotion}_{utt_id}.wav
                  evaluation/
                    ...
                  test/
                    ...

        Only English speakers (IDs 0001--0010) are included since StyleStream
        targets English voice conversion.  The emotion label is stored in the
        ``subset`` field (lowercased).
        """
        root = Path(root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"ESD root not found: {root}")

        utterances: list[Utterance] = []
        emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        split_names = ["train", "evaluation", "test"]

        # English speaker IDs
        english_speaker_ids = {f"{i:04d}" for i in range(1, 11)}

        for speaker_dir in sorted(root.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name

            # Skip non-English speakers
            if speaker_id not in english_speaker_ids:
                continue

            for emotion in emotions:
                emotion_dir = speaker_dir / emotion
                if not emotion_dir.is_dir():
                    # Also try lowercase
                    emotion_dir = speaker_dir / emotion.lower()
                    if not emotion_dir.is_dir():
                        continue

                for split in split_names:
                    split_dir = emotion_dir / split
                    if not split_dir.is_dir():
                        # ESD may also use a flat layout without split dirs
                        continue

                    wav_files = sorted(split_dir.glob("*.wav"))
                    for wav in wav_files:
                        duration = _get_duration_safe(wav)
                        sample_rate = _get_sample_rate_safe(wav, fallback=16000)

                        # Read transcript if .txt sidecar exists
                        txt_path = wav.with_suffix(".txt")
                        text = ""
                        if txt_path.exists():
                            text = txt_path.read_text(encoding="utf-8").strip()

                        utterances.append(
                            Utterance(
                                audio_path=str(wav),
                                speaker_id=speaker_id,
                                text=text,
                                duration=duration,
                                dataset="esd",
                                subset=emotion.lower(),
                                sample_rate=sample_rate,
                            )
                        )

                # Also handle flat layout: emotion dir has WAVs directly
                if not any((emotion_dir / s).is_dir() for s in split_names):
                    wav_files = sorted(emotion_dir.glob("*.wav"))
                    for wav in wav_files:
                        duration = _get_duration_safe(wav)
                        sample_rate = _get_sample_rate_safe(wav, fallback=16000)

                        txt_path = wav.with_suffix(".txt")
                        text = ""
                        if txt_path.exists():
                            text = txt_path.read_text(encoding="utf-8").strip()

                        utterances.append(
                            Utterance(
                                audio_path=str(wav),
                                speaker_id=speaker_id,
                                text=text,
                                duration=duration,
                                dataset="esd",
                                subset=emotion.lower(),
                                sample_rate=sample_rate,
                            )
                        )

        manifest = cls(utterances=utterances)
        logger.info(
            "Built ESD manifest: %d files, %.1f hours, emotions: %s",
            len(utterances),
            manifest.total_duration_hours(),
            ", ".join(e.lower() for e in emotions),
        )
        return manifest

    @classmethod
    def from_globe(cls, root_dir: str | Path) -> Manifest:
        """Build a manifest by scanning a GLOBE directory tree.

        Supports two common layouts:

        **Nested structure** (accent / speaker / files)::

            root_dir/{accent}/{speaker_id}/{utt_id}.wav

        **Flat structure** (accent / files)::

            root_dir/{accent}/{speaker_id}_{utt_id}.wav

        The accent label is inferred from the first subdirectory under root
        and stored in the ``subset`` field (lowercased).
        """
        root = Path(root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"GLOBE root not found: {root}")

        utterances: list[Utterance] = []

        wav_files = sorted(root.rglob("*.wav"))
        for wav in wav_files:
            rel = wav.relative_to(root)
            parts = rel.parts

            accent = ""
            speaker_id = ""

            if len(parts) >= 3:
                # accent / speaker / file.wav
                accent = parts[0].lower()
                speaker_id = parts[1]
            elif len(parts) >= 2:
                # accent / speaker_utt.wav -- infer speaker from filename
                accent = parts[0].lower()
                stem = wav.stem
                if "_" in stem:
                    speaker_id = stem.split("_", 1)[0]
                else:
                    speaker_id = stem
            else:
                speaker_id = wav.stem

            # Read transcript if .txt sidecar exists
            txt_path = wav.with_suffix(".txt")
            text = ""
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8").strip()

            duration = _get_duration_safe(wav)
            sample_rate = _get_sample_rate_safe(wav, fallback=16000)

            utterances.append(
                Utterance(
                    audio_path=str(wav),
                    speaker_id=speaker_id,
                    text=text,
                    duration=duration,
                    dataset="globe",
                    subset=accent,
                    sample_rate=sample_rate,
                )
            )

        manifest = cls(utterances=utterances)
        logger.info(
            "Built GLOBE manifest: %d files, %.1f hours",
            len(utterances),
            manifest.total_duration_hours(),
        )
        return manifest

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(self, **kwargs: str | int | float) -> Manifest:
        """Return a new :class:`Manifest` keeping only utterances that match
        **all** given field values.

        Example::

            manifest.filter(dataset="libritts", subset="train-clean-100")
        """
        result: list[Utterance] = []
        for utt in self.utterances:
            if all(getattr(utt, k, None) == v for k, v in kwargs.items()):
                result.append(utt)
        return Manifest(result)

    def filter_duration(
        self,
        min_sec: float = 0.5,
        max_sec: float = 30.0,
    ) -> Manifest:
        """Return a new :class:`Manifest` keeping utterances within the
        duration range ``[min_sec, max_sec]`` (inclusive)."""
        return Manifest(
            [u for u in self.utterances if min_sec <= u.duration <= max_sec]
        )

    # ------------------------------------------------------------------
    # Splitting & sampling
    # ------------------------------------------------------------------

    def split_by_speaker(
        self,
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> tuple[Manifest, Manifest]:
        """Split into train/val by **speakers** (not utterances).

        All utterances for a given speaker go entirely into the train or val
        set, preventing speaker leakage.  The *train_ratio* controls the
        fraction of *speakers* assigned to the training split.

        Returns ``(train_manifest, val_manifest)``.
        """
        speakers_to_utts: dict[str, list[Utterance]] = defaultdict(list)
        for utt in self.utterances:
            speakers_to_utts[utt.speaker_id].append(utt)

        speaker_ids = sorted(speakers_to_utts.keys())
        rng = random.Random(seed)
        rng.shuffle(speaker_ids)

        n_train = max(1, int(len(speaker_ids) * train_ratio))
        train_speakers = set(speaker_ids[:n_train])

        train_utts: list[Utterance] = []
        val_utts: list[Utterance] = []
        for sid in speaker_ids:
            target = train_utts if sid in train_speakers else val_utts
            target.extend(speakers_to_utts[sid])

        return Manifest(train_utts), Manifest(val_utts)

    def sample(self, n: int, seed: int = 42) -> Manifest:
        """Return a random sample of *n* utterances.

        If *n* >= len(self) the entire manifest is returned (unshuffled).
        """
        if n >= len(self.utterances):
            return Manifest(list(self.utterances))
        rng = random.Random(seed)
        sampled = rng.sample(self.utterances, n)
        return Manifest(sampled)

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(self, other: Manifest) -> Manifest:
        """Return a new :class:`Manifest` that concatenates *self* and *other*."""
        return Manifest(list(self.utterances) + list(other.utterances))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Compute summary statistics.

        Returns a dict with keys:

        - ``total_utterances``: int
        - ``total_hours``: float
        - ``mean_duration``: float (seconds)
        - ``min_duration``: float (seconds)
        - ``max_duration``: float (seconds)
        - ``per_dataset``: dict mapping dataset name to utterance count
        - ``per_subset``: dict mapping subset name to utterance count
        - ``num_speakers``: int (unique speaker IDs)
        """
        if not self.utterances:
            return {
                "total_utterances": 0,
                "total_hours": 0.0,
                "mean_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "per_dataset": {},
                "per_subset": {},
                "num_speakers": 0,
            }

        durations = [u.duration for u in self.utterances]
        total_sec = sum(durations)

        per_dataset: dict[str, int] = defaultdict(int)
        per_subset: dict[str, int] = defaultdict(int)
        speakers: set[str] = set()

        for utt in self.utterances:
            per_dataset[utt.dataset] += 1
            per_subset[utt.subset] += 1
            if utt.speaker_id:
                speakers.add(utt.speaker_id)

        return {
            "total_utterances": len(self.utterances),
            "total_hours": total_sec / 3600.0,
            "mean_duration": total_sec / len(self.utterances),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "per_dataset": dict(per_dataset),
            "per_subset": dict(per_subset),
            "num_speakers": len(speakers),
        }

    # ------------------------------------------------------------------
    # Convenience (backward-compatible)
    # ------------------------------------------------------------------

    def total_duration_hours(self) -> float:
        """Sum of all utterance durations in hours."""
        return sum(u.duration for u in self.utterances) / 3600.0

    def datasets(self) -> set[str]:
        """Unique dataset names present in the manifest."""
        return {u.dataset for u in self.utterances}

    def speakers(self) -> set[str]:
        """Unique speaker IDs present in the manifest."""
        return {u.speaker_id for u in self.utterances if u.speaker_id}

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Manifest: {len(self)} utterances",
            f"  Total duration : {self.total_duration_hours():.1f} hours",
            f"  Speakers       : {len(self.speakers())}",
        ]
        subsets = sorted({u.subset for u in self.utterances if u.subset})
        if subsets:
            lines.append(f"  Subsets         : {', '.join(subsets)}")
        ds = sorted(self.datasets())
        if ds:
            lines.append(f"  Datasets        : {', '.join(ds)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined manifest builders
# ---------------------------------------------------------------------------

def build_lmg_manifest(
    libritts_root: str | Path,
    esd_root: str | Path,
    globe_root: str | Path,
) -> Manifest:
    """Build the LMG combined training manifest.

    The paper uses LibriTTS + MSP-Podcast + GLOBE (~1,300 h).  Since
    MSP-Podcast requires a special access agreement from UT Dallas, we
    substitute ESD (Emotional Speech Dataset) as the emotional component,
    following the recommendation in the paper analysis.

    The three individual manifests are built, duration-filtered (0.5--30 s),
    and concatenated.  Only LibriTTS *train* subsets are included (dev/test
    splits are reserved for evaluation).

    Parameters
    ----------
    libritts_root:
        Path to the LibriTTS root directory.
    esd_root:
        Path to the ESD root directory.
    globe_root:
        Path to the GLOBE root directory.

    Returns
    -------
    Manifest
        Combined LMG manifest ready for Destylizer training.
    """
    libritts = Manifest.from_libritts(libritts_root)
    esd = Manifest.from_esd(esd_root)
    globe = Manifest.from_globe(globe_root)

    # Keep only LibriTTS train subsets for training
    libritts_train = Manifest(
        [u for u in libritts.utterances if u.subset.startswith("train")]
    )

    combined = libritts_train.merge(esd).merge(globe)
    combined = combined.filter_duration(min_sec=0.5, max_sec=30.0)

    stats = combined.get_statistics()
    logger.info(
        "LMG manifest: %d utterances, %.1f hours, %d speakers",
        stats["total_utterances"],
        stats["total_hours"],
        stats["num_speakers"],
    )
    logger.info("  Per-dataset: %s", stats["per_dataset"])

    return combined


def build_eval_manifest(
    esd_root: str | Path,
    globe_root: str | Path,
    libritts_root: str | Path,
    seed: int = 42,
) -> tuple[Manifest, Manifest, list[dict]]:
    """Build the StyleStream-Test evaluation manifest.

    The paper defines a fixed evaluation set of 3,000 pairs:

    - **Source**: 300 utterances -- 100 each from ESD (neutral), GLOBE-test,
      and LibriTTS-test-clean.
    - **Target**: 10 utterances -- 5 emotion targets (from ESD: happy, angry,
      sad, surprise, neutral) and 5 accent targets (from GLOBE: british,
      american, indian, arabic, chinese).
    - **Pairs**: Every source paired with every target = 3,000 pairs.

    Each pair is a dict with keys ``source_idx``, ``target_idx``,
    ``source_path``, ``target_path``.

    Parameters
    ----------
    esd_root:
        Path to the ESD root directory.
    globe_root:
        Path to the GLOBE root directory.
    libritts_root:
        Path to the LibriTTS root directory.
    seed:
        Random seed for reproducible utterance selection.

    Returns
    -------
    tuple[Manifest, Manifest, list[dict]]
        ``(source_manifest, target_manifest, pairs_list)``
    """
    rng = random.Random(seed)

    # --- Build base manifests ---

    esd_full = Manifest.from_esd(esd_root)
    globe_full = Manifest.from_globe(globe_root)
    libritts_full = Manifest.from_libritts(libritts_root)

    # --- Source utterances (300 total) ---

    # ESD: 100 neutral utterances as source (emotion-free content)
    esd_neutral = esd_full.filter(subset="neutral")
    esd_source = _sample_with_rng(esd_neutral.utterances, 100, rng)

    # GLOBE: 100 utterances from the full set
    globe_source = _sample_with_rng(globe_full.utterances, 100, rng)

    # LibriTTS: 100 utterances from test-clean
    libritts_test = libritts_full.filter(subset="test-clean")
    libritts_source = _sample_with_rng(libritts_test.utterances, 100, rng)

    source_manifest = Manifest(esd_source + globe_source + libritts_source)

    # --- Target utterances (10 total) ---

    # 5 emotion targets from ESD (one per emotion)
    emotion_targets: list[Utterance] = []
    for emotion in ["happy", "angry", "sad", "surprise", "neutral"]:
        candidates = esd_full.filter(subset=emotion)
        if candidates.utterances:
            chosen = rng.choice(candidates.utterances)
            emotion_targets.append(chosen)

    # 5 accent targets from GLOBE (one per accent)
    accent_targets: list[Utterance] = []
    for accent in ["british", "american", "indian", "arabic", "chinese"]:
        candidates = globe_full.filter(subset=accent)
        if candidates.utterances:
            chosen = rng.choice(candidates.utterances)
            accent_targets.append(chosen)

    target_manifest = Manifest(emotion_targets + accent_targets)

    # --- Build pairs (cartesian product) ---

    pairs: list[dict] = []
    for s_idx, s_utt in enumerate(source_manifest.utterances):
        for t_idx, t_utt in enumerate(target_manifest.utterances):
            pairs.append(
                {
                    "source_idx": s_idx,
                    "target_idx": t_idx,
                    "source_path": s_utt.audio_path,
                    "target_path": t_utt.audio_path,
                }
            )

    logger.info(
        "Eval manifest: %d sources x %d targets = %d pairs",
        len(source_manifest),
        len(target_manifest),
        len(pairs),
    )

    return source_manifest, target_manifest, pairs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_duration_safe(path: Path) -> float:
    """Return audio duration in seconds, or 0.0 on error."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        logger.debug("Could not read duration for %s", path)
        return 0.0


def _get_sample_rate_safe(path: Path, fallback: int = 16000) -> int:
    """Return sample rate from file header, or *fallback* on error."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.samplerate
    except Exception:
        return fallback


def _sample_with_rng(
    items: list[Utterance],
    n: int,
    rng: random.Random,
) -> list[Utterance]:
    """Sample *n* items using a pre-seeded :class:`random.Random` instance.

    Returns all items (as a copy) if *n* >= len(items).
    """
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)
