"""Tests for manifest management.

Exercises the :class:`Manifest` and :class:`Utterance` data classes defined in
``stylestream.data.manifest``.  All tests are self-contained and use ``tmp_path``
for file I/O so no external data is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from stylestream.data.manifest import Manifest, Utterance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_utterances(
    n: int = 12,
    datasets: list[str] | None = None,
    speakers: list[str] | None = None,
) -> list[Utterance]:
    """Create *n* synthetic :class:`Utterance` objects with varying metadata.

    Durations cycle through 2 s -- 10 s.  Datasets and speakers cycle through
    the provided lists (or defaults).
    """
    if datasets is None:
        datasets = ["libritts", "esd", "globe"]
    if speakers is None:
        speakers = ["spk_001", "spk_002", "spk_003", "spk_004"]

    subsets = ["train", "dev", "test"]
    utts: list[Utterance] = []
    for i in range(n):
        duration = 2.0 + (i % 9)  # 2 s .. 10 s, cycling
        utts.append(
            Utterance(
                audio_path=f"data/raw/{datasets[i % len(datasets)]}/utt_{i:04d}.wav",
                dataset=datasets[i % len(datasets)],
                subset=subsets[i % len(subsets)],
                speaker_id=speakers[i % len(speakers)],
                duration=duration,
                sample_rate=16000,
                text=f"This is utterance number {i}",
            )
        )
    return utts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestManifest:
    """Core Manifest functionality."""

    # -- creation / basics ------------------------------------------------

    def test_create_empty(self) -> None:
        """An empty Manifest should have length 0 and be iterable."""
        m = Manifest()
        assert len(m) == 0
        assert list(m) == []

    def test_add_utterances(self) -> None:
        """Adding utterances one-by-one should grow the manifest."""
        m = Manifest()
        u1 = Utterance(audio_path="a.wav", dataset="ds1")
        u2 = Utterance(audio_path="b.wav", dataset="ds2")

        m.add(u1)
        assert len(m) == 1

        m.add(u2)
        assert len(m) == 2
        assert m[0] is u1
        assert m[1] is u2

    def test_extend(self) -> None:
        """``extend`` should append a sequence of utterances."""
        m = Manifest()
        utts = _make_utterances(5)
        m.extend(utts)
        assert len(m) == 5

    # -- CSV round-trip ---------------------------------------------------

    def test_csv_roundtrip(self, tmp_path: Path) -> None:
        """save -> load should produce an identical manifest."""
        utts = _make_utterances(10)
        original = Manifest(utterances=utts)
        csv_path = tmp_path / "manifest.csv"

        original.save(csv_path)
        assert csv_path.exists()

        loaded = Manifest.load(csv_path)
        assert len(loaded) == len(original)

        for orig, reloaded in zip(original, loaded):
            assert orig.audio_path == reloaded.audio_path
            assert orig.dataset == reloaded.dataset
            assert orig.subset == reloaded.subset
            assert orig.speaker_id == reloaded.speaker_id
            assert abs(orig.duration - reloaded.duration) < 1e-6
            assert orig.sample_rate == reloaded.sample_rate
            assert orig.text == reloaded.text

    def test_csv_header_correct(self, tmp_path: Path) -> None:
        """The CSV should have the expected column headers."""
        m = Manifest(utterances=_make_utterances(2))
        csv_path = tmp_path / "header.csv"
        m.save(csv_path)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        expected = [
            "audio_path",
            "speaker_id",
            "text",
            "duration",
            "dataset",
            "subset",
            "sample_rate",
        ]
        assert header == expected

    def test_csv_roundtrip_empty(self, tmp_path: Path) -> None:
        """An empty manifest should round-trip through CSV."""
        m = Manifest()
        csv_path = tmp_path / "empty.csv"
        m.save(csv_path)
        loaded = Manifest.load(csv_path)
        assert len(loaded) == 0

    # -- filtering --------------------------------------------------------

    def test_filter_by_dataset(self) -> None:
        """filter(dataset=...) should return only matching utterances."""
        utts = _make_utterances(12)
        m = Manifest(utterances=utts)

        libritts_only = m.filter(dataset="libritts")
        assert len(libritts_only) > 0
        assert all(u.dataset == "libritts" for u in libritts_only)

        # Every utterance in the filtered manifest must also be in the original
        original_paths = {u.audio_path for u in m}
        for u in libritts_only:
            assert u.audio_path in original_paths

    def test_filter_by_subset(self) -> None:
        """filter(subset=...) should only keep the requested subset."""
        m = Manifest(utterances=_make_utterances(12))
        train_only = m.filter(subset="train")
        assert len(train_only) > 0
        assert all(u.subset == "train" for u in train_only)

    def test_filter_by_duration(self) -> None:
        """Filtering by exact duration matches only those utterances."""
        m = Manifest(utterances=_make_utterances(12))
        # Duration cycles: 2,3,4,...,10,2,3,4
        # Only utterances with duration==5.0 should pass
        filtered = m.filter(duration=5.0)
        assert all(u.duration == 5.0 for u in filtered)

    def test_filter_combined(self) -> None:
        """Multiple filter kwargs should be AND-ed."""
        utts = _make_utterances(30)
        m = Manifest(utterances=utts)

        filtered = m.filter(dataset="libritts", subset="train")
        assert len(filtered) > 0
        for u in filtered:
            assert u.dataset == "libritts"
            assert u.subset == "train"

    def test_filter_no_match(self) -> None:
        """Filtering with a non-existent value returns an empty manifest."""
        m = Manifest(utterances=_make_utterances(5))
        filtered = m.filter(dataset="nonexistent")
        assert len(filtered) == 0

    # -- merge (extend) ---------------------------------------------------

    def test_merge(self) -> None:
        """Merging two manifests produces the union of their utterances."""
        m1 = Manifest(utterances=_make_utterances(5))
        m2 = Manifest(utterances=_make_utterances(7))

        merged = Manifest()
        merged.extend(m1.utterances)
        merged.extend(m2.utterances)

        assert len(merged) == 12

    # -- sample (not implemented — test the concept) ----------------------

    def test_sample(self) -> None:
        """A manifest subset can be obtained via standard Python slicing.

        Note: Manifest does not have a built-in ``sample`` method, so we
        test the pattern of creating a sub-manifest from a random selection.
        """
        import random

        utts = _make_utterances(20)
        m = Manifest(utterances=utts)

        k = 5
        random.seed(42)
        sampled = random.sample(m.utterances, k)
        sub = Manifest(utterances=sampled)

        assert len(sub) == k
        # Every sampled utterance should come from the original
        orig_paths = {u.audio_path for u in m}
        for u in sub:
            assert u.audio_path in orig_paths

    # -- statistics -------------------------------------------------------

    def test_statistics(self) -> None:
        """total_duration_hours, datasets, speakers, and summary should work."""
        utts = _make_utterances(12)
        m = Manifest(utterances=utts)

        # total_duration_hours
        total_sec = sum(u.duration for u in utts)
        assert abs(m.total_duration_hours() - total_sec / 3600) < 1e-6

        # datasets
        ds = m.datasets()
        assert isinstance(ds, set)
        assert "libritts" in ds

        # speakers
        sp = m.speakers()
        assert isinstance(sp, set)
        assert len(sp) > 0

        # summary (just check it doesn't crash and returns a string)
        s = m.summary()
        assert isinstance(s, str)
        assert "Manifest" in s

    # -- len and getitem --------------------------------------------------

    def test_len_and_getitem(self) -> None:
        """__len__ and __getitem__ should behave like a list."""
        utts = _make_utterances(5)
        m = Manifest(utterances=utts)

        assert len(m) == 5

        for i in range(5):
            assert m[i] is utts[i]

        # Negative indexing should work (Python list semantics)
        assert m[-1] is utts[-1]

    def test_iteration(self) -> None:
        """Iterating over a Manifest should yield all utterances in order."""
        utts = _make_utterances(5)
        m = Manifest(utterances=utts)
        collected = list(m)
        assert len(collected) == 5
        for i, u in enumerate(collected):
            assert u is utts[i]


# ---------------------------------------------------------------------------
# Utterance tests
# ---------------------------------------------------------------------------


class TestUtterance:
    """Tests for the Utterance dataclass."""

    def test_stem_property(self) -> None:
        """stem should return the filename without extension."""
        u = Utterance(audio_path="data/raw/libritts/train/103_1240_000000.wav")
        assert u.stem == "103_1240_000000"

    def test_filename_property(self) -> None:
        """filename should return the full filename with extension."""
        u = Utterance(audio_path="data/raw/libritts/train/103_1240_000000.wav")
        assert u.filename == "103_1240_000000.wav"

    def test_default_values(self) -> None:
        """All fields except audio_path should have sensible defaults."""
        u = Utterance(audio_path="a.wav")
        assert u.dataset == ""
        assert u.subset == ""
        assert u.speaker_id == ""
        assert u.duration == 0.0
        assert u.sample_rate == 16000
        assert u.text == ""

    def test_stem_with_nested_path(self) -> None:
        """stem should work correctly for deeply-nested paths."""
        u = Utterance(audio_path="a/b/c/d/my_file.flac")
        assert u.stem == "my_file"
        assert u.filename == "my_file.flac"
