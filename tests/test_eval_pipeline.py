"""Tests for the StyleStream evaluation pipeline.

Tests cover:
    - EvalResult / PairResult data structures
    - SimilarityEvaluator cosine similarity
    - Evaluator registry
    - WhisperEvaluator text normalization
    - EvalDataset pair loading/filtering
    - MetricsAggregator statistics computation
    - StyleProbing linear classifier
    - Visualization (matplotlib integration)
    - End-to-end CLI argument parsing
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# =====================================================================
# EvalResult / PairResult
# =====================================================================


class TestEvalResult:
    def test_creation(self):
        from stylestream.eval.base import EvalResult

        r = EvalResult(
            metric_name="wer",
            value=9.2,
            direction="lower_is_better",
            unit="percent",
        )
        assert r.metric_name == "wer"
        assert r.value == 9.2
        assert r.direction == "lower_is_better"

    def test_is_better_than_lower(self):
        from stylestream.eval.base import EvalResult

        r = EvalResult(
            metric_name="wer", value=8.0, direction="lower_is_better"
        )
        assert r.is_better_than(9.0) is True
        assert r.is_better_than(7.0) is False

    def test_is_better_than_higher(self):
        from stylestream.eval.base import EvalResult

        r = EvalResult(
            metric_name="s_sim", value=0.9, direction="higher_is_better"
        )
        assert r.is_better_than(0.8) is True
        assert r.is_better_than(0.95) is False

    def test_pair_result(self):
        from stylestream.eval.base import PairResult

        pr = PairResult(
            source_id="s1",
            target_id="t1",
            metrics={"wer": 9.2, "s_sim": 0.85},
        )
        assert pr.source_id == "s1"
        assert len(pr.metrics) == 2

    def test_eval_result_default_metadata(self):
        from stylestream.eval.base import EvalResult

        r = EvalResult(metric_name="wer", value=9.2, direction="lower_is_better")
        assert r.metadata == {}

    def test_eval_result_with_metadata(self):
        from stylestream.eval.base import EvalResult

        r = EvalResult(
            metric_name="wer",
            value=9.2,
            direction="lower_is_better",
            metadata={"transcription": "hello", "reference": "hello world"},
        )
        assert r.metadata["transcription"] == "hello"

    def test_is_better_than_equal(self):
        from stylestream.eval.base import EvalResult

        r = EvalResult(metric_name="wer", value=9.0, direction="lower_is_better")
        # Equal value is not better
        assert r.is_better_than(9.0) is False

    def test_pair_result_default_metadata(self):
        from stylestream.eval.base import PairResult

        pr = PairResult(source_id="s1", target_id="t1")
        assert pr.metrics == {}
        assert pr.metadata == {}


# =====================================================================
# SimilarityEvaluator (cosine similarity)
# =====================================================================


class TestSimilarityEvaluator:
    def _make_dummy_class(self):
        from stylestream.eval.base import SimilarityEvaluator

        class DummySim(SimilarityEvaluator):
            @property
            def metric_name(self):
                return "test"

            def _load_model(self):
                pass

            def extract_embedding(self, audio):
                return torch.randn(256)

        return DummySim

    def test_cosine_similarity_identical(self):
        DummySim = self._make_dummy_class()
        evaluator = DummySim(device="cpu")
        emb = torch.randn(256)
        sim = evaluator.cosine_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        DummySim = self._make_dummy_class()
        evaluator = DummySim(device="cpu")
        emb1 = torch.zeros(256)
        emb1[0] = 1.0
        emb2 = torch.zeros(256)
        emb2[1] = 1.0
        sim = evaluator.cosine_similarity(emb1, emb2)
        assert abs(sim) < 1e-5

    def test_cosine_similarity_opposite(self):
        DummySim = self._make_dummy_class()
        evaluator = DummySim(device="cpu")
        emb = torch.randn(256)
        sim = evaluator.cosine_similarity(emb, -emb)
        assert abs(sim + 1.0) < 1e-5

    def test_cosine_similarity_range(self):
        DummySim = self._make_dummy_class()
        evaluator = DummySim(device="cpu")
        for _ in range(10):
            emb1 = torch.randn(256)
            emb2 = torch.randn(256)
            sim = evaluator.cosine_similarity(emb1, emb2)
            assert -1.0 - 1e-5 <= sim <= 1.0 + 1e-5

    def test_evaluate_pair_requires_target(self):
        DummySim = self._make_dummy_class()
        evaluator = DummySim(device="cpu")
        with pytest.raises(ValueError, match="target_audio"):
            evaluator.evaluate_pair(torch.randn(16000), target_audio=None)

    def test_direction_is_higher_is_better(self):
        DummySim = self._make_dummy_class()
        evaluator = DummySim(device="cpu")
        assert evaluator.direction == "higher_is_better"

    def test_unit_is_cosine_similarity(self):
        DummySim = self._make_dummy_class()
        evaluator = DummySim(device="cpu")
        assert evaluator.unit == "cosine_similarity"


# =====================================================================
# Evaluator Registry
# =====================================================================


class TestRegistry:
    def test_available_metrics(self):
        from stylestream.eval.registry import available_metrics

        metrics = available_metrics()
        assert "wer" in metrics
        assert "s_sim" in metrics
        assert "a_sim" in metrics
        assert "e_sim" in metrics
        assert "utmos" in metrics

    def test_get_evaluator_unknown(self):
        from stylestream.eval.registry import get_evaluator

        with pytest.raises(ValueError, match="Unknown metric"):
            get_evaluator("nonexistent")

    def test_get_evaluator_class_resolution(self):
        from stylestream.eval.registry import _EVALUATOR_CLASSES

        # Verify all registered classes have valid module paths
        for metric, (mod_path, cls_name) in _EVALUATOR_CLASSES.items():
            assert mod_path.startswith("stylestream.eval.")
            assert cls_name.endswith("Evaluator")

    def test_available_metrics_returns_sorted(self):
        from stylestream.eval.registry import available_metrics

        metrics = available_metrics()
        assert metrics == sorted(metrics)

    def test_aliases_present(self):
        from stylestream.eval.registry import available_metrics

        metrics = available_metrics()
        # cer and mos are aliases
        assert "cer" in metrics
        assert "mos" in metrics

    def test_available_metrics_returns_list(self):
        from stylestream.eval.registry import available_metrics

        metrics = available_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) >= 5


# =====================================================================
# WhisperEvaluator text normalization
# =====================================================================


class TestTextNormalization:
    def test_lowercase(self):
        from stylestream.eval.whisper_evaluator import normalize_text

        assert normalize_text("Hello World") == "hello world"

    def test_punctuation_removal(self):
        from stylestream.eval.whisper_evaluator import normalize_text

        result = normalize_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_contraction_preservation(self):
        from stylestream.eval.whisper_evaluator import normalize_text

        result = normalize_text("don't stop it's working")
        # Apostrophes in contractions should be preserved
        assert "don't" in result or "dont" in result

    def test_whitespace_collapse(self):
        from stylestream.eval.whisper_evaluator import normalize_text

        assert normalize_text("  hello   world  ") == "hello world"

    def test_empty_string(self):
        from stylestream.eval.whisper_evaluator import normalize_text

        assert normalize_text("") == ""

    def test_numbers_preserved(self):
        from stylestream.eval.whisper_evaluator import normalize_text

        result = normalize_text("I have 42 cats")
        assert "42" in result

    def test_all_punctuation(self):
        from stylestream.eval.whisper_evaluator import normalize_text

        result = normalize_text("...!!??")
        # Should be empty or whitespace-only after stripping
        assert result.strip() == ""


# =====================================================================
# WhisperEvaluator properties
# =====================================================================


class TestWhisperEvaluator:
    def test_metric_name_wer(self):
        from stylestream.eval.whisper_evaluator import WhisperEvaluator

        e = WhisperEvaluator(device="cpu", use_cer=False)
        assert e.metric_name == "wer"

    def test_metric_name_cer(self):
        from stylestream.eval.whisper_evaluator import WhisperEvaluator

        e = WhisperEvaluator(device="cpu", use_cer=True)
        assert e.metric_name == "cer"

    def test_direction(self):
        from stylestream.eval.whisper_evaluator import WhisperEvaluator

        e = WhisperEvaluator(device="cpu")
        assert e.direction == "lower_is_better"

    def test_requires_source_text(self):
        from stylestream.eval.whisper_evaluator import WhisperEvaluator

        e = WhisperEvaluator(device="cpu")
        with pytest.raises(ValueError, match="source_text"):
            e.evaluate_pair(torch.randn(16000), source_text=None)

    def test_unit_is_percent(self):
        from stylestream.eval.whisper_evaluator import WhisperEvaluator

        e = WhisperEvaluator(device="cpu")
        assert e.unit == "percent"

    def test_not_loaded_initially(self):
        from stylestream.eval.whisper_evaluator import WhisperEvaluator

        e = WhisperEvaluator(device="cpu")
        assert not e.is_loaded


# =====================================================================
# EvalDataset
# =====================================================================


class TestEvalDataset:
    def _create_pairs_csv(self, tmpdir: Path, n_pairs: int = 10) -> Path:
        csv_path = tmpdir / "pairs.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_id",
                    "target_id",
                    "source_path",
                    "target_path",
                    "source_text",
                    "target_category",
                    "target_label",
                    "source_dataset",
                ],
            )
            writer.writeheader()
            for i in range(n_pairs):
                cat = "emotion" if i % 2 == 0 else "accent"
                label = "happy" if cat == "emotion" else "british"
                writer.writerow(
                    {
                        "source_id": f"src_{i}",
                        "target_id": f"tgt_{i % 3}",
                        "source_path": f"/data/source/{i}.wav",
                        "target_path": f"/data/target/{label}.wav",
                        "source_text": f"test sentence {i}",
                        "target_category": cat,
                        "target_label": label,
                        "source_dataset": "libritts" if i % 3 == 0 else "esd",
                    }
                )
        return csv_path

    def test_load_pairs(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 10)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        assert len(ds) == 10

    def test_filter_by_category(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 10)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        emotion_pairs = ds.filter_by_category("emotion")
        accent_pairs = ds.filter_by_category("accent")
        assert len(emotion_pairs) + len(accent_pairs) == 10

    def test_filter_by_label(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 10)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        happy_pairs = ds.filter_by_label("happy")
        assert all(p.target_label == "happy" for p in happy_pairs)

    def test_filter_by_source_dataset(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 10)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        libritts_pairs = ds.filter_by_source_dataset("libritts")
        assert all(p.source_dataset == "libritts" for p in libritts_pairs)

    def test_iteration(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 5)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        pairs = list(ds)
        assert len(pairs) == 5

    def test_getitem(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 5)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        pair = ds[0]
        assert pair.source_id == "src_0"

    def test_file_not_found(self):
        from stylestream.eval.dataset import EvalDataset

        with pytest.raises(FileNotFoundError):
            EvalDataset("/nonexistent/pairs.csv")

    def test_properties(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 10)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        assert "emotion" in ds.categories
        assert "accent" in ds.categories
        assert len(ds.labels) > 0
        assert len(ds.source_datasets) > 0

    def test_filter_empty_result(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 10)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        result = ds.filter_by_category("nonexistent")
        assert result == []

    def test_getitem_last(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 5)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        pair = ds[4]
        assert pair.source_id == "src_4"

    def test_source_text_loaded(self, tmp_path):
        csv_path = self._create_pairs_csv(tmp_path, 3)
        from stylestream.eval.dataset import EvalDataset

        ds = EvalDataset(csv_path)
        pair = ds[0]
        assert pair.source_text == "test sentence 0"


# =====================================================================
# EvalPair
# =====================================================================


class TestEvalPair:
    def test_to_dict(self):
        from stylestream.eval.dataset import EvalPair

        pair = EvalPair(
            source_id="s1",
            target_id="t1",
            source_path="/a.wav",
            target_path="/b.wav",
            source_text="hello",
            target_category="emotion",
            target_label="happy",
        )
        d = pair.to_dict()
        assert d["source_id"] == "s1"
        assert d["target_category"] == "emotion"

    def test_to_dict_all_fields(self):
        from stylestream.eval.dataset import EvalPair

        pair = EvalPair(
            source_id="s1",
            target_id="t1",
            source_path="/a.wav",
            target_path="/b.wav",
            source_text="hello",
            target_category="emotion",
            target_label="happy",
            source_dataset="esd",
        )
        d = pair.to_dict()
        assert d["source_dataset"] == "esd"
        assert d["source_path"] == "/a.wav"
        assert d["target_path"] == "/b.wav"
        assert d["source_text"] == "hello"
        assert d["target_label"] == "happy"

    def test_defaults(self):
        from stylestream.eval.dataset import EvalPair

        pair = EvalPair(
            source_id="s1",
            target_id="t1",
            source_path="/a.wav",
            target_path="/b.wav",
        )
        assert pair.source_text == ""
        assert pair.target_category == ""
        assert pair.target_label == ""
        assert pair.source_dataset == ""


# =====================================================================
# MetricsAggregator
# =====================================================================


class TestMetricsAggregator:
    def _make_results(self, n: int = 100):
        from stylestream.eval.aggregator import PairMetrics

        results = []
        for i in range(n):
            results.append(
                PairMetrics(
                    source_id=f"src_{i}",
                    target_id=f"tgt_{i % 10}",
                    target_category="emotion" if i % 2 == 0 else "accent",
                    target_label="happy" if i % 2 == 0 else "british",
                    source_dataset="libritts" if i % 3 == 0 else "esd",
                    metrics={
                        "wer": 10.0 + i * 0.1,
                        "s_sim": 0.8 + i * 0.001,
                    },
                )
            )
        return results

    def test_overall_stats(self):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(100))
        stats = agg.overall_stats()
        assert "wer" in stats
        assert "s_sim" in stats
        assert stats["wer"].count == 100

    def test_stats_by_category(self):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(100))
        cat_stats = agg.stats_by_category()
        assert "emotion" in cat_stats
        assert "accent" in cat_stats

    def test_stats_by_label(self):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(100))
        label_stats = agg.stats_by_label()
        assert "happy" in label_stats
        assert "british" in label_stats

    def test_95_ci(self):
        from stylestream.eval.aggregator import compute_stats

        values = [10.0] * 100  # constant values
        stats = compute_stats(values, "test")
        assert stats.mean == 10.0
        assert stats.std == 0.0
        assert stats.ci_95_low == 10.0
        assert stats.ci_95_high == 10.0

    def test_stats_single_value(self):
        from stylestream.eval.aggregator import compute_stats

        stats = compute_stats([5.0], "test")
        assert stats.mean == 5.0
        assert stats.count == 1

    def test_stats_empty(self):
        from stylestream.eval.aggregator import compute_stats

        stats = compute_stats([], "test")
        assert stats.count == 0
        assert stats.mean == 0

    def test_save_detailed_csv(self, tmp_path):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(10))
        csv_path = tmp_path / "detailed.csv"
        agg.save_detailed_csv(csv_path)
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 10

    def test_save_summary_json(self, tmp_path):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(10))
        json_path = tmp_path / "summary.json"
        agg.save_summary_json(json_path)
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "overall" in data
        assert "by_category" in data

    def test_to_markdown_table(self):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator(
            paper_baselines={
                "stylestream_offline": {"wer": 9.2, "s_sim": 0.852}
            }
        )
        agg.add_results(self._make_results(10))
        md = agg.to_markdown_table(mode="offline")
        assert "Metric" in md
        assert "Ours" in md
        assert "Paper" in md

    def test_to_latex_table(self):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator(
            paper_baselines={"stylestream_offline": {"wer": 9.2}}
        )
        agg.add_results(self._make_results(10))
        latex = agg.to_latex_table(mode="offline")
        assert "tabular" in latex
        assert "toprule" in latex

    def test_add_result_single(self):
        from stylestream.eval.aggregator import MetricsAggregator, PairMetrics

        agg = MetricsAggregator()
        agg.add_result(
            PairMetrics(
                source_id="s1",
                target_id="t1",
                metrics={"wer": 10.0},
            )
        )
        assert agg.count == 1

    def test_metric_names(self):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(5))
        names = agg.metric_names
        assert "wer" in names
        assert "s_sim" in names
        assert names == sorted(names)

    def test_stats_by_source_dataset(self):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(100))
        ds_stats = agg.stats_by_source_dataset()
        assert "libritts" in ds_stats
        assert "esd" in ds_stats

    def test_markdown_table_no_baselines(self):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(10))
        md = agg.to_markdown_table(mode="offline")
        # Should still produce a table, with "---" for paper values
        assert "Metric" in md
        assert "---" in md

    def test_csv_has_all_columns(self, tmp_path):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator()
        agg.add_results(self._make_results(5))
        csv_path = tmp_path / "detailed.csv"
        agg.save_detailed_csv(csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        assert "source_id" in fieldnames
        assert "target_id" in fieldnames
        assert "wer" in fieldnames
        assert "s_sim" in fieldnames

    def test_summary_json_structure(self, tmp_path):
        from stylestream.eval.aggregator import MetricsAggregator

        agg = MetricsAggregator(
            paper_baselines={"stylestream_offline": {"wer": 9.2}}
        )
        agg.add_results(self._make_results(10))
        json_path = tmp_path / "summary.json"
        agg.save_summary_json(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert "total_pairs" in data
        assert data["total_pairs"] == 10
        assert "paper_baselines" in data
        assert "by_source_dataset" in data


# =====================================================================
# Probing
# =====================================================================


class TestStyleProbing:
    def test_linear_probe(self):
        from stylestream.eval.probing import LinearProbe

        probe = LinearProbe(768, 10)
        x = torch.randn(4, 768)
        logits = probe(x)
        assert logits.shape == (4, 10)

    def test_linear_probe_single_sample(self):
        from stylestream.eval.probing import LinearProbe

        probe = LinearProbe(64, 3)
        x = torch.randn(1, 64)
        logits = probe(x)
        assert logits.shape == (1, 3)

    def test_prepare_features(self):
        from stylestream.eval.probing import StyleProbing

        prober = StyleProbing(feature_dim=768, device="cpu")
        features = [torch.randn(50, 768), torch.randn(30, 768)]
        labels = [0, 1]
        X, y = prober.prepare_features(features, labels)
        assert X.shape == (2, 768)
        assert y.shape == (2,)

    def test_prepare_features_1d_input(self):
        from stylestream.eval.probing import StyleProbing

        prober = StyleProbing(feature_dim=64, device="cpu")
        # Already pooled 1D features
        features = [torch.randn(64), torch.randn(64)]
        labels = [0, 1]
        X, y = prober.prepare_features(features, labels)
        assert X.shape == (2, 64)
        assert y.shape == (2,)

    def test_train_probe(self):
        from stylestream.eval.probing import StyleProbing

        prober = StyleProbing(
            feature_dim=64, device="cpu", num_epochs=5, batch_size=8
        )
        # Create simple linearly separable data
        X = torch.cat(
            [torch.randn(20, 64) + 2, torch.randn(20, 64) - 2]
        )
        y = torch.cat([torch.zeros(20), torch.ones(20)]).long()
        probe, accuracy = prober.train_probe(X, y)
        assert 0.0 <= accuracy <= 1.0
        # Should be able to classify linearly separable data
        assert accuracy > 0.7

    def test_train_probe_with_validation(self):
        from stylestream.eval.probing import StyleProbing

        prober = StyleProbing(
            feature_dim=32, device="cpu", num_epochs=10, batch_size=8
        )
        train_X = torch.cat(
            [torch.randn(30, 32) + 3, torch.randn(30, 32) - 3]
        )
        train_y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
        val_X = torch.cat(
            [torch.randn(10, 32) + 3, torch.randn(10, 32) - 3]
        )
        val_y = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        probe, accuracy = prober.train_probe(train_X, train_y, val_X, val_y)
        assert 0.0 <= accuracy <= 1.0

    def test_run_probing(self):
        from stylestream.eval.probing import StyleProbing

        prober = StyleProbing(feature_dim=32, device="cpu", num_epochs=3)
        features = [torch.randn(10, 32) + i for i in range(6)]
        labels = [0, 0, 1, 1, 2, 2]
        result = prober.run_probing(
            "speaker", "test_features", features, labels
        )
        assert result.task == "speaker"
        assert result.feature_source == "test_features"
        assert result.num_classes == 3
        assert 0.0 <= result.accuracy <= 1.0

    def test_run_probing_with_validation(self):
        from stylestream.eval.probing import StyleProbing

        prober = StyleProbing(feature_dim=32, device="cpu", num_epochs=3)
        train_features = [torch.randn(10, 32) + i for i in range(4)]
        train_labels = [0, 0, 1, 1]
        val_features = [torch.randn(10, 32) + i for i in range(2)]
        val_labels = [0, 1]
        result = prober.run_probing(
            "emotion",
            "destylizer_offline",
            train_features,
            train_labels,
            val_features,
            val_labels,
        )
        assert result.task == "emotion"
        assert result.feature_source == "destylizer_offline"
        assert result.num_classes == 2
        assert result.num_samples == 4

    def test_probing_result_to_dict(self):
        from stylestream.eval.probing import ProbingResult

        r = ProbingResult(
            task="speaker",
            feature_source="hubert",
            accuracy=0.86,
            num_classes=100,
            num_samples=1000,
            num_epochs=20,
        )
        d = r.to_dict()
        assert d["task"] == "speaker"
        assert d["accuracy"] == 0.86

    def test_probing_result_to_dict_with_metadata(self):
        from stylestream.eval.probing import ProbingResult

        r = ProbingResult(
            task="speaker",
            feature_source="hubert",
            accuracy=0.86,
            num_classes=100,
            num_samples=1000,
            num_epochs=20,
            metadata={"lr": 0.001},
        )
        d = r.to_dict()
        assert d["lr"] == 0.001

    def test_probing_result_accuracy_rounding(self):
        from stylestream.eval.probing import ProbingResult

        r = ProbingResult(
            task="accent",
            feature_source="destylizer",
            accuracy=0.43567,
            num_classes=5,
            num_samples=500,
            num_epochs=20,
        )
        d = r.to_dict()
        assert d["accuracy"] == 0.4357  # rounded to 4 decimals


# =====================================================================
# Visualization (basic checks)
# =====================================================================


class TestVisualization:
    def test_radar_chart(self, tmp_path):
        pytest.importorskip("matplotlib")
        from stylestream.eval.visualization import generate_radar_chart

        metrics = {
            "wer": 9.5,
            "s_sim": 0.85,
            "a_sim": 0.64,
            "e_sim": 0.82,
            "utmos": 4.1,
        }
        paper = {
            "wer": 9.2,
            "s_sim": 0.852,
            "a_sim": 0.640,
            "e_sim": 0.827,
            "utmos": 4.2,
        }
        out = tmp_path / "radar.png"
        generate_radar_chart(metrics, paper, out)
        assert out.exists()

    def test_radar_chart_no_paper(self, tmp_path):
        pytest.importorskip("matplotlib")
        from stylestream.eval.visualization import generate_radar_chart

        metrics = {
            "wer": 9.5,
            "s_sim": 0.85,
            "a_sim": 0.64,
            "e_sim": 0.82,
            "utmos": 4.1,
        }
        out = tmp_path / "radar_no_paper.png"
        generate_radar_chart(metrics, None, out)
        assert out.exists()

    def test_category_bars(self, tmp_path):
        pytest.importorskip("matplotlib")
        from stylestream.eval.visualization import generate_category_bars

        stats = {
            "emotion": {
                "wer": {"mean": 10.0, "std": 2.0},
                "s_sim": {"mean": 0.85, "std": 0.05},
            },
            "accent": {
                "wer": {"mean": 9.0, "std": 1.5},
                "s_sim": {"mean": 0.83, "std": 0.04},
            },
        }
        out = tmp_path / "bars.png"
        generate_category_bars(stats, out)
        assert out.exists()

    def test_heatmap(self, tmp_path):
        pytest.importorskip("matplotlib")
        from stylestream.eval.visualization import generate_comparison_heatmap

        ours = {"wer": 9.5, "s_sim": 0.85}
        paper = {"wer": 9.2, "s_sim": 0.852}
        out = tmp_path / "heatmap.png"
        generate_comparison_heatmap(ours, paper, out)
        assert out.exists()

    def test_html_report(self, tmp_path):
        summary = {
            "total_pairs": 100,
            "overall": {
                "wer": {
                    "mean": 9.5,
                    "std": 2.0,
                    "ci_95": [9.1, 9.9],
                    "count": 100,
                }
            },
            "by_category": {},
            "by_label": {},
        }
        json_path = tmp_path / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f)

        from stylestream.eval.visualization import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(json_path, out)
        assert out.exists()
        content = out.read_text()
        assert "StyleStream" in content

    def test_html_report_with_charts_dir(self, tmp_path):
        summary = {
            "total_pairs": 50,
            "overall": {
                "wer": {
                    "mean": 10.0,
                    "std": 1.0,
                    "ci_95": [9.5, 10.5],
                    "count": 50,
                }
            },
            "by_category": {},
            "by_label": {},
        }
        json_path = tmp_path / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f)

        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()

        from stylestream.eval.visualization import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(json_path, out, charts_dir)
        assert out.exists()


# =====================================================================
# Base evaluator context manager
# =====================================================================


class TestBaseEvaluatorContextManager:
    def test_context_manager(self):
        from stylestream.eval.base import BaseEvaluator, EvalResult

        class DummyEval(BaseEvaluator):
            @property
            def metric_name(self):
                return "dummy"

            @property
            def direction(self):
                return "higher_is_better"

            def _load_model(self):
                self._model = "loaded"

            def evaluate_pair(
                self, converted_audio, target_audio=None, source_text=None
            ):
                return EvalResult(
                    metric_name="dummy",
                    value=1.0,
                    direction="higher_is_better",
                )

        with DummyEval(device="cpu") as evaluator:
            assert evaluator.is_loaded
        assert not evaluator.is_loaded

    def test_lazy_loading(self):
        from stylestream.eval.base import BaseEvaluator, EvalResult

        class DummyEval(BaseEvaluator):
            @property
            def metric_name(self):
                return "dummy"

            @property
            def direction(self):
                return "higher_is_better"

            def _load_model(self):
                self._model = "loaded"

            def evaluate_pair(
                self, converted_audio, target_audio=None, source_text=None
            ):
                self._ensure_loaded()
                return EvalResult(
                    metric_name="dummy",
                    value=1.0,
                    direction="higher_is_better",
                )

        evaluator = DummyEval(device="cpu")
        assert not evaluator.is_loaded
        # Should lazy-load on evaluate
        result = evaluator.evaluate_pair(torch.randn(16000))
        assert evaluator.is_loaded

    def test_explicit_load_unload(self):
        from stylestream.eval.base import BaseEvaluator, EvalResult

        class DummyEval(BaseEvaluator):
            @property
            def metric_name(self):
                return "dummy"

            @property
            def direction(self):
                return "higher_is_better"

            def _load_model(self):
                self._model = "loaded"

            def evaluate_pair(
                self, converted_audio, target_audio=None, source_text=None
            ):
                return EvalResult(
                    metric_name="dummy",
                    value=1.0,
                    direction="higher_is_better",
                )

        evaluator = DummyEval(device="cpu")
        evaluator.load()
        assert evaluator.is_loaded
        evaluator.unload()
        assert not evaluator.is_loaded
        # Unload when not loaded should be a no-op
        evaluator.unload()
        assert not evaluator.is_loaded

    def test_load_idempotent(self):
        from stylestream.eval.base import BaseEvaluator, EvalResult

        call_count = 0

        class DummyEval(BaseEvaluator):
            @property
            def metric_name(self):
                return "dummy"

            @property
            def direction(self):
                return "higher_is_better"

            def _load_model(self):
                nonlocal call_count
                call_count += 1
                self._model = "loaded"

            def evaluate_pair(
                self, converted_audio, target_audio=None, source_text=None
            ):
                return EvalResult(
                    metric_name="dummy",
                    value=1.0,
                    direction="higher_is_better",
                )

        evaluator = DummyEval(device="cpu")
        evaluator.load()
        evaluator.load()  # second call should be no-op
        assert call_count == 1

    def test_evaluate_batch_default(self):
        from stylestream.eval.base import BaseEvaluator, EvalResult

        class DummyEval(BaseEvaluator):
            @property
            def metric_name(self):
                return "dummy"

            @property
            def direction(self):
                return "higher_is_better"

            def _load_model(self):
                self._model = "loaded"

            def evaluate_pair(
                self, converted_audio, target_audio=None, source_text=None
            ):
                self._ensure_loaded()
                return EvalResult(
                    metric_name="dummy",
                    value=float(converted_audio.sum()),
                    direction="higher_is_better",
                )

        evaluator = DummyEval(device="cpu")
        audios = [torch.randn(16000) for _ in range(3)]
        results = evaluator.evaluate_batch(audios)
        assert len(results) == 3
        assert all(r.metric_name == "dummy" for r in results)


# =====================================================================
# MetricStats
# =====================================================================


class TestMetricStats:
    def test_to_dict(self):
        from stylestream.eval.aggregator import MetricStats

        stats = MetricStats(
            name="wer",
            mean=9.5,
            std=2.0,
            ci_95_low=9.1,
            ci_95_high=9.9,
            count=100,
            min=5.0,
            max=15.0,
            median=9.3,
        )
        d = stats.to_dict()
        assert d["name"] == "wer"
        assert d["ci_95"] == [9.1, 9.9]

    def test_median_computation(self):
        from stylestream.eval.aggregator import compute_stats

        # Odd number of values
        stats = compute_stats([1.0, 2.0, 3.0], "test")
        assert stats.median == 2.0
        # Even number of values
        stats = compute_stats([1.0, 2.0, 3.0, 4.0], "test")
        assert stats.median == 2.5

    def test_to_dict_rounding(self):
        from stylestream.eval.aggregator import MetricStats

        stats = MetricStats(
            name="test",
            mean=9.12345,
            std=2.56789,
            ci_95_low=8.11111,
            ci_95_high=10.13333,
            count=50,
            min=5.98765,
            max=14.12345,
            median=9.05555,
        )
        d = stats.to_dict()
        # Verify values are rounded to 4 decimal places
        assert d["mean"] == 9.1235
        assert d["std"] == 2.5679

    def test_min_max_computation(self):
        from stylestream.eval.aggregator import compute_stats

        stats = compute_stats([5.0, 10.0, 15.0, 20.0], "test")
        assert stats.min == 5.0
        assert stats.max == 20.0

    def test_ci_width_decreases_with_samples(self):
        from stylestream.eval.aggregator import compute_stats

        stats_small = compute_stats([10.0, 12.0, 11.0, 13.0, 9.0], "test")
        large_values = [10.0, 12.0, 11.0, 13.0, 9.0] * 20
        stats_large = compute_stats(large_values, "test")
        ci_width_small = stats_small.ci_95_high - stats_small.ci_95_low
        ci_width_large = stats_large.ci_95_high - stats_large.ci_95_low
        assert ci_width_large < ci_width_small


# =====================================================================
# Batch inference file naming
# =====================================================================


class TestBatchInference:
    def test_output_path_format(self):
        from stylestream.eval.batch_inference import BatchInference

        bi = BatchInference(
            destylizer_checkpoint="",
            stylizer_checkpoint="",
            vocoder_checkpoint="",
            output_dir="/tmp/test_output",
        )
        path = bi._get_output_path("src_001", "emotion_happy")
        assert path == Path("/tmp/test_output/src_001__emotion_happy.wav")

    def test_resume_detection(self, tmp_path):
        from stylestream.eval.batch_inference import BatchInference

        bi = BatchInference(
            destylizer_checkpoint="",
            stylizer_checkpoint="",
            vocoder_checkpoint="",
            output_dir=str(tmp_path),
        )
        # No file exists yet
        assert not bi._is_already_converted("src_001", "tgt_001")
        # Create the file
        (tmp_path / "src_001__tgt_001.wav").touch()
        assert bi._is_already_converted("src_001", "tgt_001")

    def test_output_dir_stored(self):
        from stylestream.eval.batch_inference import BatchInference

        bi = BatchInference(
            destylizer_checkpoint="/ckpt/dest",
            stylizer_checkpoint="/ckpt/styl",
            vocoder_checkpoint="/ckpt/voc",
            output_dir="/tmp/output",
        )
        assert bi.output_dir == Path("/tmp/output")
        assert bi.destylizer_checkpoint == "/ckpt/dest"
        assert bi.stylizer_checkpoint == "/ckpt/styl"
        assert bi.vocoder_checkpoint == "/ckpt/voc"

    def test_default_parameters(self):
        from stylestream.eval.batch_inference import BatchInference

        bi = BatchInference(
            destylizer_checkpoint="",
            stylizer_checkpoint="",
            vocoder_checkpoint="",
            output_dir="/tmp/out",
        )
        assert bi.device == "cuda"
        assert bi.sample_rate == 16000
        assert bi.use_streaming is False
        assert bi.nfe == 16
        assert bi.cfg_strength == 2.0


# =====================================================================
# CLI argument parsing (evaluate.py)
# =====================================================================


class TestEvaluateCLI:
    def test_imports(self):
        """Test that evaluate.py can be imported."""
        import importlib

        # scripts directory doesn't have __init__.py, so import the file
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "evaluate",
            str(
                Path(__file__).resolve().parent.parent
                / "scripts"
                / "evaluate.py"
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
        assert hasattr(mod, "load_audio")
        assert hasattr(mod, "setup_logging")


# =====================================================================
# Module imports
# =====================================================================


class TestModuleImports:
    def test_eval_init(self):
        from stylestream.eval import (
            BaseEvaluator,
            EvalResult,
            PairResult,
            SimilarityEvaluator,
        )

        assert BaseEvaluator is not None
        assert SimilarityEvaluator is not None
        assert EvalResult is not None
        assert PairResult is not None

    def test_eval_registry(self):
        from stylestream.eval import available_metrics, get_evaluator

        assert callable(get_evaluator)
        metrics = available_metrics()
        assert len(metrics) >= 5

    def test_eval_dataset_import(self):
        from stylestream.eval.dataset import (
            EvalDataset,
            EvalPair,
            build_eval_pairs,
        )

        assert EvalDataset is not None
        assert EvalPair is not None
        assert callable(build_eval_pairs)

    def test_eval_aggregator_import(self):
        from stylestream.eval.aggregator import (
            MetricsAggregator,
            PairMetrics,
            compute_stats,
        )

        assert MetricsAggregator is not None
        assert PairMetrics is not None
        assert callable(compute_stats)

    def test_eval_probing_import(self):
        from stylestream.eval.probing import (
            LinearProbe,
            ProbingResult,
            StyleProbing,
        )

        assert StyleProbing is not None
        assert LinearProbe is not None
        assert ProbingResult is not None

    def test_eval_visualization_import(self):
        from stylestream.eval.visualization import (
            generate_category_bars,
            generate_comparison_heatmap,
            generate_html_report,
            generate_radar_chart,
        )

        assert callable(generate_radar_chart)
        assert callable(generate_category_bars)
        assert callable(generate_comparison_heatmap)
        assert callable(generate_html_report)

    def test_batch_inference_import(self):
        from stylestream.eval.batch_inference import BatchInference

        assert BatchInference is not None

    def test_whisper_evaluator_import(self):
        from stylestream.eval.whisper_evaluator import (
            WhisperEvaluator,
            normalize_text,
        )

        assert WhisperEvaluator is not None
        assert callable(normalize_text)

    def test_resemblyzer_evaluator_import(self):
        from stylestream.eval.resemblyzer_evaluator import (
            ResemblyzerEvaluator,
        )

        assert ResemblyzerEvaluator is not None

    def test_accent_evaluator_import(self):
        from stylestream.eval.accent_evaluator import AccentEvaluator

        assert AccentEvaluator is not None

    def test_emotion_evaluator_import(self):
        from stylestream.eval.emotion_evaluator import EmotionEvaluator

        assert EmotionEvaluator is not None

    def test_utmos_evaluator_import(self):
        from stylestream.eval.utmos_evaluator import UTMOSEvaluator

        assert UTMOSEvaluator is not None

    def test_eval_init_all_exports(self):
        import stylestream.eval as eval_mod

        for name in [
            "BaseEvaluator",
            "SimilarityEvaluator",
            "EvalResult",
            "PairResult",
            "BatchInference",
            "MetricsAggregator",
            "MetricStats",
            "PairMetrics",
            "compute_stats",
            "get_evaluator",
            "available_metrics",
        ]:
            assert hasattr(eval_mod, name), f"Missing export: {name}"


# =====================================================================
# PairMetrics (aggregator)
# =====================================================================


class TestPairMetrics:
    def test_creation(self):
        from stylestream.eval.aggregator import PairMetrics

        pm = PairMetrics(
            source_id="s1",
            target_id="t1",
            target_category="emotion",
            target_label="happy",
            source_dataset="esd",
            metrics={"wer": 9.5, "s_sim": 0.85},
        )
        assert pm.source_id == "s1"
        assert pm.metrics["wer"] == 9.5

    def test_defaults(self):
        from stylestream.eval.aggregator import PairMetrics

        pm = PairMetrics(source_id="s1", target_id="t1")
        assert pm.target_category == ""
        assert pm.target_label == ""
        assert pm.source_dataset == ""
        assert pm.metrics == {}
        assert pm.metadata == {}
