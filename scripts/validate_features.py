"""Validate preprocessed features for StyleStream training.

Performs comprehensive quality checks on resampled audio, mel spectrograms,
and HuBERT features to ensure consistency before training.

Checks (from Phase 1 milestone M4.1):
  - 50 Hz synchronization between mel and HuBERT features
  - No NaN / Inf values in any features
  - Correct shapes (mel: 100 x T, HuBERT: 768 x T)
  - Value ranges are reasonable
  - Duration statistics match expectations

Usage:
    python scripts/validate_features.py --manifest data/manifests/libritts.csv --processed-dir data/processed
    python scripts/validate_features.py --manifest data/manifests/lmg.csv --processed-dir data/processed --check-hubert
    python scripts/validate_features.py --manifest data/manifests/lmg.csv --processed-dir data/processed --sample-pct 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import torch

from stylestream.data.manifest import Manifest, Utterance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching project specs (config.py, mel.py, paper)
# ---------------------------------------------------------------------------

EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_MEL_BINS = 100
EXPECTED_HUBERT_DIM = 768
HOP_LENGTH = 320  # 16000 / 320 = 50 Hz
FRAME_RATE = EXPECTED_SAMPLE_RATE / HOP_LENGTH  # 50.0

# Log-mel typical range (log(mel + 1e-5) with Vocos convention)
MEL_VALUE_MIN = -12.0
MEL_VALUE_MAX = 8.0

# Maximum tolerable frame difference for 50 Hz sync
MAX_SYNC_DRIFT = 1

# Cap the number of per-file error messages stored in a report section
MAX_ERROR_MESSAGES = 20


def _capped_append(errors: list[str], msg: str) -> None:
    """Append *msg* to *errors* only if the cap has not been reached."""
    if len(errors) < MAX_ERROR_MESSAGES:
        errors.append(msg)


# ======================================================================
# FeatureValidator
# ======================================================================


class FeatureValidator:
    """Validates preprocessed features for training readiness."""

    def __init__(
        self,
        manifest: Manifest,
        processed_dir: str | Path,
        sample_pct: float = 1.0,
        hop_length: int = HOP_LENGTH,
        sample_rate: int = EXPECTED_SAMPLE_RATE,
    ):
        self.manifest = manifest
        self.processed_dir = Path(processed_dir)
        self.sample_pct = max(0.0, min(1.0, sample_pct))
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Path helpers (mirror PreprocessingPipeline layout)
    # ------------------------------------------------------------------

    def _resampled_path(self, utt: Utterance) -> Path:
        """``processed_dir/16k/{dataset}/{subset}/{stem}.wav``"""
        return (
            self.processed_dir / "16k" / utt.dataset / utt.subset / f"{utt.stem}.wav"
        )

    def _mel_path(self, utt: Utterance) -> Path:
        """``processed_dir/mel/{dataset}/{subset}/{stem}.pt``"""
        return (
            self.processed_dir / "mel" / utt.dataset / utt.subset / f"{utt.stem}.pt"
        )

    def _hubert_path(self, utt: Utterance) -> Path:
        """``processed_dir/hubert/{dataset}/{subset}/{stem}.pt``"""
        return (
            self.processed_dir
            / "hubert"
            / utt.dataset
            / utt.subset
            / f"{utt.stem}.pt"
        )

    # ------------------------------------------------------------------
    # Sampling helper
    # ------------------------------------------------------------------

    def _sampled_utterances(self) -> list[Utterance]:
        """Return utterances to check, respecting *sample_pct*."""
        all_utts = list(self.manifest)
        if self.sample_pct >= 1.0:
            return all_utts
        k = max(1, int(len(all_utts) * self.sample_pct))
        return random.sample(all_utts, k)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_audio(self) -> dict:
        """Check resampled audio files.

        Verifies: file exists, loadable, 16 kHz, mono, no NaN, duration
        within a reasonable range and broadly consistent with manifest
        metadata.
        """
        import soundfile as sf  # lazy so the module is importable without sf

        utts = self._sampled_utterances()
        total = len(utts)
        passed = 0
        failed = 0
        errors: list[str] = []

        durations: list[float] = []
        amplitudes: list[float] = []

        for utt in utts:
            audio_path = self._resampled_path(utt)
            stem = utt.stem

            if not audio_path.exists():
                failed += 1
                _capped_append(errors, f"{stem}: audio file missing at {audio_path}")
                continue

            try:
                info = sf.info(str(audio_path))
            except Exception as exc:
                failed += 1
                _capped_append(errors, f"{stem}: cannot read info -- {exc}")
                continue

            file_ok = True

            # Sample rate
            if info.samplerate != self.sample_rate:
                file_ok = False
                _capped_append(
                    errors,
                    f"{stem}: sample_rate={info.samplerate}, expected {self.sample_rate}",
                )

            # Channels (mono)
            if info.channels != 1:
                file_ok = False
                _capped_append(
                    errors,
                    f"{stem}: channels={info.channels}, expected 1 (mono)",
                )

            # Duration sanity (non-zero, not absurdly long)
            dur = info.frames / info.samplerate
            if dur <= 0.0:
                file_ok = False
                _capped_append(errors, f"{stem}: duration is {dur:.4f}s (non-positive)")
            elif dur > 300.0:
                file_ok = False
                _capped_append(errors, f"{stem}: duration {dur:.1f}s exceeds 300s limit")

            # Spot-check waveform values (NaN)
            try:
                data, _ = sf.read(str(audio_path), dtype="float32")
                wav_tensor = torch.from_numpy(data)
                if torch.isnan(wav_tensor).any():
                    file_ok = False
                    _capped_append(errors, f"{stem}: waveform contains NaN")
                if torch.isinf(wav_tensor).any():
                    file_ok = False
                    _capped_append(errors, f"{stem}: waveform contains Inf")
                amplitudes.append(wav_tensor.abs().max().item())
            except Exception as exc:
                file_ok = False
                _capped_append(errors, f"{stem}: cannot read waveform -- {exc}")

            if file_ok:
                passed += 1
                durations.append(dur)
            else:
                failed += 1

        stats: dict = {}
        if durations:
            dur_t = torch.tensor(durations)
            stats = {
                "duration_mean": dur_t.mean().item(),
                "duration_std": dur_t.std().item(),
                "duration_min": dur_t.min().item(),
                "duration_max": dur_t.max().item(),
                "total_hours": dur_t.sum().item() / 3600.0,
            }
        if amplitudes:
            amp_t = torch.tensor(amplitudes)
            stats["amplitude_max_mean"] = amp_t.mean().item()
            stats["amplitude_max_max"] = amp_t.max().item()

        return {
            "check": "audio",
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "stats": stats,
        }

    def check_mel(self) -> dict:
        """Check mel spectrograms.

        Verifies: file exists, loadable, shape (100, T),
        T = ceil(audio_samples / 320), no NaN/Inf, value range reasonable
        (log-mel typically -12 to 8).
        """
        import soundfile as sf

        utts = self._sampled_utterances()
        total = len(utts)
        passed = 0
        failed = 0
        errors: list[str] = []

        frame_counts: list[int] = []
        value_mins: list[float] = []
        value_maxs: list[float] = []
        value_means: list[float] = []

        for utt in utts:
            mel_path = self._mel_path(utt)
            stem = utt.stem

            if not mel_path.exists():
                failed += 1
                _capped_append(errors, f"{stem}: mel file missing at {mel_path}")
                continue

            try:
                mel = torch.load(mel_path, weights_only=True, map_location="cpu")
            except Exception as exc:
                failed += 1
                _capped_append(errors, f"{stem}: cannot load mel -- {exc}")
                continue

            file_ok = True

            # Tensor check
            if not isinstance(mel, torch.Tensor):
                failed += 1
                _capped_append(errors, f"{stem}: mel is not a Tensor ({type(mel)})")
                continue

            # Dimension check
            if mel.dim() != 2:
                file_ok = False
                _capped_append(
                    errors,
                    f"{stem}: mel.dim()={mel.dim()}, expected 2",
                )

            # n_mels check
            if mel.dim() == 2 and mel.shape[0] != EXPECTED_MEL_BINS:
                file_ok = False
                _capped_append(
                    errors,
                    f"{stem}: mel shape[0]={mel.shape[0]}, expected {EXPECTED_MEL_BINS}",
                )

            # T vs expected from audio
            if mel.dim() == 2:
                T_mel = mel.shape[1]
                audio_path = self._resampled_path(utt)
                if audio_path.exists():
                    try:
                        info = sf.info(str(audio_path))
                        expected_T = math.ceil(info.frames / self.hop_length)
                        drift = abs(T_mel - expected_T)
                        if drift > MAX_SYNC_DRIFT:
                            file_ok = False
                            _capped_append(
                                errors,
                                f"{stem}: mel T={T_mel}, expected ~{expected_T} "
                                f"(drift={drift} > {MAX_SYNC_DRIFT})",
                            )
                    except Exception:
                        pass  # audio check handles this

                frame_counts.append(T_mel)

            # NaN / Inf
            if torch.isnan(mel).any():
                file_ok = False
                nan_count = torch.isnan(mel).sum().item()
                _capped_append(errors, f"{stem}: mel contains {nan_count} NaN values")

            if torch.isinf(mel).any():
                file_ok = False
                inf_count = torch.isinf(mel).sum().item()
                _capped_append(errors, f"{stem}: mel contains {inf_count} Inf values")

            # Value range
            v_min = mel.min().item()
            v_max = mel.max().item()
            v_mean = mel.mean().item()

            if v_min < MEL_VALUE_MIN or v_max > MEL_VALUE_MAX:
                file_ok = False
                _capped_append(
                    errors,
                    f"{stem}: mel value range [{v_min:.2f}, {v_max:.2f}] "
                    f"outside expected [{MEL_VALUE_MIN}, {MEL_VALUE_MAX}]",
                )

            value_mins.append(v_min)
            value_maxs.append(v_max)
            value_means.append(v_mean)

            if file_ok:
                passed += 1
            else:
                failed += 1

        stats: dict = {}
        if frame_counts:
            fc = torch.tensor(frame_counts, dtype=torch.float32)
            stats["frame_count_mean"] = fc.mean().item()
            stats["frame_count_std"] = fc.std().item()
            stats["frame_count_min"] = int(fc.min().item())
            stats["frame_count_max"] = int(fc.max().item())
        if value_mins:
            stats["value_min"] = min(value_mins)
            stats["value_max"] = max(value_maxs)
            stats["value_mean"] = sum(value_means) / len(value_means)

        return {
            "check": "mel",
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "stats": stats,
        }

    def check_hubert(self) -> dict:
        """Check HuBERT features.

        Verifies: file exists, loadable, shape (768, T), no NaN/Inf,
        50 Hz sync with mel (T difference <= 1 frame).
        """
        utts = self._sampled_utterances()
        total = len(utts)
        passed = 0
        failed = 0
        errors: list[str] = []

        frame_counts: list[int] = []
        value_mins: list[float] = []
        value_maxs: list[float] = []
        value_means: list[float] = []
        value_stds: list[float] = []

        for utt in utts:
            hubert_path = self._hubert_path(utt)
            stem = utt.stem

            if not hubert_path.exists():
                failed += 1
                _capped_append(
                    errors, f"{stem}: hubert file missing at {hubert_path}"
                )
                continue

            try:
                feat = torch.load(hubert_path, weights_only=True, map_location="cpu")
            except Exception as exc:
                failed += 1
                _capped_append(errors, f"{stem}: cannot load hubert -- {exc}")
                continue

            file_ok = True

            if not isinstance(feat, torch.Tensor):
                failed += 1
                _capped_append(
                    errors, f"{stem}: hubert is not a Tensor ({type(feat)})"
                )
                continue

            # Dimension
            if feat.dim() != 2:
                file_ok = False
                _capped_append(
                    errors,
                    f"{stem}: hubert.dim()={feat.dim()}, expected 2",
                )

            # Feature dimension
            if feat.dim() == 2 and feat.shape[0] != EXPECTED_HUBERT_DIM:
                file_ok = False
                _capped_append(
                    errors,
                    f"{stem}: hubert shape[0]={feat.shape[0]}, expected {EXPECTED_HUBERT_DIM}",
                )

            if feat.dim() == 2:
                frame_counts.append(feat.shape[1])

            # NaN / Inf
            if torch.isnan(feat).any():
                file_ok = False
                nan_count = torch.isnan(feat).sum().item()
                _capped_append(
                    errors, f"{stem}: hubert contains {nan_count} NaN values"
                )

            if torch.isinf(feat).any():
                file_ok = False
                inf_count = torch.isinf(feat).sum().item()
                _capped_append(
                    errors, f"{stem}: hubert contains {inf_count} Inf values"
                )

            # Value statistics
            v_min = feat.min().item()
            v_max = feat.max().item()
            v_mean = feat.mean().item()
            v_std = feat.std().item()

            value_mins.append(v_min)
            value_maxs.append(v_max)
            value_means.append(v_mean)
            value_stds.append(v_std)

            if file_ok:
                passed += 1
            else:
                failed += 1

        stats: dict = {}
        if frame_counts:
            fc = torch.tensor(frame_counts, dtype=torch.float32)
            stats["frame_count_mean"] = fc.mean().item()
            stats["frame_count_std"] = fc.std().item()
            stats["frame_count_min"] = int(fc.min().item())
            stats["frame_count_max"] = int(fc.max().item())
        if value_mins:
            stats["value_min"] = min(value_mins)
            stats["value_max"] = max(value_maxs)
            stats["value_mean"] = sum(value_means) / len(value_means)
            stats["value_std_mean"] = sum(value_stds) / len(value_stds)

        return {
            "check": "hubert",
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "stats": stats,
        }

    def check_sync(self) -> dict:
        """Check 50 Hz synchronization between mel and HuBERT.

        For each file, verify: ``|mel_T - hubert_T| <= 1``.
        This is THE most critical check -- a desynchronised pair will
        silently corrupt training.
        """
        utts = self._sampled_utterances()
        total = 0
        passed = 0
        failed = 0
        errors: list[str] = []
        drifts: list[int] = []

        for utt in utts:
            mel_path = self._mel_path(utt)
            hubert_path = self._hubert_path(utt)
            stem = utt.stem

            # Both files must exist for a sync check to be meaningful
            if not mel_path.exists() or not hubert_path.exists():
                continue

            try:
                mel = torch.load(mel_path, weights_only=True, map_location="cpu")
                feat = torch.load(hubert_path, weights_only=True, map_location="cpu")
            except Exception as exc:
                failed += 1
                total += 1
                _capped_append(errors, f"{stem}: load error during sync check -- {exc}")
                continue

            if not isinstance(mel, torch.Tensor) or not isinstance(feat, torch.Tensor):
                continue
            if mel.dim() != 2 or feat.dim() != 2:
                continue

            total += 1
            T_mel = mel.shape[1]
            T_hub = feat.shape[1]
            drift = abs(T_mel - T_hub)
            drifts.append(drift)

            if drift > MAX_SYNC_DRIFT:
                failed += 1
                _capped_append(
                    errors,
                    f"{stem}: mel_T={T_mel}, hubert_T={T_hub}, drift={drift} "
                    f"(max allowed {MAX_SYNC_DRIFT})",
                )
            else:
                passed += 1

        stats: dict = {}
        if drifts:
            d = torch.tensor(drifts, dtype=torch.float32)
            stats["drift_mean"] = d.mean().item()
            stats["drift_max"] = int(d.max().item())
            stats["drift_zero_pct"] = (d == 0).float().mean().item() * 100.0
            stats["drift_le1_pct"] = (d <= 1).float().mean().item() * 100.0

        return {
            "check": "sync",
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "stats": stats,
        }

    def check_text(self) -> dict:
        """Check that text transcriptions are present and tokenizable.

        Uses :class:`stylestream.data.text.CharTokenizer` to verify that
        each utterance's text is non-empty and produces at least one token.
        """
        from stylestream.data.text import CharTokenizer

        tokenizer = CharTokenizer()

        utts = self._sampled_utterances()
        total = len(utts)
        passed = 0
        failed = 0
        errors: list[str] = []

        token_lengths: list[int] = []
        empty_text_count = 0

        for utt in utts:
            stem = utt.stem
            text = utt.text

            if not text or not text.strip():
                failed += 1
                empty_text_count += 1
                _capped_append(errors, f"{stem}: text field is empty")
                continue

            try:
                ids = tokenizer.encode(text)
            except Exception as exc:
                failed += 1
                _capped_append(errors, f"{stem}: tokenization error -- {exc}")
                continue

            if len(ids) == 0:
                failed += 1
                _capped_append(
                    errors,
                    f"{stem}: text '{text[:60]}' produces 0 tokens after normalization",
                )
                continue

            passed += 1
            token_lengths.append(len(ids))

        stats: dict = {"empty_text_count": empty_text_count}
        if token_lengths:
            tl = torch.tensor(token_lengths, dtype=torch.float32)
            stats["token_length_mean"] = tl.mean().item()
            stats["token_length_std"] = tl.std().item()
            stats["token_length_min"] = int(tl.min().item())
            stats["token_length_max"] = int(tl.max().item())

        return {
            "check": "text",
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    # Aggregated run
    # ------------------------------------------------------------------

    def run_all(self, check_hubert: bool = False) -> dict:
        """Run all checks and return a comprehensive report.

        Parameters
        ----------
        check_hubert : bool
            If *True*, also run HuBERT feature checks and mel/HuBERT
            synchronization checks.  These are skipped by default
            because HuBERT extraction is a later pipeline stage.

        Returns
        -------
        dict
            Top-level report with per-check sections and a global
            ``has_errors`` flag.
        """
        logger.info(
            "Starting feature validation: %d utterances, sample_pct=%.2f, "
            "check_hubert=%s",
            len(self.manifest),
            self.sample_pct,
            check_hubert,
        )

        report: dict = {
            "manifest_size": len(self.manifest),
            "sample_pct": self.sample_pct,
            "processed_dir": str(self.processed_dir),
            "checks": {},
            "has_errors": False,
        }

        # Always run these three
        for name, fn in [
            ("audio", self.check_audio),
            ("mel", self.check_mel),
            ("text", self.check_text),
        ]:
            logger.info("Running check: %s ...", name)
            result = fn()
            report["checks"][name] = result
            if result["failed"] > 0:
                report["has_errors"] = True
            logger.info(
                "  %s: %d/%d passed, %d failed",
                name,
                result["passed"],
                result["total"],
                result["failed"],
            )

        # HuBERT-dependent checks
        if check_hubert:
            for name, fn in [
                ("hubert", self.check_hubert),
                ("sync", self.check_sync),
            ]:
                logger.info("Running check: %s ...", name)
                result = fn()
                report["checks"][name] = result
                if result["failed"] > 0:
                    report["has_errors"] = True
                logger.info(
                    "  %s: %d/%d passed, %d failed",
                    name,
                    result["passed"],
                    result["total"],
                    result["failed"],
                )

        status = "FAIL" if report["has_errors"] else "PASS"
        logger.info("Validation complete: %s", status)
        return report

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_report(self, report: dict) -> None:
        """Print a formatted validation report to stdout."""
        print()
        print("=" * 70)
        print("  StyleStream Feature Validation Report")
        print("=" * 70)
        print(f"  Manifest size   : {report['manifest_size']}")
        print(f"  Sample %        : {report['sample_pct'] * 100:.1f}%")
        print(f"  Processed dir   : {report['processed_dir']}")
        print()

        for name, section in report.get("checks", {}).items():
            total = section["total"]
            passed = section["passed"]
            failed = section["failed"]
            status = "PASS" if failed == 0 else "FAIL"

            print(f"--- {name.upper()} [{status}] ---")
            print(f"  Total : {total}")
            print(f"  Passed: {passed}")
            print(f"  Failed: {failed}")

            # Stats
            stats = section.get("stats", {})
            if stats:
                print("  Stats:")
                for k, v in stats.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")

            # Errors (capped)
            errs = section.get("errors", [])
            if errs:
                print(f"  Errors (showing up to {MAX_ERROR_MESSAGES}):")
                for e in errs:
                    print(f"    - {e}")
            print()

        overall = "FAIL" if report.get("has_errors", False) else "PASS"
        print(f"Overall: {overall}")
        print("=" * 70)
        print()


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate preprocessed features for StyleStream training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/validate_features.py "
            "--manifest data/manifests/libritts.csv --processed-dir data/processed\n"
            "  python scripts/validate_features.py "
            "--manifest data/manifests/lmg.csv --processed-dir data/processed "
            "--check-hubert\n"
            "  python scripts/validate_features.py "
            "--manifest data/manifests/lmg.csv --processed-dir data/processed "
            "--sample-pct 0.01\n"
        ),
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest CSV (created by preprocessing pipeline).",
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Root directory containing processed features (16k/, mel/, hubert/).",
    )
    parser.add_argument(
        "--check-hubert",
        action="store_true",
        help="Also validate HuBERT features and mel/HuBERT sync.",
    )
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=1.0,
        help=(
            "Fraction of manifest entries to check (0.0 - 1.0). "
            "Use < 1.0 for large datasets like Emilia-EN. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If set, save the JSON report to this path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reproducibility
    random.seed(args.seed)

    # Load manifest
    manifest = Manifest.load(args.manifest)
    logger.info("Loaded manifest: %d utterances from %s", len(manifest), args.manifest)

    # Run validation
    validator = FeatureValidator(
        manifest=manifest,
        processed_dir=args.processed_dir,
        sample_pct=args.sample_pct,
    )
    report = validator.run_all(check_hubert=args.check_hubert)
    validator.print_report(report)

    # Optionally persist report as JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", output_path)

    # Exit with error code if any check failed
    if report.get("has_errors", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
