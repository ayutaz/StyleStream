"""Batch audio preprocessing pipeline for StyleStream training.

Three stages executed in order:

1. **Resample** -- convert all audio to 16 kHz mono WAV
2. **Mel** -- compute 100-bin log-mel spectrograms and save as ``.pt`` files
3. **Features** -- (delegated to HuBERTExtractor for GPU processing, not here)

Both resampling and mel computation are CPU-bound and parallelised via
:class:`concurrent.futures.ProcessPoolExecutor`.

Directory layout produced::

    output_dir/
      16k/{dataset}/{subset}/{stem}.wav      # resampled audio
      mel/{dataset}/{subset}/{stem}.pt        # mel tensor (100, T), float16
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from stylestream.data.manifest import Manifest, Utterance

logger = logging.getLogger(__name__)


# ======================================================================
# Top-level worker functions (must be importable at module level for
# multiprocessing / pickle).
# ======================================================================


def _resample_single(args: tuple) -> dict:
    """Worker function for parallel resampling.

    Parameters
    ----------
    args : tuple
        ``(input_path, output_path, target_sr, skip_existing)``

    Returns
    -------
    dict
        ``{"input_path": ..., "output_path": ..., "status": "ok"|"skipped"|"error",
           "error": str | None, "duration": float}``
    """
    # Lazy imports so each worker process gets its own copies and avoids
    # serialising heavy torch state from the parent.
    from stylestream.utils.audio import load_audio, save_audio  # noqa: C0415

    input_path, output_path, target_sr, skip_existing = args
    input_path = Path(input_path)
    output_path = Path(output_path)

    result: dict = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "ok",
        "error": None,
        "duration": 0.0,
    }

    try:
        if skip_existing and output_path.exists():
            result["status"] = "skipped"
            return result

        waveform = load_audio(input_path, sr=target_sr)
        result["duration"] = waveform.shape[0] / target_sr
        save_audio(output_path, waveform, sr=target_sr)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _resample_and_mel_single(args: tuple) -> dict:
    """Combined resample + mel in single pass. Eliminates intermediate WAV I/O.

    Parameters
    ----------
    args : tuple
        ``(input_path, mel_output_path, wav_output_path, target_sr,
          skip_existing, keep_resampled)``

    Returns
    -------
    dict
        ``{"input_path": ..., "mel_output_path": ..., "wav_output_path": ...,
           "status": "ok"|"skipped"|"error", "error": str | None,
           "duration": float, "shape": tuple | None}``
    """
    from stylestream.utils.audio import load_audio, save_audio  # noqa: C0415
    from stylestream.utils.mel import MelSpectrogramTransform  # noqa: C0415

    input_path, mel_output_path, wav_output_path, target_sr, skip_existing, keep_resampled = args
    input_path = Path(input_path)
    mel_output_path = Path(mel_output_path)
    wav_output_path = Path(wav_output_path) if wav_output_path is not None else None

    result: dict = {
        "input_path": str(input_path),
        "mel_output_path": str(mel_output_path),
        "wav_output_path": str(wav_output_path) if wav_output_path is not None else None,
        "status": "ok",
        "error": None,
        "duration": 0.0,
        "shape": None,
    }

    try:
        # Determine what we can skip
        mel_exists = skip_existing and mel_output_path.exists()
        wav_exists = (
            skip_existing
            and wav_output_path is not None
            and wav_output_path.exists()
        )
        mel_needed = not mel_exists
        wav_needed = keep_resampled and not wav_exists

        if not mel_needed and not wav_needed:
            result["status"] = "skipped"
            return result

        # Load and resample to target_sr in memory -- single I/O read
        waveform = load_audio(input_path, sr=target_sr)
        result["duration"] = waveform.shape[0] / target_sr

        # Compute mel spectrogram in memory (no intermediate disk write)
        if mel_needed:
            transform = MelSpectrogramTransform()
            mel = transform(waveform.unsqueeze(0)).squeeze(0)  # (100, T)
            mel_output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mel.to(torch.float16), mel_output_path)
            result["shape"] = tuple(mel.shape)

        # Optionally save the resampled WAV
        if wav_needed:
            save_audio(wav_output_path, waveform, sr=target_sr)

    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _compute_mel_single(args: tuple) -> dict:
    """Worker function for parallel mel spectrogram computation.

    Parameters
    ----------
    args : tuple
        ``(audio_path, output_path, skip_existing)``

    Returns
    -------
    dict
        ``{"audio_path": ..., "output_path": ..., "status": "ok"|"skipped"|"error",
           "error": str | None, "shape": tuple | None}``
    """
    from stylestream.utils.audio import load_audio  # noqa: C0415
    from stylestream.utils.mel import MelSpectrogramTransform  # noqa: C0415

    audio_path, output_path, skip_existing = args
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    result: dict = {
        "audio_path": str(audio_path),
        "output_path": str(output_path),
        "status": "ok",
        "error": None,
        "shape": None,
    }

    try:
        if skip_existing and output_path.exists():
            result["status"] = "skipped"
            return result

        waveform = load_audio(audio_path, sr=16000)  # already 16k after stage 1
        transform = MelSpectrogramTransform()
        # forward expects (batch, samples), returns (batch, 100, T)
        mel = transform(waveform.unsqueeze(0)).squeeze(0)  # (100, T)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as float16 to halve disk usage and I/O time.
        # Consumers cast back to float32 on load.
        torch.save(mel.to(torch.float16), output_path)
        result["shape"] = tuple(mel.shape)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ======================================================================
# Main pipeline class
# ======================================================================


class PreprocessingPipeline:
    """Batch audio preprocessing for all datasets.

    Three stages:

    1. **Resample** -- convert all audio to 16 kHz mono WAV.
    2. **Mel** -- compute and save mel spectrograms as ``.pt`` files.
    3. **Features** -- (delegated to HuBERTExtractor for GPU processing).

    Parameters
    ----------
    manifest : Manifest
        Input manifest listing all utterances to process.
    output_dir : str | Path
        Root directory for processed outputs.
    sample_rate : int
        Target sample rate (default 16000).
    num_workers : int
        Number of parallel workers for ProcessPoolExecutor.
    """

    def __init__(
        self,
        manifest: Manifest,
        output_dir: str | Path,
        sample_rate: int = 16000,
        num_workers: int = 8,
    ) -> None:
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.num_workers = num_workers

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_resampled_path(self, utterance: Utterance) -> Path:
        """Return the path where resampled audio would be saved.

        Layout: ``output_dir/16k/{dataset}/{subset}/{stem}.wav``
        """
        return (
            self.output_dir
            / "16k"
            / utterance.dataset
            / utterance.subset
            / f"{utterance.stem}.wav"
        )

    def get_mel_path(self, utterance: Utterance) -> Path:
        """Return the path where the mel spectrogram would be saved.

        Layout: ``output_dir/mel/{dataset}/{subset}/{stem}.pt``
        """
        return (
            self.output_dir
            / "mel"
            / utterance.dataset
            / utterance.subset
            / f"{utterance.stem}.pt"
        )

    # ------------------------------------------------------------------
    # Stage 1: Resample
    # ------------------------------------------------------------------

    def run_resample(self, skip_existing: bool = True) -> Manifest:
        """Resample all audio to 16 kHz mono WAV.

        Saves to: ``output_dir/16k/{dataset}/{subset}/{stem}.wav``

        Returns a **new** :class:`Manifest` whose ``audio_path`` fields
        point to the resampled files and ``sample_rate`` is updated.

        Uses :class:`ProcessPoolExecutor` for parallelism.
        """
        logger.info(
            "Stage 1: Resampling %d utterances to %d Hz (workers=%d)",
            len(self.manifest),
            self.sample_rate,
            self.num_workers,
        )
        t0 = time.time()

        # Build task list
        tasks: list[tuple] = []
        for utt in self.manifest:
            out_path = self.get_resampled_path(utt)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((utt.audio_path, str(out_path), self.sample_rate, skip_existing))

        # Execute in parallel
        results = self._run_parallel(_resample_single, tasks, stage_name="resample")

        # Build new manifest
        new_utterances: list[Utterance] = []
        for utt, res in zip(self.manifest, results):
            if res["status"] == "error":
                logger.warning("Resample failed for %s: %s", utt.audio_path, res["error"])
                continue
            new_utt = Utterance(
                audio_path=str(self.get_resampled_path(utt)),
                dataset=utt.dataset,
                subset=utt.subset,
                speaker_id=utt.speaker_id,
                duration=res.get("duration", utt.duration),
                sample_rate=self.sample_rate,
                text=utt.text,
            )
            new_utterances.append(new_utt)

        elapsed = time.time() - t0
        logger.info(
            "Resampling complete: %d/%d succeeded in %.1fs",
            len(new_utterances),
            len(self.manifest),
            elapsed,
        )
        return Manifest(utterances=new_utterances)

    # ------------------------------------------------------------------
    # Stage 2: Mel spectrograms
    # ------------------------------------------------------------------

    def run_mel(
        self, input_manifest: Manifest | None = None, skip_existing: bool = True
    ) -> None:
        """Compute mel spectrograms for all audio.

        Saves to: ``output_dir/mel/{dataset}/{subset}/{stem}.pt``

        Each ``.pt`` file contains a float16 tensor of shape ``(100, T)``.

        Parameters
        ----------
        input_manifest : Manifest | None
            If provided, use this manifest (typically the output of
            :meth:`run_resample`) instead of ``self.manifest``.
        skip_existing : bool
            Skip files that already exist on disk.

        Notes
        -----
        Uses :class:`ProcessPoolExecutor`.  Mel computation on CPU is
        fast so the bottleneck is typically audio I/O.
        """
        manifest = input_manifest if input_manifest is not None else self.manifest
        logger.info(
            "Stage 2: Computing mel spectrograms for %d utterances (workers=%d)",
            len(manifest),
            self.num_workers,
        )
        t0 = time.time()

        tasks: list[tuple] = []
        for utt in manifest:
            mel_path = self.get_mel_path(utt)
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((utt.audio_path, str(mel_path), skip_existing))

        results = self._run_parallel(_compute_mel_single, tasks, stage_name="mel")

        ok = sum(1 for r in results if r["status"] == "ok")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errors = sum(1 for r in results if r["status"] == "error")
        for r in results:
            if r["status"] == "error":
                logger.warning("Mel failed for %s: %s", r["audio_path"], r["error"])

        elapsed = time.time() - t0
        logger.info(
            "Mel computation complete: ok=%d, skipped=%d, errors=%d in %.1fs",
            ok,
            skipped,
            errors,
            elapsed,
        )

    # ------------------------------------------------------------------
    # Combined stage: Resample + Mel
    # ------------------------------------------------------------------

    def run_resample_and_mel(
        self, skip_existing: bool = True, keep_resampled: bool = False
    ) -> Manifest:
        """Combined resample + mel in single pass. Saves ~80% I/O by skipping intermediate WAV.

        Loads each original audio file once, resamples to 16 kHz in memory,
        computes the mel spectrogram, and saves only the mel ``.pt`` file.
        The resampled WAV is written only when *keep_resampled* is True.

        Parameters
        ----------
        skip_existing : bool
            Skip output files that already exist on disk.
        keep_resampled : bool
            If True, also save the resampled 16 kHz WAV file alongside the
            mel spectrogram. Default False (mel-only, maximum I/O savings).

        Returns
        -------
        Manifest
            A new manifest whose ``audio_path`` fields point to the
            resampled WAV locations (whether or not they were actually
            written). The ``sample_rate`` is set to the target rate.
        """
        logger.info(
            "Combined resample+mel: %d utterances to %d Hz (workers=%d, keep_wav=%s)",
            len(self.manifest),
            self.sample_rate,
            self.num_workers,
            keep_resampled,
        )
        t0 = time.time()

        # Build task list
        tasks: list[tuple] = []
        for utt in self.manifest:
            mel_path = self.get_mel_path(utt)
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path = self.get_resampled_path(utt)
            if keep_resampled:
                wav_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((
                utt.audio_path,
                str(mel_path),
                str(wav_path),
                self.sample_rate,
                skip_existing,
                keep_resampled,
            ))

        # Execute in parallel
        results = self._run_parallel(
            _resample_and_mel_single, tasks, stage_name="resample+mel"
        )

        # Build new manifest and collect stats
        new_utterances: list[Utterance] = []
        ok = 0
        skipped = 0
        errors = 0
        for utt, res in zip(self.manifest, results):
            if res["status"] == "error":
                errors += 1
                logger.warning(
                    "Resample+mel failed for %s: %s",
                    utt.audio_path,
                    res["error"],
                )
                continue
            if res["status"] == "skipped":
                skipped += 1
            else:
                ok += 1

            new_utt = Utterance(
                audio_path=str(self.get_resampled_path(utt)),
                dataset=utt.dataset,
                subset=utt.subset,
                speaker_id=utt.speaker_id,
                duration=res.get("duration", utt.duration),
                sample_rate=self.sample_rate,
                text=utt.text,
            )
            new_utterances.append(new_utt)

        elapsed = time.time() - t0
        logger.info(
            "Resample+mel complete: ok=%d, skipped=%d, errors=%d in %.1fs",
            ok,
            skipped,
            errors,
            elapsed,
        )
        return Manifest(utterances=new_utterances)

    # ------------------------------------------------------------------
    # Run all stages
    # ------------------------------------------------------------------

    def run_all(
        self,
        skip_existing: bool = True,
        keep_resampled: bool = True,
    ) -> Manifest:
        """Run all preprocessing stages. Uses combined resample+mel for efficiency.

        Returns the resampled manifest (with updated audio paths).

        Parameters
        ----------
        skip_existing : bool
            Skip output files that already exist on disk.
        keep_resampled : bool
            If True (default), save the resampled 16 kHz WAV alongside the
            mel spectrogram.  Set to False for maximum I/O savings when only
            the mel is needed downstream.

        .. note::

           HuBERT feature extraction (stage 3) is delegated to
           :class:`~stylestream.data.hubert_extractor.HuBERTExtractor` which
           requires GPU and is run separately.
        """
        resampled_manifest = self.run_resample_and_mel(
            skip_existing=skip_existing,
            keep_resampled=keep_resampled,
        )
        return resampled_manifest

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, manifest: Manifest, check_mel: bool = True) -> dict:
        """Verify that all processed files exist and are valid.

        Parameters
        ----------
        manifest : Manifest
            The manifest to verify (typically the output of
            :meth:`run_resample` or :meth:`run_all`).
        check_mel : bool
            Also verify mel spectrogram ``.pt`` files.

        Returns
        -------
        dict
            Statistics dictionary with keys:

            - ``total`` -- number of utterances checked
            - ``audio_ok`` -- number of valid resampled audio files
            - ``audio_missing`` -- number of missing audio files
            - ``mel_ok`` -- number of valid mel files (if *check_mel*)
            - ``mel_missing`` -- number of missing mel files
            - ``nan_count`` -- number of mel files containing NaN
            - ``duration_total_hours`` -- total audio duration in hours
            - ``mel_shapes`` -- set of unique mel shapes found
        """
        stats: dict = {
            "total": len(manifest),
            "audio_ok": 0,
            "audio_missing": 0,
            "mel_ok": 0,
            "mel_missing": 0,
            "nan_count": 0,
            "duration_total_hours": 0.0,
            "mel_shapes": set(),
        }

        for utt in manifest:
            # Check resampled audio
            audio_path = Path(utt.audio_path)
            if audio_path.exists():
                stats["audio_ok"] += 1
                stats["duration_total_hours"] += utt.duration / 3600.0
            else:
                stats["audio_missing"] += 1
                logger.debug("Missing audio: %s", audio_path)

            # Check mel
            if check_mel:
                mel_path = self.get_mel_path(utt)
                if mel_path.exists():
                    try:
                        mel = torch.load(mel_path, weights_only=True).float()
                        stats["mel_shapes"].add(tuple(mel.shape))
                        if torch.isnan(mel).any():
                            stats["nan_count"] += 1
                            logger.warning("NaN in mel: %s", mel_path)
                        else:
                            stats["mel_ok"] += 1
                    except Exception as exc:  # noqa: BLE001
                        stats["mel_missing"] += 1
                        logger.warning("Cannot load mel %s: %s", mel_path, exc)
                else:
                    stats["mel_missing"] += 1
                    logger.debug("Missing mel: %s", mel_path)

        # Convert set to sorted list for JSON serialisability
        stats["mel_shapes"] = sorted(stats["mel_shapes"])

        logger.info("Verification: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        worker_fn,
        tasks: list[tuple],
        stage_name: str = "",
    ) -> list[dict]:
        """Execute *worker_fn* over *tasks* using a process pool.

        Returns results in the **same order** as *tasks*.

        A progress message is logged every 10 % of tasks completed.
        """
        total = len(tasks)
        if total == 0:
            return []

        # Maintain ordering: map future -> index
        results: list[dict | None] = [None] * total
        log_interval = max(1, total // 10)
        completed = 0

        # When num_workers <= 1, avoid the overhead of multiprocessing
        # entirely. This also makes debugging easier.
        if self.num_workers <= 1:
            for idx, task in enumerate(tasks):
                results[idx] = worker_fn(task)
                completed += 1
                if completed % log_interval == 0 or completed == total:
                    logger.info(
                        "[%s] %d / %d (%.0f%%)",
                        stage_name,
                        completed,
                        total,
                        100 * completed / total,
                    )
            return results  # type: ignore[return-value]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {}
            for idx, task in enumerate(tasks):
                fut = executor.submit(worker_fn, task)
                future_to_idx[fut] = idx

            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[idx] = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                completed += 1
                if completed % log_interval == 0 or completed == total:
                    logger.info(
                        "[%s] %d / %d (%.0f%%)",
                        stage_name,
                        completed,
                        total,
                        100 * completed / total,
                    )

        return results  # type: ignore[return-value]
