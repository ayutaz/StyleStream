"""Download Emotional Speech Dataset (ESD).

ESD: 350 parallel utterances x 10 English speakers x 5 emotions.
Used as alternative to MSP-Podcast (which requires research application).

URL: https://hltsingapore.github.io/ESD/

Emotions: Angry, Happy, Neutral, Sad, Surprise
English speakers: 0001-0010 (speakers 0011-0020 are Mandarin Chinese)

The dataset is hosted on Zenodo and distributed as a single zip file.

Usage:
    python scripts/download_esd.py --output-dir data/raw/esd
    python scripts/download_esd.py --output-dir data/raw/esd --speakers 0011 0012 0013
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

# ESD is hosted on Zenodo.  The English portion is speakers 0001-0010.
# Full dataset: https://zenodo.org/records/7572904
# The download link points to the zip containing all speakers.
ESD_ZENODO_URL = "https://zenodo.org/records/7572904/files/Emotional%20Speech%20Dataset%20(ESD).zip"
ESD_FILENAME = "Emotional_Speech_Dataset_ESD.zip"

# English speaker IDs (Mandarin Chinese speakers are 0011-0020)
ENGLISH_SPEAKERS = [f"{i:04d}" for i in range(1, 11)]  # 0001..0010
ALL_SPEAKERS = [f"{i:04d}" for i in range(1, 21)]  # 0001..0020
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
SPLITS = ["train", "evaluation", "test"]

MAX_RETRIES = 3
RETRY_DELAY_SEC = 10


# ======================================================================
# Helpers
# ======================================================================

class _ProgressHook:
    """urllib reporthook that prints a tqdm-style progress bar."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.started = False
        self.last_print = 0.0
        try:
            from tqdm import tqdm  # noqa: F401

            self.use_tqdm = True
        except ImportError:
            self.use_tqdm = False
        self.pbar = None

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if self.use_tqdm:
            self._tqdm_hook(block_num, block_size, total_size)
        else:
            self._simple_hook(block_num, block_size, total_size)

    def _tqdm_hook(self, block_num: int, block_size: int, total_size: int) -> None:
        from tqdm import tqdm

        if not self.started:
            self.pbar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=self.filename,
            )
            self.started = True

        if self.pbar is not None:
            self.pbar.update(block_size)
            downloaded = block_num * block_size
            if total_size > 0 and downloaded >= total_size:
                self.pbar.close()

    def _simple_hook(self, block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        now = time.time()
        if now - self.last_print < 2.0 and (total_size <= 0 or downloaded < total_size):
            return
        self.last_print = now

        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            dl_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(
                f"\r  {self.filename}: {dl_mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)",
                end="",
                flush=True,
            )
            if downloaded >= total_size:
                print()
        else:
            dl_mb = downloaded / (1024 * 1024)
            print(f"\r  {self.filename}: {dl_mb:.1f} MB", end="", flush=True)

    def finish(self) -> None:
        if self.use_tqdm and self.pbar is not None:
            self.pbar.close()


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute the MD5 hex digest of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, dest: Path) -> bool:
    """Download a file with retry logic.

    Parameters
    ----------
    url:
        URL to download from.
    dest:
        Local destination path.

    Returns
    -------
    bool
        True if the file was downloaded or already exists.
    """
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "Downloading %s (attempt %d/%d) ...", dest.name, attempt, MAX_RETRIES
            )
            hook = _ProgressHook(dest.name)
            urllib.request.urlretrieve(url, str(dest), reporthook=hook)
            hook.finish()
            logger.info("Downloaded: %s", dest.name)
            return True

        except Exception as e:
            logger.error("Download failed: %s", e)
            dest.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                logger.info("Retrying in %d seconds ...", RETRY_DELAY_SEC)
                time.sleep(RETRY_DELAY_SEC)
            else:
                logger.error("All %d attempts failed for %s", MAX_RETRIES, dest.name)
                return False

    return False


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a zip file to output_dir.

    Parameters
    ----------
    zip_path:
        Path to the zip file.
    output_dir:
        Directory to extract into.
    """
    logger.info("Extracting %s -> %s ...", zip_path.name, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Security: filter out absolute paths and paths with ..
        safe_members = []
        for info in zf.infolist():
            if info.filename.startswith("/") or ".." in info.filename:
                logger.warning("Skipping suspicious zip member: %s", info.filename)
                continue
            safe_members.append(info)

        total = len(safe_members)
        try:
            from tqdm import tqdm

            for info in tqdm(safe_members, desc="Extracting", unit="files"):
                zf.extract(info, output_dir)
        except ImportError:
            for i, info in enumerate(safe_members):
                zf.extract(info, output_dir)
                if (i + 1) % 1000 == 0 or (i + 1) == total:
                    logger.info("  Extracted %d/%d files", i + 1, total)

    logger.info("Extraction complete: %d files", total)


def reorganize_esd(raw_dir: Path, output_dir: Path, english_only: bool = True) -> Path:
    """Reorganise the extracted ESD files into a clean layout.

    The Zenodo zip extracts to a messy nested structure.  This function
    copies (or moves) files into::

        output_dir/<speaker_id>/<emotion>/<split>/<filename>.wav

    Parameters
    ----------
    raw_dir:
        Directory where the zip was extracted.
    output_dir:
        Destination directory for reorganised files.
    english_only:
        If True, only include English speakers (0001-0010).

    Returns
    -------
    Path
        The output directory root that contains speaker subdirectories.
    """
    # Find the actual ESD root (zip may have a top-level folder)
    esd_root = raw_dir
    candidates = list(raw_dir.glob("**/0001"))
    if candidates:
        # Go up one level from a speaker directory
        esd_root = candidates[0].parent

    speakers = ENGLISH_SPEAKERS if english_only else ALL_SPEAKERS
    file_count = 0

    for speaker_id in speakers:
        speaker_dir = esd_root / speaker_id
        if not speaker_dir.is_dir():
            logger.warning("Speaker directory not found: %s", speaker_dir)
            continue

        for emotion in EMOTIONS:
            emotion_dir = speaker_dir / emotion
            if not emotion_dir.is_dir():
                continue

            for split in SPLITS:
                split_dir = emotion_dir / split
                if not split_dir.is_dir():
                    continue

                # Create the target directory
                target_dir = output_dir / speaker_id / emotion / split
                target_dir.mkdir(parents=True, exist_ok=True)

                for wav in sorted(split_dir.glob("*.wav")):
                    target = target_dir / wav.name
                    if not target.exists():
                        # Use hard link if possible, else copy
                        try:
                            target.hardlink_to(wav)
                        except OSError:
                            import shutil

                            shutil.copy2(wav, target)
                    file_count += 1

    logger.info("Reorganised %d files into %s", file_count, output_dir)
    return output_dir


def count_esd_files(root_dir: Path) -> tuple[int, dict[str, int], dict[str, int]]:
    """Count WAV files, grouping by speaker and emotion.

    Returns
    -------
    tuple
        (total_files, speaker_counts, emotion_counts)
    """
    speaker_counts: dict[str, int] = {}
    emotion_counts: dict[str, int] = {}
    total = 0

    for speaker_dir in sorted(root_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        spk = speaker_dir.name
        spk_count = 0

        for emotion in EMOTIONS:
            emotion_dir = speaker_dir / emotion
            if not emotion_dir.is_dir():
                continue

            emo_files = list(emotion_dir.rglob("*.wav"))
            n = len(emo_files)
            spk_count += n
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + n

        speaker_counts[spk] = spk_count
        total += spk_count

    return total, speaker_counts, emotion_counts


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Emotional Speech Dataset (ESD).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_esd.py --output-dir data/raw/esd\n"
            "  python scripts/download_esd.py --output-dir data/raw/esd --english-only\n"
            "  python scripts/download_esd.py --output-dir data/raw/esd --no-extract\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/esd",
        help="Directory to download and extract into (default: data/raw/esd).",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=ESD_ZENODO_URL,
        help="Override download URL for the ESD zip file.",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        default=True,
        help="Only include English speakers 0001-0010 (default: True).",
    )
    parser.add_argument(
        "--all-speakers",
        action="store_true",
        help="Include all speakers (overrides --english-only).",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download the zip but do not extract.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip manifest generation after download.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path for the output manifest CSV. Defaults to <output-dir>/manifest.csv.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip after extraction (default: delete).",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    english_only = args.english_only and not args.all_speakers

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    zip_path = output_dir / ESD_FILENAME

    ok = download_file(args.download_url, zip_path)
    if not ok:
        logger.error("Failed to download ESD. Exiting.")
        sys.exit(1)

    if args.no_extract:
        print(f"\nDownloaded: {zip_path}")
        print("Use --no-extract=false or re-run without --no-extract to extract.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------
    raw_extract_dir = output_dir / "_raw_extract"
    extract_zip(zip_path, raw_extract_dir)

    # Reorganise into clean structure
    logger.info("Reorganising ESD directory structure ...")
    data_dir = reorganize_esd(raw_extract_dir, output_dir, english_only=english_only)

    if not args.keep_zip:
        logger.info("Removing zip archive: %s", zip_path.name)
        zip_path.unlink(missing_ok=True)

    # Clean up raw extract directory
    import shutil

    if raw_extract_dir.exists():
        logger.info("Removing temporary extraction directory ...")
        shutil.rmtree(raw_extract_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    if not args.no_manifest:
        logger.info("Building manifest ...")
        try:
            from stylestream.data.manifest import Manifest

            manifest = Manifest.from_esd(data_dir)
            manifest_path = Path(args.manifest_path) if args.manifest_path else (
                output_dir / "manifest.csv"
            )
            manifest.save(manifest_path)
            print(f"\n{manifest.summary()}")
        except Exception as e:
            logger.error("Failed to build manifest: %s", e)
            logger.info(
                "You can build the manifest later with:\n"
                "  python -c \"from stylestream.data.manifest import Manifest; "
                "m = Manifest.from_esd('%s'); m.save('%s/manifest.csv')\"",
                data_dir,
                output_dir,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_files, speaker_counts, emotion_counts = count_esd_files(data_dir)

    print("\n" + "=" * 60)
    print("ESD Download Summary")
    print("=" * 60)
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  Total files      : {total_files}")
    print(f"  Speakers         : {len(speaker_counts)}")
    lang = "English only (0001-0010)" if english_only else "All (0001-0020)"
    print(f"  Language         : {lang}")
    print(f"  Emotions         : {', '.join(EMOTIONS)}")
    if emotion_counts:
        print("  Files per emotion:")
        for emo, cnt in sorted(emotion_counts.items()):
            print(f"    {emo:<12} : {cnt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
