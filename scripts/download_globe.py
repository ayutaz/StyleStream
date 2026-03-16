"""Download GLOBE dataset for accent-diverse speech.

GLOBE: Multi-accent English speech corpus.
Used in LMG dataset for Destylizer training and for accent evaluation targets.

The GLOBE corpus provides recordings of English speech from speakers with
various L1 (native language) backgrounds, making it valuable for accent
conversion research.

Note: GLOBE access may require registration.  This script supports
downloading from a configurable URL and organises the files into the
standard layout expected by the StyleStream pipeline.

Usage:
    python scripts/download_globe.py --output-dir data/raw/globe
    python scripts/download_globe.py --output-dir data/raw/globe --source-url https://example.com/globe.tar.gz
    python scripts/download_globe.py --output-dir data/raw/globe --local-archive /path/to/globe.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
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

# GLOBE does not have a single canonical download URL.
# The dataset may be obtained from the authors or an institutional mirror.
# Users should provide the URL or a local archive path.
DEFAULT_GLOBE_URL = ""

# Common archive names
ARCHIVE_EXTENSIONS = {".tar.gz", ".tgz", ".zip", ".tar.bz2"}

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


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract a tar.gz, tgz, tar.bz2, or zip archive.

    Parameters
    ----------
    archive_path:
        Path to the archive file.
    output_dir:
        Directory to extract into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()

    if name.endswith(".zip"):
        logger.info("Extracting zip: %s -> %s ...", archive_path.name, output_dir)
        with zipfile.ZipFile(archive_path, "r") as zf:
            safe_members = [
                m for m in zf.infolist()
                if not m.filename.startswith("/") and ".." not in m.filename
            ]
            try:
                from tqdm import tqdm

                for info in tqdm(safe_members, desc="Extracting", unit="files"):
                    zf.extract(info, output_dir)
            except ImportError:
                total = len(safe_members)
                for i, info in enumerate(safe_members):
                    zf.extract(info, output_dir)
                    if (i + 1) % 500 == 0 or (i + 1) == total:
                        logger.info("  Extracted %d/%d files", i + 1, total)

    elif name.endswith((".tar.gz", ".tgz", ".tar.bz2")):
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:bz2"
        logger.info("Extracting tar (%s): %s -> %s ...", mode, archive_path.name, output_dir)
        with tarfile.open(archive_path, mode) as tar:
            safe_members = [
                m for m in tar.getmembers()
                if not m.name.startswith("/") and ".." not in m.name
            ]
            tar.extractall(path=output_dir, members=safe_members)
    else:
        logger.error("Unsupported archive format: %s", archive_path.name)
        sys.exit(1)

    logger.info("Extraction complete: %s", archive_path.name)


def scan_globe_structure(root_dir: Path) -> dict[str, list[str]]:
    """Scan a GLOBE directory and report its structure.

    Returns
    -------
    dict
        Mapping from accent (or speaker) directory name to list of WAV
        filenames found under it.
    """
    structure: dict[str, list[str]] = {}

    for item in sorted(root_dir.iterdir()):
        if not item.is_dir():
            continue

        wav_files = list(item.rglob("*.wav"))
        if wav_files:
            structure[item.name] = [str(w.relative_to(root_dir)) for w in wav_files]

    return structure


def find_globe_root(extract_dir: Path) -> Path:
    """Find the actual GLOBE data root inside an extraction directory.

    Archives sometimes have a top-level wrapper directory.  This function
    tries to find the deepest directory that contains accent or speaker
    subdirectories with WAV files.

    Parameters
    ----------
    extract_dir:
        Where the archive was extracted.

    Returns
    -------
    Path
        The effective root directory for GLOBE data.
    """
    # Check if extract_dir itself contains WAV-bearing subdirectories
    wav_files = list(extract_dir.rglob("*.wav"))
    if not wav_files:
        return extract_dir

    # Find the common prefix of all WAV files relative to extract_dir
    first_wav = wav_files[0]
    rel = first_wav.relative_to(extract_dir)

    # If there's a single top-level directory, descend into it
    top_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    if len(top_dirs) == 1:
        candidate = top_dirs[0]
        inner_wavs = list(candidate.rglob("*.wav"))
        if inner_wavs:
            # Check if there's yet another single wrapper
            inner_dirs = [d for d in candidate.iterdir() if d.is_dir()]
            if len(inner_dirs) == 1:
                inner_inner_wavs = list(inner_dirs[0].rglob("*.wav"))
                if inner_inner_wavs:
                    return inner_dirs[0]
            return candidate

    return extract_dir


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GLOBE dataset for accent-diverse speech.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_globe.py --output-dir data/raw/globe "
            "--source-url https://example.com/globe.tar.gz\n"
            "  python scripts/download_globe.py --output-dir data/raw/globe "
            "--local-archive /downloads/globe.zip\n"
            "  python scripts/download_globe.py --output-dir data/raw/globe "
            "--scan-only\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/globe",
        help="Directory to store the extracted GLOBE dataset (default: data/raw/globe).",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default="",
        help=(
            "URL to download the GLOBE archive from.  "
            "GLOBE may require institutional access; provide the URL after obtaining it."
        ),
    )
    parser.add_argument(
        "--local-archive",
        type=str,
        default=None,
        help=(
            "Path to a locally available GLOBE archive (tar.gz, zip, etc.).  "
            "Use this if you have already downloaded the file manually."
        ),
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download but do not extract the archive.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip manifest generation after extraction.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path for the output manifest CSV. Defaults to <output-dir>/manifest.csv.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the archive after extraction (default: delete).",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help=(
            "Do not download.  Scan --output-dir for existing GLOBE data "
            "and build a manifest."
        ),
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Scan-only mode: just build manifest from existing data
    # ------------------------------------------------------------------
    if args.scan_only:
        wav_count = len(list(output_dir.rglob("*.wav")))
        if wav_count == 0:
            logger.error("No WAV files found in %s. Nothing to scan.", output_dir)
            sys.exit(1)

        logger.info("Found %d WAV files in %s", wav_count, output_dir)
        _build_and_save_manifest(output_dir, args.manifest_path)
        _print_summary(output_dir)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Determine archive source
    # ------------------------------------------------------------------
    archive_path: Path | None = None

    if args.local_archive:
        archive_path = Path(args.local_archive)
        if not archive_path.exists():
            logger.error("Local archive not found: %s", archive_path)
            sys.exit(1)
        logger.info("Using local archive: %s", archive_path)

    elif args.source_url:
        archive_name = args.source_url.split("/")[-1].split("?")[0]
        if not archive_name or "." not in archive_name:
            archive_name = "globe_dataset.tar.gz"
        archive_path = output_dir / archive_name

        ok = download_file(args.source_url, archive_path)
        if not ok:
            logger.error("Failed to download GLOBE dataset. Exiting.")
            sys.exit(1)
    else:
        # No URL and no local archive -- check if data already exists
        existing_wavs = list(output_dir.rglob("*.wav"))
        if existing_wavs:
            logger.info(
                "No download source specified, but found %d WAV files in %s.",
                len(existing_wavs),
                output_dir,
            )
            logger.info("Building manifest from existing data ...")
            _build_and_save_manifest(output_dir, args.manifest_path)
            _print_summary(output_dir)
            sys.exit(0)

        logger.error(
            "No download source specified.\n\n"
            "GLOBE dataset access may require registration.  Please provide either:\n"
            "  --source-url <URL>          URL to download the archive\n"
            "  --local-archive <path>      path to a pre-downloaded archive\n"
            "\n"
            "If you already have GLOBE data extracted in %s, use --scan-only\n"
            "to build a manifest from existing files.",
            output_dir,
        )
        sys.exit(1)

    if args.no_extract:
        print(f"\nDownloaded: {archive_path}")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------
    if archive_path is not None:
        extract_dir = output_dir / "_raw_extract"
        extract_archive(archive_path, extract_dir)

        # Find the actual data root
        globe_root = find_globe_root(extract_dir)

        if globe_root != output_dir:
            # Move contents from globe_root to output_dir
            import shutil

            for item in globe_root.iterdir():
                target = output_dir / item.name
                if target.exists() and target.name.startswith("_"):
                    continue
                if item.is_dir():
                    if target.exists():
                        # Merge directories
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.move(str(item), str(target))
                else:
                    shutil.move(str(item), str(target))

        # Cleanup
        if extract_dir.exists():
            import shutil

            shutil.rmtree(extract_dir, ignore_errors=True)

        if not args.keep_archive and archive_path.exists():
            logger.info("Removing archive: %s", archive_path.name)
            archive_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    if not args.no_manifest:
        _build_and_save_manifest(output_dir, args.manifest_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(output_dir)


def _build_and_save_manifest(data_dir: Path, manifest_path: str | None) -> None:
    """Build and save a GLOBE manifest."""
    logger.info("Building manifest ...")
    try:
        from stylestream.data.manifest import Manifest

        manifest = Manifest.from_globe(data_dir)
        save_path = Path(manifest_path) if manifest_path else (data_dir / "manifest.csv")
        manifest.save(save_path)
        print(f"\n{manifest.summary()}")
    except Exception as e:
        logger.error("Failed to build manifest: %s", e)
        logger.info(
            "You can build the manifest later with:\n"
            "  python -c \"from stylestream.data.manifest import Manifest; "
            "m = Manifest.from_globe('%s'); m.save('%s/manifest.csv')\"",
            data_dir,
            data_dir,
        )


def _print_summary(data_dir: Path) -> None:
    """Print a summary of the GLOBE data on disk."""
    wav_files = list(data_dir.rglob("*.wav"))
    structure = scan_globe_structure(data_dir)

    # Try to compute total duration
    total_duration_hrs = 0.0
    try:
        import soundfile as sf

        total_seconds = 0.0
        for wav in wav_files:
            try:
                info = sf.info(str(wav))
                total_seconds += info.frames / info.samplerate
            except Exception:
                pass
        total_duration_hrs = total_seconds / 3600.0
    except ImportError:
        pass

    accents = sorted(structure.keys())

    print("\n" + "=" * 60)
    print("GLOBE Download Summary")
    print("=" * 60)
    print(f"  Output directory : {data_dir.resolve()}")
    print(f"  Total WAV files  : {len(wav_files)}")
    if total_duration_hrs > 0:
        print(f"  Total duration   : {total_duration_hrs:.1f} hours")
    if accents:
        print(f"  Accents/groups   : {len(accents)}")
        print("  Files per group:")
        for acc in accents:
            if acc.startswith("_"):
                continue
            print(f"    {acc:<20} : {len(structure[acc])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
