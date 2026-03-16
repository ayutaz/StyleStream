"""Download LibriTTS dataset from OpenSLR.

LibriTTS: ~585 hours of English read speech at 24kHz.
URL: https://www.openslr.org/60/

Subsets:
  - train-clean-100 (28.3h)
  - train-clean-360 (104.0h)
  - train-other-500 (151.2h)
  - dev-clean (5.4h)
  - dev-other (5.4h)
  - test-clean (4.8h)
  - test-other (5.1h)

Usage:
    python scripts/download_libritts.py --output-dir data/raw/libritts
    python scripts/download_libritts.py --output-dir data/raw/libritts --subsets train-clean-100 dev-clean test-clean
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
import time
import urllib.request
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

BASE_URL = "https://www.openslr.org/resources/60"

# Subset name -> (tar.gz filename, MD5 checksum)
# Checksums from https://www.openslr.org/60/
SUBSETS: dict[str, dict[str, str]] = {
    "train-clean-100": {
        "filename": "train-clean-100.tar.gz",
        "md5": "7e3965da8f329b6d2a7988f960987862",
    },
    "train-clean-360": {
        "filename": "train-clean-360.tar.gz",
        "md5": "554873673bca6dca4f0b59969e2564e5",
    },
    "train-other-500": {
        "filename": "train-other-500.tar.gz",
        "md5": "bc96e79cee1e4f1a1b5806136e399b37",
    },
    "dev-clean": {
        "filename": "dev-clean.tar.gz",
        "md5": "0c3076c1e5245bb3f0af7b4fc4b5b8a2",
    },
    "dev-other": {
        "filename": "dev-other.tar.gz",
        "md5": "815555d8d75a3223e8fc9f3f3ea1c25c",
    },
    "test-clean": {
        "filename": "test-clean.tar.gz",
        "md5": "40df97ee70eca72e6e93a1a3c0eae807",
    },
    "test-other": {
        "filename": "test-other.tar.gz",
        "md5": "004af0ef26aa2838bd2ce3a4e6bae3b1",
    },
}

ALL_SUBSET_NAMES = list(SUBSETS.keys())

MAX_RETRIES = 3
RETRY_DELAY_SEC = 5


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
            from tqdm import tqdm

            self.pbar: tqdm | None = None
            self.use_tqdm = True
        except ImportError:
            self.use_tqdm = False

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

        downloaded = block_num * block_size
        if self.pbar is not None:
            self.pbar.update(block_size)
            if total_size > 0 and downloaded >= total_size:
                self.pbar.close()

    def _simple_hook(self, block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        now = time.time()
        if now - self.last_print < 2.0 and downloaded < total_size:
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
        """Ensure progress bar is closed cleanly."""
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


def download_file(url: str, dest: Path, md5: str | None = None) -> bool:
    """Download a file with retry logic and optional MD5 verification.

    Parameters
    ----------
    url:
        URL to download from.
    dest:
        Local destination path.
    md5:
        Expected MD5 checksum.  If provided, the downloaded file is
        verified and the download is retried on mismatch.

    Returns
    -------
    bool
        True if the file was downloaded (or already existed) and verified.
    """
    # Skip if already downloaded and verified
    if dest.exists():
        if md5:
            existing_md5 = compute_md5(dest)
            if existing_md5 == md5:
                logger.info("Already downloaded and verified: %s", dest.name)
                return True
            else:
                logger.warning(
                    "Existing file %s has MD5 mismatch (got %s, expected %s). Re-downloading.",
                    dest.name,
                    existing_md5,
                    md5,
                )
        else:
            logger.info("Already downloaded (no checksum to verify): %s", dest.name)
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

            # Verify checksum
            if md5:
                actual_md5 = compute_md5(dest)
                if actual_md5 != md5:
                    logger.error(
                        "MD5 mismatch for %s: got %s, expected %s",
                        dest.name,
                        actual_md5,
                        md5,
                    )
                    dest.unlink(missing_ok=True)
                    if attempt < MAX_RETRIES:
                        logger.info("Retrying in %d seconds ...", RETRY_DELAY_SEC)
                        time.sleep(RETRY_DELAY_SEC)
                        continue
                    return False

            logger.info("Downloaded: %s", dest.name)
            return True

        except Exception as e:
            logger.error("Download failed for %s: %s", dest.name, e)
            dest.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                logger.info("Retrying in %d seconds ...", RETRY_DELAY_SEC)
                time.sleep(RETRY_DELAY_SEC)
            else:
                logger.error("All %d attempts failed for %s", MAX_RETRIES, dest.name)
                return False

    return False


def extract_tarball(tar_path: Path, output_dir: Path) -> None:
    """Extract a tar.gz file to output_dir.

    Parameters
    ----------
    tar_path:
        Path to the .tar.gz file.
    output_dir:
        Directory to extract into.
    """
    logger.info("Extracting %s -> %s ...", tar_path.name, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        # Security: filter out absolute paths and paths with ..
        members = []
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                logger.warning("Skipping suspicious tar member: %s", member.name)
                continue
            members.append(member)

        tar.extractall(path=output_dir, members=members)

    logger.info("Extraction complete: %s", tar_path.name)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download LibriTTS dataset from OpenSLR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_libritts.py --output-dir data/raw/libritts\n"
            "  python scripts/download_libritts.py --output-dir data/raw/libritts "
            "--subsets train-clean-100 dev-clean test-clean\n"
            "  python scripts/download_libritts.py --output-dir data/raw/libritts "
            "--no-extract\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/libritts",
        help="Directory to download and extract into (default: data/raw/libritts).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=ALL_SUBSET_NAMES,
        default=ALL_SUBSET_NAMES,
        help="Subsets to download (default: all).",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download tar.gz files but do not extract.",
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
        "--keep-tar",
        action="store_true",
        help="Keep tar.gz archives after extraction (default: delete them).",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    succeeded: list[str] = []
    failed: list[str] = []

    for subset_name in args.subsets:
        info = SUBSETS[subset_name]
        url = f"{BASE_URL}/{info['filename']}"
        tar_path = output_dir / info["filename"]

        ok = download_file(url, tar_path, md5=info["md5"])
        if not ok:
            failed.append(subset_name)
            continue

        # Extract
        if not args.no_extract:
            extract_tarball(tar_path, output_dir)
            if not args.keep_tar:
                logger.info("Removing archive: %s", tar_path.name)
                tar_path.unlink(missing_ok=True)

        succeeded.append(subset_name)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    if not args.no_manifest and succeeded and not args.no_extract:
        logger.info("Building manifest ...")
        try:
            from stylestream.data.manifest import Manifest

            manifest = Manifest.from_libritts(output_dir, subsets=succeeded)
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
                "m = Manifest.from_libritts('%s'); m.save('%s/manifest.csv')\"",
                output_dir,
                output_dir,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LibriTTS Download Summary")
    print("=" * 60)
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  Succeeded        : {len(succeeded)} subsets ({', '.join(succeeded) or 'none'})")
    if failed:
        print(f"  Failed           : {len(failed)} subsets ({', '.join(failed)})")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
