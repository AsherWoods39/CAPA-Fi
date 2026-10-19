"""
CAPA-Fi: Robust Data Downloader Utility
Downloads the WiMANS CSI dataset automatically from Kaggle using kagglehub with:
- Automatic resume and retry on network dropouts / TCP connection reset
- Zero-copy NTFS hard-linking on Windows to avoid duplicating 50 GB of disk space
- Optional cache archive cleanup to reclaim storage
"""

import argparse
import logging
import os
from pathlib import Path
import shutil
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CAPA-Fi.DataDownloader")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "WiMANS"


def download_wimans_dataset(
    target_dir: Path = DEFAULT_TARGET_DIR,
    max_retries: int = 50,
    retry_delay: float = 5.0,
    cleanup_cache: bool = False,
) -> Path:
    """
    Downloads WiMANS dataset via kagglehub with automatic resume on connection drops.
    Ensures .npy files are linked to target_dir (default: data/raw/WiMANS).
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check if files are already present in target directory
    existing_files = list(target_dir.glob("*.npy"))
    if len(existing_files) > 0:
        total_size_gb = sum(f.stat().st_size for f in existing_files) / (1024 ** 3)
        logger.info(
            f"Dataset already available at {target_dir} "
            f"({len(existing_files)} .npy files found, {total_size_gb:.2f} GB)."
        )
        return target_dir

    logger.info("Initiating WiMANS dataset download from Kaggle (dataset: shuokanghuang/wimans)...")
    logger.info("Kagglehub automatically stores progress and resumes broken streams.")

    import kagglehub

    downloaded_path = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[Attempt {attempt}/{max_retries}] Connecting to Kaggle...")
            # kagglehub uses HTTP Range headers to resume from existing .archive progress
            res = kagglehub.dataset_download("shuokanghuang/wimans")
            downloaded_path = Path(res)
            logger.info(f"Download complete! Files extracted to: {downloaded_path}")
            break
        except Exception as e:
            logger.warning(
                f"[Attempt {attempt}/{max_retries}] Download interrupted: {e}\n"
                f"Progress is preserved in the local archive. Resuming in {retry_delay:.0f} seconds..."
            )
            if attempt == max_retries:
                logger.error(f"Failed after {max_retries} attempts.")
                raise RuntimeError(f"Download failed after {max_retries} attempts: {e}") from e
            time.sleep(retry_delay)

    if downloaded_path is None or not downloaded_path.exists():
        raise RuntimeError("Download completed but destination path does not exist.")

    # 2. Recursively discover .npy files inside Kaggle download directory
    npy_files = list(downloaded_path.rglob("*.npy"))
    if len(npy_files) == 0:
        logger.warning(f"No .npy files found in {downloaded_path}. Searching for sub-archives or subdirectories...")
        for child in downloaded_path.iterdir():
            logger.info(f"  Found entry: {child.name} (is_dir={child.is_dir()})")

    logger.info(f"Discovered {len(npy_files)} .npy files. Linking to target directory {target_dir}...")

    linked_count = 0
    copied_count = 0
    skipped_count = 0

    for npy_file in npy_files:
        dest_file = target_dir / npy_file.name
        if dest_file.exists():
            skipped_count += 1
            continue

        # Try NTFS hard link first (0 additional disk space, instantaneous on drive C:)
        try:
            os.link(npy_file, dest_file)
            linked_count += 1
        except Exception:
            # Fallback to copy if hard link fails (e.g. cross-filesystem)
            try:
                shutil.copy2(npy_file, dest_file)
                copied_count += 1
            except Exception as copy_err:
                logger.error(f"Failed to link or copy {npy_file.name}: {copy_err}")

    logger.info(
        f"Target directory populated: {linked_count} hard-linked (zero extra disk space), "
        f"{copied_count} copied, {skipped_count} already existed in {target_dir}."
    )

    # 3. Optional archive cleanup to reclaim ~50 GB in ~/.cache/kagglehub
    if cleanup_cache:
        user_home = Path(os.environ.get("USERPROFILE", os.path.expanduser("~")))
        archive_dir = user_home / ".cache" / "kagglehub" / "datasets" / "shuokanghuang" / "wimans"
        if archive_dir.exists():
            archives = list(archive_dir.glob("*.archive"))
            for arch in archives:
                arch_size_gb = arch.stat().st_size / (1024 ** 3)
                try:
                    arch.unlink()
                    logger.info(f"Cleaned up cache archive {arch.name} ({arch_size_gb:.2f} GB reclaimed).")
                except Exception as del_err:
                    logger.warning(f"Could not delete {arch}: {del_err}")

    return target_dir


def main():
    parser = argparse.ArgumentParser(description="CAPA-Fi: WiMANS Dataset Downloader with Auto-Resume")
    parser.add_argument(
        "--target-dir",
        type=str,
        default=str(DEFAULT_TARGET_DIR),
        help=f"Destination directory for raw .npy files (default: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=50,
        help="Maximum resume retries on network dropouts (default: 50)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Seconds to wait before reconnecting after an error (default: 5.0)",
    )
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Delete the downloaded .archive file from ~/.cache after extraction to free 50 GB disk space",
    )
    args = parser.parse_args()

    download_wimans_dataset(
        target_dir=Path(args.target_dir),
        max_retries=args.retries,
        retry_delay=args.retry_delay,
        cleanup_cache=args.cleanup_cache,
    )


if __name__ == "__main__":
    main()