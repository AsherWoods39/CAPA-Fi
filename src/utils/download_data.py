"""
CAPA-Fi: Data Downloader Utility
Downloads the WiMANS CSI dataset automatically from Kaggle using kagglehub.
"""

from pathlib import Path
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CAPA-Fi.DataDownloader")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "WiMANS"


def download_wimans_dataset(target_dir: Path = DEFAULT_TARGET_DIR) -> Path:
    """
    Downloads WiMANS dataset via kagglehub and ensures files are accessible 
    in target_dir (default: data/raw/WiMANS).
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if files are already present
    existing_files = list(target_dir.glob("*.npy"))
    if len(existing_files) > 0:
        logger.info(f"Dataset already available at {target_dir} ({len(existing_files)} .npy files found).")
        return target_dir

    logger.info("Downloading WiMANS dataset from Kaggle via kagglehub...")
    try:
        import kagglehub
        downloaded_path = Path(kagglehub.dataset_download("shuokanghuang/wimans"))
        logger.info(f"Dataset downloaded by kagglehub to: {downloaded_path}")

        # Recursively discover .npy files inside Kaggle download directory
        npy_files = list(downloaded_path.rglob("*.npy"))
        logger.info(f"Discovered {len(npy_files)} .npy files. Linking/copying to {target_dir}...")

        for npy_file in npy_files:
            dest_file = target_dir / npy_file.name
            if not dest_file.exists():
                try:
                    # Attempt symlink first to conserve disk space
                    os.symlink(npy_file, dest_file)
                except (OSError, AttributeError):
                    # Fallback to copy on Windows without admin permissions
                    shutil.copy2(npy_file, dest_file)

        logger.info(f"Dataset successfully prepared in {target_dir}")
        return target_dir

    except Exception as e:
        logger.error(f"Failed to download dataset via kagglehub: {e}")
        raise RuntimeError(f"Data download error: {e}")


if __name__ == "__main__":
    download_wimans_dataset()