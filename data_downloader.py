"""
FusionKitchen — Automated Dataset Downloader
============================================
Downloads the ~5GB pipeline dataset from Hugging Face Datasets Hub
(or optional Google Drive / direct link) into the project's `data/` directory
if the files are not already present on disk.
"""

import os
import sys
import shutil

import config

REQUIRED_FILES = [
    ("recommender_output", "recipe_vectors_word2vec.npy"),
    ("recommender_output", "recipe_lookup_table.parquet"),
    ("embeddings_output", "word2vec_vectors_canonical.parquet"),
]


def is_data_present() -> bool:
    """Checks if the required pipeline artifacts are present locally or in data/."""
    # Check if config paths already exist (local Windows dev or data/ folder)
    if all(os.path.exists(p) for p in [
        config.RECIPE_VECTORS_W2V_PATH,
        config.RECIPE_LOOKUP_PATH,
        config.W2V_VECTORS_PATH,
    ]):
        return True

    # Check directly inside the data directory
    data_dir = config.DATA_DIR
    for folder, fname in REQUIRED_FILES:
        target_file = os.path.join(data_dir, folder, fname)
        if not os.path.exists(target_file):
            return False
    return True


def ensure_data_downloaded():
    """Ensures data is present. If missing, attempts download from Hugging Face Datasets."""
    if is_data_present():
        return True

    hf_repo = os.environ.get("HF_DATASET_REPO", config.HF_DATASET_REPO).strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    if not hf_repo:
        print("[Dataset Downloader] No HF_DATASET_REPO configured and local data files not found.")
        print("[Dataset Downloader] App will continue in DEMO DATA mode unless data is downloaded or uploaded.")
        return False

    print(f"\n{'=' * 60}")
    print(f"[Dataset Downloader] Downloading pipeline dataset from Hugging Face: {hf_repo}...")
    print(f"Target directory: {config.DATA_DIR}")
    print(f"{'=' * 60}\n")

    try:
        from huggingface_hub import snapshot_download

        os.makedirs(config.DATA_DIR, exist_ok=True)

        downloaded_path = snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=config.DATA_DIR,
            token=hf_token,
            resume_download=True,
        )

        print(f"\n[Dataset Downloader] Download complete! Files stored at: {downloaded_path}\n")
        return True

    except ImportError:
        print("[Dataset Downloader] huggingface_hub package is not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[Dataset Downloader] Failed to download dataset from {hf_repo}: {e}")
        return False


if __name__ == "__main__":
    ensure_data_downloaded()
