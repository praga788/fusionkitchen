"""
FusionKitchen — Helper Script: Upload 5GB Dataset to Hugging Face
=================================================================
This script uploads your local 5GB dataset folders to your free Hugging Face
Dataset repository in one command.

Usage:
1. Make sure you have your free Hugging Face Write Token:
   - Go to: https://huggingface.co/settings/tokens
   - Create a token with 'Write' permissions.

2. Run:
   python upload_dataset_to_hf.py --repo your-username/fusionkitchen-data --token your_hf_token
"""

import os
import argparse
import config

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please run: pip install huggingface_hub")
    exit(1)


def upload_dataset(repo_id: str, token: str, data_dir: str = None):
    api = HfApi(token=token)

    # Check / create the dataset repository
    print(f"Checking dataset repository '{repo_id}' on Hugging Face...")
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True, private=False)
        print(f"Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"Notice during repo verification: {e}")

    # Determine folders to upload
    subfolders = [
        ("recommender_output", config.RECOMMENDER_DIR),
        ("embeddings_output", config.EMBEDDINGS_DIR),
        ("genai_output", config.GENAI_DIR),
    ]

    for subfolder_name, local_path in subfolders:
        if not os.path.exists(local_path):
            # Check under data/ folder as fallback
            fallback = os.path.join(config.DATA_DIR, subfolder_name)
            if os.path.exists(fallback):
                local_path = fallback
            else:
                print(f"[Skipping] Folder not found locally: {local_path}")
                continue

        print(f"\nUploading '{subfolder_name}' from {local_path} -> {repo_id}...")
        try:
            api.upload_folder(
                folder_path=local_path,
                path_in_repo=subfolder_name,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            print(f"Successfully uploaded '{subfolder_name}'!")
        except Exception as e:
            print(f"Error uploading '{subfolder_name}': {e}")

    print(f"\n{'=' * 60}")
    print(f"All available folders uploaded to https://huggingface.co/datasets/{repo_id}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload FusionKitchen dataset to Hugging Face")
    parser.add_argument("--repo", type=str, required=True, help="HF dataset repo ID (e.g. username/fusionkitchen-data)")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""), help="Hugging Face Write Token")

    args = parser.parse_args()

    if not args.token:
        args.token = input("Enter your Hugging Face Write Token: ").strip()

    upload_dataset(repo_id=args.repo, token=args.token)
