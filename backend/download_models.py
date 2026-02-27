"""
download_models.py — Run this ONCE after uploading models to Hugging Face Hub.

Usage:
    python download_models.py

This script downloads model files from your Hugging Face Hub repository into
the local backend/models/ directory. The Render deployment calls this at startup
via the start command: python download_models.py && python app.py
"""

import os
from pathlib import Path

def download_models():
    """Download model files from Hugging Face Hub if not already present"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_MODEL_REPO")  # e.g. "yourname/detectifai-models"

    if not hf_repo:
        print("⚠️  HF_MODEL_REPO not set — skipping model download.")
        return

    # Directory to save models
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    # List of model files stored on Hugging Face Hub
    model_files = [
        "merged_fire_knife_gun.pt",
        "fire_YOLO11.pt",
        "weapon_YOLO11.pt",
        "classifier_svm.pkl",
        "label_encoder.pkl",
    ]

    for filename in model_files:
        dest = models_dir / filename
        if dest.exists():
            print(f"✅ {filename} already present — skipping download.")
            continue

        try:
            print(f"⬇️  Downloading {filename} from {hf_repo} ...")
            downloaded_path = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                token=hf_token,
                local_dir=str(models_dir),
            )
            print(f"✅ {filename} saved to {downloaded_path}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")


if __name__ == "__main__":
    download_models()
