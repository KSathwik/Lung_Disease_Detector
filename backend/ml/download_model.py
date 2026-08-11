"""
Model Downloader Utility for LungAI
Verifies and downloads trained model checkpoints (ResNet50 & CNN) if missing.
"""

import sys
import io
import urllib.request
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Release URL endpoints (Replace with active release assets on deployment)
MODEL_URLS = {
    "resnet_model.h5": "https://github.com/KSathwik/Lung_Disease_Detector/releases/download/v1.0.0/resnet_model.h5",
    "cnn_model.h5": "https://github.com/KSathwik/Lung_Disease_Detector/releases/download/v1.0.0/cnn_model.h5",
    "training_results.json": "https://github.com/KSathwik/Lung_Disease_Detector/releases/download/v1.0.0/training_results.json",
}

def verify_or_download_models():
    """Verify presence of model files or download from GitHub Releases."""
    print("=== LungAI Model Checkpoint Verification ===")
    
    for filename, url in MODEL_URLS.items():
        filepath = MODELS_DIR / filename
        if filepath.exists() and filepath.stat().st_size > 1000:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} found ({size_mb:.1f} MB)")
        else:
            print(f"📥 Downloading {filename} from GitHub Releases...")
            try:
                def progress(count, block_size, total_size):
                    if total_size > 0:
                        percent = min(100, int(count * block_size * 100 / total_size))
                        sys.stdout.write(f"\r   Progress: {percent}% [{count * block_size / 1024 / 1024:.1f} MB]")
                        sys.stdout.flush()

                urllib.request.urlretrieve(url, filepath, reporthook=progress)
                print(f"\n✅ {filename} downloaded successfully.")
            except Exception as e:
                print(f"\n⚠️  Unable to download {filename} automatically ({e}).")
                print(f"   Please download manually from: {url}")
                print(f"   and place it in the `models/` directory.")

if __name__ == "__main__":
    verify_or_download_models()
