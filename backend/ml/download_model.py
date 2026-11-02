"""
Model Downloader Utility for LungAI
Verifies and downloads trained model checkpoints if missing:
- Model D (DenseNet-121 Frequency V5) — the diagnostic classifier.
- CXR Gate model — the Stage-2 anatomical plausibility classifier.

CRITICAL: if the CXR Gate checkpoint (cxr_gate_model.keras) is missing, the
application does NOT fail — ml/cxr_gate.py silently falls back to a much
weaker heuristic (see CXRGate._anatomical_profile_score). That fallback was
measured to accept plain geometric shapes (grey squares, discs) as "plausible
chest radiographs." This script therefore treats a missing gate checkpoint as
a critical warning, not a cosmetic one, and exits non-zero so it can't be
missed in CI/deployment logs.
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
# NOTE: as of this writing, the v1.0.0 release has NO attached assets at all —
# every URL below currently 404s. Attach the actual files to a GitHub Release
# (repo -> Releases -> edit v1.0.0 or cut a new tag -> upload binaries) before
# relying on this script to fetch them automatically.
MODEL_URLS = {
    "densenet121_frequency_v5.h5": "https://github.com/KSathwik/Lung_Disease_Detector/releases/download/v1.0.0/densenet121_frequency_v5.h5",
    "training_results.json": "https://github.com/KSathwik/Lung_Disease_Detector/releases/download/v1.0.0/training_results.json",
}

# Checkpoints that gate the SAFETY of the pipeline (not just its accuracy).
# A missing file here means the app still runs, but silently in a degraded
# safety mode. Flagged separately so that's impossible to miss.
CRITICAL_SAFETY_MODELS = {
    "cxr_gate_model.keras": "https://github.com/KSathwik/Lung_Disease_Detector/releases/download/v1.0.0/cxr_gate_model.keras",
}


def _download(filename: str, url: str, filepath: Path) -> bool:
    """Attempt one download. Returns True on success."""
    print(f"📥 Downloading {filename} from GitHub Releases...")
    try:
        def progress(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, int(count * block_size * 100 / total_size))
                sys.stdout.write(f"\r   Progress: {percent}% [{count * block_size / 1024 / 1024:.1f} MB]")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, filepath, reporthook=progress)
        print(f"\n✅ {filename} downloaded successfully.")
        return True
    except Exception as e:
        print(f"\n⚠️  Unable to download {filename} automatically ({e}).")
        print(f"   Please download manually from: {url}")
        print(f"   and place it in the `models/` directory.")
        return False


def verify_or_download_models() -> bool:
    """
    Verify presence of model files or download from GitHub Releases.
    Returns True iff every CRITICAL_SAFETY_MODELS file is present afterwards.
    """
    print("=== LungAI Model Checkpoint Verification ===")

    for filename, url in MODEL_URLS.items():
        filepath = MODELS_DIR / filename
        if filepath.exists() and filepath.stat().st_size > 1000:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} found ({size_mb:.1f} MB)")
        else:
            _download(filename, url, filepath)

    print("\n=== Safety-Critical Checkpoint Verification (CXR Gate) ===")
    gate_ok = True
    for filename, url in CRITICAL_SAFETY_MODELS.items():
        filepath = MODELS_DIR / filename
        if filepath.exists() and filepath.stat().st_size > 1000:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} found ({size_mb:.1f} MB)")
        else:
            downloaded = _download(filename, url, filepath)
            if not downloaded:
                gate_ok = False

    if not gate_ok:
        print(
            "\n🛑 CRITICAL: The trained CXR Gate classifier is missing. "
            "The application will still start, but ml/cxr_gate.py will silently "
            "fall back to a much weaker heuristic that accepts non-anatomical "
            "images (e.g. plain shapes) as plausible chest radiographs. "
            "Do not treat the gate's behavior as validated until this file is present."
        )
    else:
        print("\n✅ All safety-critical checkpoints present.")

    return gate_ok


if __name__ == "__main__":
    ok = verify_or_download_models()
    sys.exit(0 if ok else 1)
