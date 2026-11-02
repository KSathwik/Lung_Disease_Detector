"""Install the canonical checkpoint into models/ after verifying its SHA-256.

    python scripts/install_checkpoint.py --from /path/to/densenet121_frequency_v5.h5
    python scripts/install_checkpoint.py --url https://<location you control>/densenet121_frequency_v5.h5
    python scripts/install_checkpoint.py --check            # verify the file already in models/

The checkpoint is not distributed with the repository and there is no public release.
A file whose SHA-256 differs from lungai/configs/lungai_densenet121.json is rejected.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lungai.config import CHECKPOINT_FILENAME, CHECKPOINT_SHA256, REPO_ROOT, sha256_file  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"


def verify(path: Path) -> bool:
    digest = sha256_file(path)
    ok = digest == CHECKPOINT_SHA256
    print(f"{path.name}: sha256 {digest} -> {'MATCHES canonical' if ok else 'DOES NOT MATCH canonical ' + CHECKPOINT_SHA256}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from", dest="src", type=Path, help="local checkpoint file to copy")
    g.add_argument("--url", help="URL to download the checkpoint from")
    g.add_argument("--check", action="store_true", help="verify models/ copy only")
    args = ap.parse_args()
    target = MODELS_DIR / CHECKPOINT_FILENAME

    if args.check:
        if not target.is_file():
            print(f"missing: {target.relative_to(REPO_ROOT)}")
            return 1
        return 0 if verify(target) else 1

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / CHECKPOINT_FILENAME
        if args.src:
            shutil.copyfile(args.src, staged)
        else:
            urllib.request.urlretrieve(args.url, staged)
        if not verify(staged):
            print("Not installed.")
            return 1
        MODELS_DIR.mkdir(exist_ok=True)
        shutil.move(str(staged), target)
    print(f"installed: {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
