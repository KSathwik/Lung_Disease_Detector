"""Backend ML services: inference engine and CXR input gate.

The canonical model definition, class mapping and preprocessing live in the top-level
``lungai`` package. In a repository checkout it sits next to ``backend/``; in the Docker
image it is copied to /app/lungai, which is already importable.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "lungai").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
