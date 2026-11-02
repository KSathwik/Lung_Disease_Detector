"""Repository-wide consistency: one model identity, one class mapping, no mock paths, no stale claims."""

import re
import subprocess

import pytest

from lungai.config import CLASS_NAMES, REPO_ROOT

ACTIVE_CODE = ["backend/main.py", "backend/api", "backend/ml", "backend/database", "frontend/src", "lungai"]
ACTIVE_DOCS = ["README.md", "docs"]
EXCLUDE = re.compile(r"(^|/)(archive|node_modules|__pycache__|build)(/|$)")


def _tracked(paths, suffixes):
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", *paths],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split()
    except Exception:
        pytest.skip("git not available")
    return [REPO_ROOT / p for p in out if p.endswith(suffixes) and not EXCLUDE.search(p) and (REPO_ROOT / p).is_file()]


def _grep(files, pattern):
    rx = re.compile(pattern)
    hits = []
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(REPO_ROOT).as_posix()}:{n}: {line.strip()[:100]}")
    return hits


def test_no_competing_model_names_in_active_code_or_docs():
    files = _tracked(ACTIVE_CODE + ACTIVE_DOCS, (".py", ".js", ".md", ".json"))
    assert not _grep(files, r"\bModel [A-E]\b|Final Model|Best Model|Latest Model|ResNet|\bCNN\b"), "ambiguous model names"


def test_class_names_are_not_redeclared_in_active_code():
    """The six class names may appear as a list only in the canonical config."""
    files = _tracked(["backend/main.py", "backend/api", "backend/ml", "frontend/src", "lungai"], (".py", ".js"))
    hits = [h for h in _grep(files, r"[\"']Pleural Effusion[\"']\s*,\s*$|[\"']COVID-19[\"'],\s*[\"']Normal[\"']")
            if "/test" not in h and ".test.js" not in h]
    assert not hits, hits


def test_no_local_absolute_paths_in_active_code():
    files = _tracked(ACTIVE_CODE + ["backend/tests", "tests", "scripts", "Dockerfile"], (".py", ".js", "Dockerfile"))
    # test_manifest.py feeds a real historical manifest path to the relocation function on purpose
    files = [f for f in files if f.name not in ("test_manifest.py", "test_consistency.py")]
    assert not _grep(files, r"[A-Za-z]:\\\\(Users|Sathwik)|[dD]:/Sathwik|C:\\\\Users\\\\skatkam"), "machine-specific paths"


def test_no_mock_or_hardcoded_prediction_paths():
    files = _tracked(["backend/ml", "backend/api", "frontend/src/pages", "frontend/src/services"], (".py", ".js"))
    hits = [h for h in _grep(files, r"demo[_ ]mode|mock_prediction|DEMO_PROBS|0\.85,\s*0\.0|sample presets|\?\? 0\)")
            if ".test.js" not in h]
    assert not hits, hits


def test_unsupported_claims_absent_from_active_docs():
    files = _tracked(ACTIVE_DOCS, (".md",))
    bad = _grep(files, r"(?i)leak-free|zero (cross-split )?leakage|100% (external )?(TB )?sensitivity|(?<!not )clinically validated|44/44")
    assert not bad, bad


def test_frontend_reads_class_order_from_api():
    src = (REPO_ROOT / "frontend/src/pages/AnalyzePage.js").read_text(encoding="utf-8")
    assert "classScores(" in src
    for name in CLASS_NAMES:
        assert f'"{name}",' not in src
