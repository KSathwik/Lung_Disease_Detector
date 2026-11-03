"""Acquire raw source datasets listed in scripts/data/sources.json — dry run unless --execute.

    python scripts/data/acquire.py --list                       # what each source needs
    python scripts/data/acquire.py nih_cxr14                    # dry run: print the plan, download nothing
    python scripts/data/acquire.py nih_cxr14 --execute          # download (resumable) into <root>/raw/<source>/
    python scripts/data/acquire.py covid_radiography --execute  # uses the official `kaggle` CLI and YOUR credentials

Rules enforced here:
- Only URLs from the registry are used ('nlm_index' sources: files listed in the publisher's own index.json). 'manual' sources (login, registration or credentialing) are never
  fetched; the script prints the official page and the target folder instead.
- Kaggle sources use the `kaggle` CLI with the credentials already configured on this machine
  (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). Nothing is stored or passed by this script.
- Existing files are never overwritten. Partial downloads (*.part) resume with HTTP Range when supported.
- Every completed file is recorded in <root>/raw/_provenance/<source>.jsonl with URL, UTC time, size and SHA-256,
  and checked against a publisher checksum when the registry has one.

<root> is LUNGAI_RAW_ROOT or --root (default: <repo>/data). Raw data lives in <root>/raw/<source>/; derived data
must go elsewhere (see scripts/data/build_v6_manifest.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name("sources.json")
CHUNK = 1 << 20


def load_registry() -> Dict[str, dict]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {s["id"]: s for s in data["sources"]}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def record(prov_dir: Path, source_id: str, entry: dict) -> None:
    prov_dir.mkdir(parents=True, exist_ok=True)
    with open(prov_dir / f"{source_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def completed_files(prov_dir: Path, source_id: str) -> Dict[str, str]:
    """file name -> SHA-256 for downloads already logged as complete."""
    log = prov_dir / f"{source_id}.jsonl"
    if not log.exists():
        return {}
    out = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if "file" in e and "sha256" in e:
            out[e["file"]] = e["sha256"]
    return out


def download(url: str, dest: Path, expected_sha256: Optional[str] = None, attempts: int = 20) -> dict:
    """Resumable HTTPS download to dest (never overwrites a completed file)."""
    if not url.lower().startswith("https://"):
        raise ValueError(f"refusing non-HTTPS URL: {url}")
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; not overwriting (delete it yourself if it is corrupt)")
    part = dest.with_name(dest.name + ".part")
    expected_total: Optional[int] = None
    resumed = False
    total = None
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if expected_total is not None and have == expected_total:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "lungai-acquire/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if ctype.startswith("text/html"):
                    # e.g. a quota/login page instead of the file: never let it replace data
                    raise PermissionError(f"server returned an HTML page instead of the file ({ctype}); not saved")
                partial = bool(have) and resp.status == 206
                resumed = resumed or partial
                total = resp.headers.get("Content-Length")
                if total is not None:
                    expected_total = int(total) + (have if partial else 0)
                with open(part, "ab" if partial else "wb") as out:
                    shutil.copyfileobj(resp, out, CHUNK)
            got = part.stat().st_size
            if expected_total is None or got == expected_total:
                break
            raise IOError(f"connection closed early: {got:,} of {expected_total:,} bytes")
        except PermissionError:
            raise
        except (OSError, urllib.error.URLError) as e:  # timeouts, resets, truncation: resume from .part
            if attempt == attempts:
                raise IOError(f"{dest.name}: giving up after {attempts} attempts ({e}); partial data kept in {part.name}") from e
            print(f"  {dest.name}: attempt {attempt} failed ({e}); resuming in {min(60, 5 * attempt)} s", flush=True)
            time.sleep(min(60, 5 * attempt))
    if expected_total is not None and part.stat().st_size != expected_total:
        raise IOError(f"{dest.name}: size {part.stat().st_size:,} != announced {expected_total:,}; kept as {part.name}")
    digest = sha256_of(part)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for {dest.name}: got {digest}, publisher says {expected_sha256}; kept as {part}")
    part.rename(dest)
    return {"file": dest.name, "url": url, "bytes": dest.stat().st_size, "sha256": digest,
            "publisher_sha256_checked": bool(expected_sha256), "resumed": bool(resumed),
            "content_length_header": total, "size_verified": True}


def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def nlm_listing(index_url: str, include: list) -> list:
    """Walk the publisher's S3-style index.json tree; return [(relative_key, url, size, md5_or_None)]."""
    base = index_url.rsplit("/", 1)[0] + "/"
    out = []

    def walk(prefix_rel: str):
        with urllib.request.urlopen(urllib.request.Request(base + prefix_rel + "index.json",
                                    headers={"User-Agent": "lungai-acquire/1.0"}), timeout=60) as r:
            d = json.load(r)
        for c in d.get("Contents") or []:
            key = c["Key"]
            if key in ("index.json", "index.html"):
                continue
            etag = c.get("ETag", "").strip('"')
            # A single-part S3 ETag is the MD5 of the object; multipart ETags contain '-' and are not
            out.append((prefix_rel + key, base + prefix_rel + key, int(c["Size"]), None if "-" in etag else etag))
        for p in d.get("CommonPrefixes") or []:
            sub = prefix_rel + p["Prefix"]
            if any(sub.startswith(i) or i.startswith(sub) for i in include):
                walk(sub)

    walk("")
    return [f for f in out if "/" not in f[0] or any(f[0].startswith(i) for i in include)]


def nlm_download(src: dict, target: Path, prov: Path, started: str, workers: int = 1) -> int:
    from concurrent.futures import ThreadPoolExecutor
    import threading
    files = nlm_listing(src["index_url"], src["include_prefixes"])
    counter = {"done": 0, "skipped": 0}
    lock = threading.Lock()

    def one(item):
        rel, url, size, md5 = item
        dest = target / rel
        if dest.exists():
            if dest.stat().st_size == size and (md5 is None or _md5_of(dest) == md5):
                with lock:
                    counter["skipped"] += 1
                return
            raise FileExistsError(f"{dest} exists but does not match the publisher index; not overwriting")
        dest.parent.mkdir(parents=True, exist_ok=True)
        info = download(url, dest)
        got_md5 = _md5_of(dest)
        if md5 and got_md5 != md5:
            bad = dest.with_name(dest.name + ".md5_mismatch")
            dest.rename(bad)
            raise ValueError(f"MD5 mismatch for {rel}: publisher {md5}, got {got_md5}; kept as {bad.name}")
        if info["bytes"] != size:
            raise ValueError(f"size mismatch for {rel}: publisher {size}, got {info['bytes']}")
        with lock:
            record(prov, src["id"], {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "source": src["id"], "official_url": src["official_url"], "relative_path": rel,
                                     "publisher_md5_etag": md5, "md5_verified": bool(md5), **info})
            counter["done"] += 1
            if counter["done"] % 25 == 0:
                print(f"  {counter['done']} downloaded, {counter['skipped']} already present / {len(files)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, files))
    print(f"done: {counter['done']} downloaded, {counter['skipped']} already present and verified, {len(files)} listed by the publisher")
    return 0


def gdrive_download(file_id: str, dest: Path) -> dict:
    """Official Google Drive share (large file): uses the documented confirm=t download endpoint; resumable."""
    return download(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t", dest)


def git_clone(repo: str, commit_or_branch: str, dest: Path) -> dict:
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty; not overwriting")
    subprocess.run(["git", "clone", "--depth", "1", "--branch", commit_or_branch, repo, str(dest)], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return {"repo": repo, "branch": commit_or_branch, "commit": head}


def kaggle_download(ref: str, dest_dir: Path) -> dict:
    # The venv's Scripts dir is not on PATH when python.exe is invoked by full path.
    exe = shutil.which("kaggle") or shutil.which("kaggle", path=str(Path(sys.executable).parent))
    if exe is None:
        raise RuntimeError("the official `kaggle` CLI is not installed (pip install kaggle) — or download manually from the Kaggle page")
    if any(dest_dir.iterdir()) if dest_dir.exists() else False:
        raise FileExistsError(f"{dest_dir} is not empty; not overwriting")
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Credentials are read by the kaggle CLI itself from the user's own configuration.
    subprocess.run([exe, "datasets", "download", "-d", ref, "-p", str(dest_dir)], check=True)
    meta = subprocess.run([exe, "datasets", "metadata", ref, "-p", str(dest_dir)], capture_output=True, text=True)
    files = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in sorted(dest_dir.glob("*.zip"))]
    return {"kaggle_ref": ref, "files": files, "metadata_saved": meta.returncode == 0}


def plan(src: dict, root: Path) -> str:
    target = root / "raw" / src["id"]
    lines = [f"[{src['id']}] {src['name']}",
             f"  official: {src['official_url']}",
             f"  license : {src['license']}",
             f"  access  : {src['access']} -> method '{src['acquisition']}'",
             f"  target  : {target}"]
    if src["acquisition"] == "direct":
        for f in src.get("files", []):
            lines.append(f"  file    : {f['name']} <- {f['url']} ({f.get('size_note', 'size unknown')})")
    elif src["acquisition"] == "nlm_index":
        lines.append(f"  index   : {src['index_url']} (folders: {', '.join(src['include_prefixes'])}); MD5 checked against publisher ETags")
    elif src["acquisition"] == "kaggle_cli":
        lines.append(f"  kaggle  : kaggle datasets download -d {src['kaggle_ref']} -p {target}")
    else:
        lines.append("  manual  : download from the official page after accepting its terms; place files under the target folder")
    if src.get("risk"):
        lines.append(f"  RISK    : {src['risk']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="source id from sources.json")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--execute", action="store_true", help="actually download (default is a dry run)")
    ap.add_argument("--workers", type=int, default=1, help="parallel downloads for 'nlm_index' sources")
    ap.add_argument("--only", nargs="+", default=None, help="restrict a 'direct' source to these file names")
    ap.add_argument("--root", type=Path, default=Path(os.getenv("LUNGAI_RAW_ROOT", REPO_ROOT / "data")))
    args = ap.parse_args(argv)
    reg = load_registry()

    if args.list or not args.source:
        for s in reg.values():
            print(plan(s, args.root), end="\n\n")
        return 0
    if args.source not in reg:
        ap.error(f"unknown source '{args.source}'. Known: {', '.join(reg)}")
    src = reg[args.source]
    print(plan(src, args.root))
    if not args.execute:
        print("\nDry run only. Re-run with --execute after the plan has been approved.")
        return 0

    target = args.root / "raw" / src["id"]
    prov = args.root / "raw" / "_provenance"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if src["acquisition"] == "manual":
        print(f"\n'{src['id']}' requires manual download (login, registration or credentialing). Nothing was downloaded.")
        return 3
    if src["acquisition"] == "direct":
        target.mkdir(parents=True, exist_ok=True)
        logged = completed_files(prov, src["id"])
        for f in src["files"]:
            if args.only and f["name"] not in args.only:
                continue
            dest = target / f["name"]
            if dest.exists() and logged.get(f["name"]) == sha256_of(dest):
                print(f"skip: {f['name']} already downloaded and matches its logged SHA-256")
                continue
            info = download(f["url"], dest, f.get("sha256"))
            record(prov, src["id"], {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "source": src["id"], "official_url": src["official_url"], **info})
            print(f"done: {info['file']} {info['bytes']:,} bytes sha256 {info['sha256']}")
    elif src["acquisition"] == "gdrive":
        target.mkdir(parents=True, exist_ok=True)
        dest = target / src["gdrive_file"]
        if dest.exists() and completed_files(prov, src["id"]).get(dest.name) == sha256_of(dest):
            print(f"skip: {dest.name} already downloaded")
            return 0
        info = gdrive_download(src["gdrive_id"], dest)
        record(prov, src["id"], {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "source": src["id"], "official_url": src["official_url"], **info})
        print(json.dumps(info, indent=2))
    elif src["acquisition"] == "git":
        info = git_clone(src["git_repo"], src["git_branch"], target)
        record(prov, src["id"], {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "source": src["id"], "official_url": src["official_url"], **info})
        print(json.dumps(info, indent=2))
    elif src["acquisition"] == "nlm_index":
        return nlm_download(src, target, prov, started, workers=args.workers)
    elif src["acquisition"] == "kaggle_cli":
        info = kaggle_download(src["kaggle_ref"], target)
        record(prov, src["id"], {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "source": src["id"], "official_url": src["official_url"], **info})
        print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
