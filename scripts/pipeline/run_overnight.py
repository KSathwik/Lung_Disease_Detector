"""Unattended, resumable pipeline: downloads -> extract -> build V6 -> train -> evaluate.

    python scripts/pipeline/run_overnight.py --build-deadline 2026-11-03T03:30 --train-until 2026-11-03T10:30

Every stage records its status in artifacts/execution_state.json and is skipped on re-run once done,
so the command can simply be started again after an interruption. Logs: artifacts/pipeline/pipeline.log.

- downloads: waits until no acquire.py process is running, or until --build-deadline (local time);
  then uses whatever completed (partial *.part files are ignored, never extracted).
- kaggle: if ~/.kaggle/kaggle.json exists, also fetches the COVID-19 Radiography and Rahman TB archives.
- extract: completed NIH tarballs, Kermany zip, TBX11K zip, Kaggle zips -> next to the archive (existing files kept).
- overlap: Montgomery vs every development source (scripts/data/check_external_overlap.py) when the Rahman TB archive exists.
- build: data/derived/v6_build_001 (never overwritten).
- train: lungai.training.train_v6 (resumes from its last epoch), time-boxed to finish by --train-until.
- evaluate: val, then test and external once (--final).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
RAW = REPO / "data" / "raw"
BUILD = REPO / "data" / "derived" / "v6_build_001"
RUN = "v6_densenet121_seed42"
STATE = REPO / "artifacts" / "execution_state.json"
LOG = REPO / "artifacts" / "pipeline" / "pipeline.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


TAG = ""  # suffix for stage keys and evaluation folders, so a second build/run keeps the first one's record


def set_stage(key: str, status: str, **info) -> None:
    s = state()
    s["stages"].setdefault(key + TAG, {}).update(status=status, updated=datetime.now(timezone.utc).isoformat(timespec="seconds"), **info)
    STATE.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")


def done(key: str) -> bool:
    return state()["stages"].get(key + TAG, {}).get("status") == "done"


def run(cmd: list, **kw) -> int:
    log("RUN " + " ".join(str(c) for c in cmd))
    with open(LOG, "a", encoding="utf-8") as f:
        return subprocess.run([str(c) for c in cmd], cwd=REPO, stdout=f, stderr=subprocess.STDOUT, **kw).returncode


def acquire_running() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*acquire.py*' }).Count"],
                         capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def wait_for_downloads(deadline: datetime) -> None:
    while acquire_running() and datetime.now() < deadline:
        log("downloads still running; waiting 5 min")
        time.sleep(300)
    log("downloads finished" if not acquire_running() else "build deadline reached; continuing with completed downloads")


def kaggle() -> None:
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        log("no Kaggle token: skipping COVID-19 Radiography and Rahman TB")
        return
    for sid in ("covid_radiography", "rahman_tb"):
        if not (RAW / sid).exists():
            run([PY, "scripts/data/acquire.py", sid, "--execute"])


def extract() -> dict:
    out = {}
    for tgz in sorted((RAW / "nih_cxr14").glob("images_*.tar.gz")):
        marker = tgz.with_suffix(".extracted")
        if marker.exists():
            continue
        log(f"extract {tgz.name}")
        with tarfile.open(tgz, "r:gz") as t:
            for m in t:
                if m.isfile() and not (tgz.parent / m.name).exists():
                    t.extract(m, tgz.parent, filter="data")
        marker.write_text("ok")
        out[tgz.name] = "extracted"
    for z in [*RAW.glob("kermany/*.zip"), *RAW.glob("tbx11k/*.zip"), *RAW.glob("covid_radiography/*.zip"), *RAW.glob("rahman_tb/*.zip")]:
        marker = z.with_suffix(".extracted")
        if marker.exists():
            continue
        log(f"extract {z}")
        with zipfile.ZipFile(z) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"corrupt member {bad} in {z}")
            for m in zf.infolist():
                if not m.is_dir() and not (z.parent / m.filename).exists():
                    zf.extract(m, z.parent)
        marker.write_text("ok")
        out[z.name] = "extracted"
    return out


def available_sources() -> list:
    checks = {
        "nih_cxr14": lambda: (RAW / "nih_cxr14" / "Data_Entry_2017_v2020.csv").exists() and any((RAW / "nih_cxr14").glob("*.extracted")),
        "montgomery": lambda: len(list((RAW / "montgomery" / "CXR_png").glob("*.png"))) == 138,
        "shenzhen": lambda: len(list((RAW / "shenzhen" / "CXR_png").glob("*.png"))) >= 600,
        "kermany": lambda: any((RAW / "kermany").glob("*.extracted")),
        "tbx11k": lambda: any((RAW / "tbx11k").glob("*.extracted")),
        "cohen_covid": lambda: (RAW / "cohen_covid" / "metadata.csv").exists(),
        "covid_radiography": lambda: any((RAW / "covid_radiography").glob("*.extracted")),
    }
    return [s for s, ok in checks.items() if (RAW / s).exists() and ok()]


def main() -> int:
    global BUILD, RUN, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-deadline", required=True, help="local time, e.g. 2026-11-03T03:30")
    ap.add_argument("--train-until", required=True, help="local time by which training must stop")
    ap.add_argument("--build-name", default=BUILD.name, help="folder under data/derived/ (never overwritten)")
    ap.add_argument("--run-name", default=RUN)
    ap.add_argument("--tag", default="", help="suffix for stage keys and evaluation folders, e.g. _r2")
    ap.add_argument("--skip-overlap", action="store_true",
                    help="skip the standalone Montgomery overlap scan (the builder still checks external vs development)")
    ap.add_argument("--seed", type=int, default=None, help="training seed (default: config)")
    ap.add_argument("--phase2-epochs", type=int, default=8, help="maximum fine-tune epochs (early stopping and the time box still apply)")
    args = ap.parse_args()
    BUILD, RUN, TAG = BUILD.parent / args.build_name, args.run_name, args.tag
    build_deadline = datetime.fromisoformat(args.build_deadline)
    train_until = datetime.fromisoformat(args.train_until)

    if not done("2_recover_data"):
        wait_for_downloads(build_deadline)
        kaggle()
        set_stage("2_recover_data", "in_progress", extracted=extract())
        set_stage("2_recover_data", "done", sources_available=available_sources())

    if not done("3_build_v6"):
        srcs = available_sources()
        if (RAW / "rahman_tb").exists() and not args.skip_overlap:
            run([PY, "scripts/data/check_external_overlap.py", "--external", RAW / "montgomery",
                 "--candidates", *[RAW / s for s in srcs if s != "montgomery"], RAW / "rahman_tb", "--out", REPO / "artifacts" / "data_quality" / "external_overlap"])
        if not BUILD.exists():
            set_stage("3_build_v6", "in_progress", sources=srcs)
            rc = run([PY, "scripts/data/build_v6_manifest.py", "--raw-root", RAW, "--out", BUILD, "--sources", *srcs])
            if rc != 0:
                set_stage("3_build_v6", "failed", returncode=rc)
                return rc
        set_stage("3_build_v6", "done", build=str(BUILD.relative_to(REPO)))

    if not done("4_train"):
        hours = max(0.5, (train_until - datetime.now()).total_seconds() / 3600 - 0.75)  # leave time for evaluation
        set_stage("4_train", "in_progress", run=RUN, max_hours=round(hours, 2))
        rc = run([PY, "-m", "lungai.training.train_v6", "--build", BUILD, "--run-name", RUN, "--max-hours", f"{hours:.2f}",
                  "--phase2-epochs", args.phase2_epochs, *(["--seed", args.seed] if args.seed is not None else [])])
        if rc != 0:
            set_stage("4_train", "failed", returncode=rc)
            return rc
        set_stage("4_train", "done", run=RUN)

    if not done("5_evaluate"):
        ckpt = REPO / "runs" / RUN / "model.h5"
        reports = {}
        for split, final in (("val", []), ("test", ["--final"]), ("external", ["--final"])):
            out = REPO / "artifacts" / "evaluation_reports" / f"v6{TAG}_{split}"
            if out.exists():
                reports[split] = str(out.relative_to(REPO))
                continue
            rc = run([PY, "-m", "lungai.evaluation.evaluate_v6", "--build", BUILD, "--checkpoint", ckpt, "--split", split, *final, "--out", out])
            if rc != 0:
                set_stage("5_evaluate", "failed", split=split, returncode=rc)
                return rc
            reports[split] = str(out.relative_to(REPO))
        set_stage("5_evaluate", "done", reports=reports)
    log("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
