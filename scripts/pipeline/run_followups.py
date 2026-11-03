"""Run the follow-up training jobs one after another (docs/data/run_selection_protocol.md, "Follow-up runs").

    python scripts/pipeline/run_followups.py --build v6_build_002 [--hours-per-run 8]

Each job calls scripts/pipeline/run_overnight.py with its own run name, tag (stage keys + evaluation folders)
and seed, so it is resumable and never touches earlier runs. A job that is already done is skipped.
Jobs run sequentially: two trainings at once do not fit in 8 GB of RAM.
"""
import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def jobs(build: str):
    return [
        {"tag": "_r3", "run": "v6r3_densenet121_seed43", "seed": 43, "build": build},
        {"tag": "_r4", "run": "v6r4_densenet121_seed44", "seed": 44, "build": build},
        {"tag": "_r5abl", "run": "v6r5_densenet121_seed42_nofilters", "seed": 42, "build": f"{build}_no_filters"},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--hours-per-run", type=float, default=8.0, help="wall-clock limit per run (set above the expected time)")
    ap.add_argument("--only", nargs="*", help="tags to run (default: all)")
    ap.add_argument("--phase2-epochs", type=int, required=True,
                    help="fine-tune epoch limit; set to what the selected run actually trained so every run gets the same schedule")
    args = ap.parse_args()
    for j in jobs(args.build):
        if args.only and j["tag"] not in args.only:
            continue
        if not (REPO / "data" / "derived" / j["build"]).exists():
            print(f"skip {j['run']}: build {j['build']} missing", flush=True)
            continue
        now = datetime.now()
        cmd = [sys.executable, "-u", "scripts/pipeline/run_overnight.py",
               "--build-deadline", now.isoformat(timespec="minutes"),
               "--train-until", (now + timedelta(hours=args.hours_per_run)).isoformat(timespec="minutes"),
               "--build-name", j["build"], "--run-name", j["run"], "--tag", j["tag"], "--skip-overlap",
               "--phase2-epochs", str(args.phase2_epochs), "--seed", str(j["seed"])]
        print(f"{now:%H:%M} start {j['run']} (seed {j['seed']}, build {j['build']})", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        print(f"{datetime.now():%H:%M} end {j['run']} rc={rc}", flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
