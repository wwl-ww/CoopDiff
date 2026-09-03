#!/usr/bin/env python
"""Run resumable case-wise evaluation for a ternary JSON index."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluation_common import completed_report


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def select_entries(index_path, start=0, limit=None):
    """Load a deterministic slice from an evaluation index."""
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    if start < 0 or (limit is not None and limit < 1):
        raise ValueError("start must be non-negative and limit must be positive")
    return entries[start:] if limit is None else entries[start:start + limit]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/ternary_test.json")
    parser.add_argument("--tribrain-ckpt", required=True)
    parser.add_argument("--out-dir", default="results/deepternary_test")
    parser.add_argument("--native-root", default="data/TernaryDB_release/pdbs")
    parser.add_argument("--cache-dir", default="results/featurize_cache_deepternary")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    index = Path(args.index).resolve()
    checkpoint = str(Path(args.tribrain_ckpt))
    entries = select_entries(index, args.start, args.limit)
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    for position, entry in enumerate(entries, start=1):
        case = entry["case"]
        report_path = output / f"{case}.json"
        if not args.no_resume and completed_report(report_path, case, checkpoint):
            print(f"[{position}/{len(entries)}] SKIP completed {case}", flush=True)
            continue
        command = [
            sys.executable, str(_PROJECT_ROOT / "scripts/evaluate_entry.py"),
            "--index", str(index), "--case", case,
            "--native-root", args.native_root,
            "--tribrain-ckpt", checkpoint,
            "--cache-dir", args.cache_dir,
            "--n-walkers", str(args.n_walkers),
            "--steps", str(args.steps), "--seed", str(args.seed),
            "--device", args.device, "--out", str(report_path),
        ]
        print(f"[{position}/{len(entries)}] RUN {case}", flush=True)
        result = subprocess.run(command, cwd=_PROJECT_ROOT)
        if result.returncode:
            failures.append({"case": case, "exit_code": result.returncode})
            print(f"[{position}/{len(entries)}] FAIL {case}: {result.returncode}",
                  flush=True)

    status = {
        "requested_cases": len(entries),
        "failed_cases": failures,
        "complete": not failures,
    }
    (output / "batch_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8",
    )
    print(json.dumps(status, indent=2), flush=True)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
