"""Ten-hour GPU3-only sequential factorial runner with atomic status updates."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GPU3 = "GPU-3ac265b1-130f-cf53-b680-8730370a5881"
RUN = Path(os.environ.get("FACTORIAL_RUN_DIR",
                          "/tmp/coopdiff/results/overnight_root_cause_v3_run01"))
ARMS = [(c, t, a) for c in ("old", "ccd") for t in ("legacy", "single_bond")
        for a in ("frozen", "lora")]


def write_status(payload):
    temporary = RUN / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(RUN / "status.json")


def main():
    if RUN.exists():
        raise FileExistsError(RUN)
    RUN.mkdir(parents=True)
    plan = json.loads(Path(
        "/tmp/coopdiff/results/overnight_root_cause_v3_plan/plan.json").read_text())
    (RUN / "plan.json").write_text(json.dumps(plan, indent=2))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU3,
               HF_HOME="/tmp/coopdiff/results/huggingface",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TRICOOPDIFF_CACHE_DIR="/tmp/coopdiff/results/runtime_cache/diffdock",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", CC="/usr/bin/gcc-11")
    env.pop("GCC_ROOT", None)
    started = time.monotonic()
    hard_deadline = started + 36000
    state = {"status": "running", "pid": os.getpid(), "gpu_uuid": GPU3,
             "started_unix": time.time(), "paired_rng_reset": True, "arms": []}
    write_status(state)
    # First seed covers the full 2x2x2 factorial. Two preregistered main arms get seed 2.
    jobs = [(arm, 1729) for arm in ARMS] + [
        (("ccd", "legacy", "frozen"), 2718),
        (("ccd", "legacy", "lora"), 2718),
    ]
    for index, ((chemistry, torsion, adaptation), seed) in enumerate(jobs):
        remaining = hard_deadline - time.monotonic()
        reserve_report = 600
        if remaining <= reserve_report + 120:
            state["arms"].append({"name": "remaining", "status": "budget_exhausted"})
            break
        # Initial factorial arms get up to 40 minutes; second-seed arms share remaining time.
        seconds = min(2400 if index < 8 else 2700,
                      int(remaining - reserve_report))
        name = f"{chemistry}_{torsion}_{adaptation}_seed{seed}"
        output = RUN / name
        argv = [sys.executable, str(ROOT / "scripts/factorial_arm.py"),
                "--chemistry", chemistry, "--torsion", torsion,
                "--adaptation", adaptation, "--seed", str(seed),
                "--seconds", str(seconds), "--out", str(output)]
        record = {"name": name, "status": "running", "seconds": seconds,
                  "started_unix": time.time(), "argv": argv}
        state["arms"].append(record)
        write_status(state)
        with (RUN / f"{name}.log").open("x") as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=seconds + 30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
                code = 124
        record.update(status="complete" if code == 0 else "failed",
                      exit_code=code, ended_unix=time.time())
        write_status(state)
        if code != 0:
            # An implementation/data failure invalidates later paired comparisons.
            state["status"] = "failed"
            state["failed_arm"] = name
            write_status(state)
            return 1
    summaries = []
    for item in state["arms"]:
        path = RUN / item["name"] / "summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text()))
    (RUN / "summary.json").write_text(json.dumps({
        "scope": "paired internal cohort; validation loss is the selection metric; no benchmark tuning",
        "completed_arms": len(summaries), "arms": summaries}, indent=2))
    state["status"] = "complete"
    state["ended_unix"] = time.time()
    write_status(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
