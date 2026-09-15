"""Wait for a background factorial runner, then evaluate successful completion."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    while True:
        status_path = args.run / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text()).get("status")
            if status != "running":
                break
        time.sleep(30)
    if status != "complete":
        print(f"training ended with status={status}; evaluation not started", flush=True)
        return 1
    command = [sys.executable, str(Path(__file__).with_name(
        "evaluate_factorial_checkpoints.py")), "--run", str(args.run)]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
