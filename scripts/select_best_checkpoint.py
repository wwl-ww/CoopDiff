#!/usr/bin/env python
"""Verify that a completed training log and best checkpoint agree."""

import argparse
import json
import re
from pathlib import Path

from evaluation_common import sha256_file, validate_checkpoint_payload


EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+).*val_loss=([0-9.eE+-]+)")
SAVED_RE = re.compile(r"Saved best \(val_loss=([0-9.eE+-]+)\)")


def parse_training_log(path):
    """Extract epoch validation losses and recorded best saves."""
    epochs = []
    saves = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            epochs.append({"epoch": int(match.group(1)),
                           "total_epochs": int(match.group(2)),
                           "val_loss": float(match.group(3))})
        saved = SAVED_RE.search(line)
        if saved:
            saves.append(float(saved.group(1)))
    return epochs, saves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    epochs, saves = parse_training_log(args.log)
    if len(epochs) != args.expected_epochs or epochs[-1]["epoch"] != args.expected_epochs:
        raise SystemExit(
            f"FATAL: found {len(epochs)} completed epochs; expected {args.expected_epochs}"
        )
    best = min(epochs, key=lambda item: item["val_loss"])
    if not saves or abs(saves[-1] - best["val_loss"]) > 5e-4:
        raise SystemExit("FATAL: last Saved best marker is not minimum validation loss")
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu")
    validate_checkpoint_payload(payload)
    if args.steps_per_epoch is not None:
        expected_step = best["epoch"] * args.steps_per_epoch
        if payload.get("global_step") != expected_step:
            raise SystemExit(
                f"FATAL: checkpoint global_step={payload.get('global_step')}; "
                f"expected {expected_step} for best epoch {best['epoch']}"
            )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_version": payload["checkpoint_version"],
        "global_step": payload.get("global_step"),
        "best_epoch": best["epoch"],
        "best_validation_loss": best["val_loss"],
        "completed_epochs": len(epochs),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
