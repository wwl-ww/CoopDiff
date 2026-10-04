#!/usr/bin/env python
"""Train a standalone listwise pose ranker on non-benchmark DockQ labels."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tricoopdiff.ranking.ranking_inference import (  # noqa: E402
    PoseRanker, listwise_ranking_loss, selected_pose_metrics,
)


def load_cases(root, expected_walkers=8):
    """Load per-case invariant features and exact DockQ labels."""
    cases = []
    for path in sorted(Path(root).glob("*.json")):
        if path.name in {"evaluation_manifest.json", "batch_status.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        walkers = payload.get("walkers", [])
        if len(walkers) != expected_walkers:
            raise ValueError(f"{path}: expected {expected_walkers} walkers")
        if not payload.get("run_config", {}).get("rank_features_saved"):
            raise ValueError(f"{path}: rank feature provenance missing")
        features = torch.tensor([row["rank_features"] for row in walkers], dtype=torch.float32)
        dockq = torch.tensor([row["dockq"] for row in walkers], dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] != expected_walkers:
            raise ValueError(f"{path}: invalid rank feature matrix")
        if not torch.isfinite(features).all() or not torch.isfinite(dockq).all():
            raise ValueError(f"{path}: non-finite training data")
        cases.append({"case": payload["case"], "features": features, "dockq": dockq})
    if not cases:
        raise ValueError(f"no pose reports found under {root}")
    return cases


@torch.no_grad()
def evaluate(model, cases):
    rows = []
    losses = []
    for case in cases:
        scores = model(case["features"])
        losses.append(float(listwise_ranking_loss(scores, case["dockq"])))
        rows.append({"case": case["case"], **selected_pose_metrics(scores, case["dockq"])})
    return {
        "listwise_loss": sum(losses) / len(losses),
        "mean_selected_dockq": sum(row["selected_dockq"] for row in rows) / len(rows),
        "mean_oracle_dockq": sum(row["oracle_dockq"] for row in rows) / len(rows),
        "mean_oracle_retention": sum(row["oracle_retention"] for row in rows) / len(rows),
        "selected_hits": sum(row["selected_hit"] for row in rows),
        "oracle_hits": sum(row["oracle_hit"] for row in rows),
        "exact_top1_matches": sum(row["selected_index"] == row["oracle_index"] for row in rows),
        "cases": len(rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and learning rate must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_cases(args.train)
    val = load_cases(args.val)
    train_ids = {case["case"].split("_", 1)[0].upper() for case in train}
    val_ids = {case["case"].split("_", 1)[0].upper() for case in val}
    if train_ids & val_ids:
        raise ValueError("train/validation ranker cases share PDB IDs")
    input_dim = train[0]["features"].shape[1]
    if any(case["features"].shape[1] != input_dim for case in train + val):
        raise ValueError("rank feature dimensions differ")

    model = PoseRanker(input_dim=input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for index in order:
            case = train[index]
            loss = listwise_ranking_loss(model(case["features"]), case["dockq"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        metrics = evaluate(model, val)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses),
               **{key: value for key, value in metrics.items() if key != "rows"}}
        history.append(row)
        if best is None or metrics["listwise_loss"] < best["metrics"]["listwise_loss"]:
            best = {"epoch": epoch, "metrics": metrics,
                    "state_dict": {key: value.detach().clone()
                                   for key, value in model.state_dict().items()}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    generator_hashes = set()
    for directory in (args.train, args.val):
        manifest = json.loads((directory / "evaluation_manifest.json").read_text())
        generator_hashes.add(manifest["checkpoint_sha256"])
    if len(generator_hashes) != 1:
        raise ValueError("ranker splits were generated by different checkpoints")
    payload = {
        "checkpoint_version": 1,
        "architecture": "standalone_pose_ranker_v1",
        "model_config": model.config,
        "state_dict": best["state_dict"],
        "selection": "minimum non-benchmark validation listwise loss",
        "best_epoch": best["epoch"],
        "best_validation": best["metrics"],
        "history": history,
        "generator_checkpoint_sha256": next(iter(generator_hashes)),
        "data": {
            "train": str(args.train.resolve()), "val": str(args.val.resolve()),
            "train_cases": len(train), "val_cases": len(val),
            "walkers_per_case": 8, "benchmark_cases": False,
        },
    }
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.out)
    report = {key: value for key, value in payload.items()
              if key not in {"state_dict", "history"}}
    report["status"] = "complete"
    report["checkpoint_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
