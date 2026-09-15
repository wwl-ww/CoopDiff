"""Evaluate all completed factorial checkpoints on fixed paired validation paths."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
import tricoopdiff.diffdock_compat  # noqa: F401
from factorial_arm import PairedTorsionFlow, restrict_non_single
from train_entry import TernaryDataLoader, _load_score_model_args
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import (ExperimentalConsensus,
                                               ExperimentalConsensusTrainer)
from tricoopdiff.expert_factory import build_diffdock_expert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    index_dir = Path("/tmp/coopdiff/results/overnight_root_cause_v3_plan/indexes")
    expert_dir = "/tmp/coopdiff/checkpoints/score_model"
    score_args = _load_score_model_args(expert_dir)
    ccd = json.loads((index_dir / "ternary_val_ccd.json").read_text())
    ccd_paths = {e["case"]: e["ligand"] for e in ccd["entries"]}
    rows = []
    for summary_path in sorted(args.run.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        checkpoint = torch.load(summary["checkpoint"], map_location="cuda",
                                weights_only=False)
        chemistry, torsion = summary["chemistry"], summary["torsion"]
        loader = TernaryDataLoader(
            index_dir / f"ternary_val_{chemistry}.json", score_args,
            shuffle=False, device="cuda",
            cache_dir=f"/tmp/coopdiff/results/overnight_root_cause_v3_plan/cache_{chemistry}")
        batches = []
        for batch in loader:
            if torsion == "single_bond":
                batch = restrict_non_single(batch, ccd_paths[batch["case"]])
            batches.append(batch)
        expert = build_diffdock_expert(expert_dir, "cuda")
        fuse = ConsensusFuse(**checkpoint["model_config"])
        model = ExperimentalConsensus(expert, fuse)
        runtime = checkpoint["runtime_config"]
        trainer = ExperimentalConsensusTrainer(
            model, PairedTorsionFlow(), device="cuda",
            experiment_contract=checkpoint["experiment_contract"],
            lr=1e-4,
            activation_clamp=runtime["activation_clamp"],
            activation_clamp_mode=runtime["activation_clamp_mode"],
            residual_weight=runtime["residual_weight"],
            ema_decay=runtime["ema_decay"])
        trainer.load_checkpoint(summary["checkpoint"])
        with trainer.ema_parameters():
            model.eval()
            with torch.no_grad():
                for batch in batches:
                    for value in (0.1, 0.5, 0.9):
                        digest = hashlib.sha256(
                            f"factorial-eval-v1:{batch['case']}:{value}".encode()).digest()
                        seed = int.from_bytes(digest[:4], "little") % (2**31)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        coupling = trainer.diffusion.sample_coupling(
                            batch["e3_data"]["coords"], batch["poi_data"]["coords"],
                            batch["ligand_data"])
                        result = trainer.compute_loss(
                            **copy.deepcopy(batch), coupling_override=coupling,
                            t_override=torch.tensor([value], device="cuda"),
                            diagnostics=True)
                        rows.append({
                            "chemistry": summary["chemistry"],
                            "torsion": summary["torsion"],
                            "adaptation": summary["adaptation"],
                            "seed": summary["seed"], "case": batch["case"],
                            "time": value, "loss_total": float(result["loss_total"]),
                            "components": {k: float(v) for k, v in result["components"].items()},
                            "rms": result["rms"],
                        })
        del trainer, model, expert, batches
        torch.cuda.empty_cache()
    groups = []
    keys = sorted({(r["chemistry"], r["torsion"], r["adaptation"], r["seed"])
                   for r in rows})
    for key in keys:
        selected = [r for r in rows if tuple(r[k] for k in
                    ("chemistry", "torsion", "adaptation", "seed")) == key]
        component_names = sorted(selected[0]["components"])
        groups.append({"chemistry": key[0], "torsion": key[1],
                       "adaptation": key[2], "seed": key[3],
                       "states": len(selected),
                       "loss_total": float(np.mean([r["loss_total"] for r in selected])),
                       "components": {name: float(np.mean([
                           r["components"][name] for r in selected]))
                           for name in component_names}})
    output = {"scope": "22 internal validation cases x t={.1,.5,.9}; fixed case/time coupling seeds; EMA checkpoints; no benchmark cases",
              "groups": groups, "rows": rows}
    (args.run / "fixed_validation_evaluation.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    main()
