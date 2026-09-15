"""One bounded arm of the chemistry/torsion/expert factorial experiment."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from rdkit import Chem

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parent)]
import tricoopdiff.diffdock_compat  # noqa: F401
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import (ExperimentalConsensus,
                                               ExperimentalConsensusTrainer)
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.sdpi_flow import SDPIFlow
from train_entry import TernaryDataLoader, _load_score_model_args


def restrict_non_single(batch, ccd_ligand):
    """Keep tensor layout while preventing non-single axes from moving or training."""
    mol = Chem.SDMolSupplier(ccd_ligand, removeHs=True)[0]
    if mol is None:
        raise ValueError(f"cannot parse CCD ligand {ccd_ligand}")
    ligand = batch["ligand_data"]
    edges = ligand["edge_index"].T[ligand["edge_mask"]].detach().cpu()
    keep = torch.tensor([
        mol.GetBondBetweenAtoms(int(u), int(v)).GetBondType() == Chem.BondType.SINGLE
        for u, v in edges
    ], dtype=torch.bool, device=ligand["edge_mask"].device)
    selected = ligand["edge_mask"].nonzero().flatten()
    edge_mask = ligand["edge_mask"].clone()
    edge_mask[selected[~keep]] = False
    masks = ligand["mask_rotate"]
    if isinstance(masks, (list, tuple)):
        masks = masks[0]
    ligand = {**ligand, "edge_mask": edge_mask,
              "mask_rotate": np.asarray(masks)[keep.cpu().numpy()],
              "_factorial_full_edge_mask": ligand["edge_mask"].clone(),
              "_factorial_full_mask_rotate": np.asarray(masks).copy()}
    return {**batch, "ligand_data": ligand}


class PairedTorsionFlow(SDPIFlow):
    """Draw the full coupling first, then project its torsion coordinates."""

    @staticmethod
    def sample_coupling(e3_coords, poi_coords, ligand_data, no_torsion=False):
        full_mask = ligand_data.get("_factorial_full_edge_mask")
        if full_mask is None:
            return SDPIFlow.sample_coupling(
                e3_coords, poi_coords, ligand_data, no_torsion=no_torsion)
        full = {**ligand_data,
                "edge_mask": full_mask,
                "mask_rotate": ligand_data["_factorial_full_mask_rotate"]}
        coupling = SDPIFlow.sample_coupling(
            e3_coords, poi_coords, full, no_torsion=no_torsion)
        keep = ligand_data["edge_mask"][full_mask].detach().cpu().numpy()
        return {**coupling,
                "angles": coupling["angles"][keep],
                "bonds": coupling["bonds"][keep],
                "masks": np.asarray(coupling["masks"])[keep]}


def mean_validation(trainer, batches, seed):
    values = []
    with torch.random.fork_rng(devices=[0]):
        np.random.seed(seed)
        torch.manual_seed(seed)
        with trainer.ema_parameters():
            trainer.model.eval()
            with torch.no_grad():
                for batch in batches:
                    values.append(float(trainer.compute_loss(**copy.deepcopy(batch))["loss_total"]))
    return float(np.mean(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemistry", choices=("old", "ccd"), required=True)
    ap.add_argument("--torsion", choices=("legacy", "single_bond"), required=True)
    ap.add_argument("--adaptation", choices=("frozen",), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seconds", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    expert_dir = "/tmp/coopdiff/checkpoints/score_model"
    index_dir = Path("/tmp/coopdiff/results/overnight_root_cause_v3_plan/indexes")
    score_args = _load_score_model_args(expert_dir)
    cache = f"/tmp/coopdiff/results/overnight_root_cause_v3_plan/cache_{args.chemistry}"
    train = TernaryDataLoader(index_dir / f"ternary_train_{args.chemistry}.json",
                              score_args, shuffle=True, device="cuda", cache_dir=cache)
    val = TernaryDataLoader(index_dir / f"ternary_val_{args.chemistry}.json",
                            score_args, shuffle=False, device="cuda", cache_dir=cache)
    ccd = json.loads((index_dir / "ternary_train_ccd.json").read_text())
    ccd_paths = {e["case"]: e["ligand"] for e in ccd["entries"]}
    ccd_val = json.loads((index_dir / "ternary_val_ccd.json").read_text())
    ccd_paths.update({e["case"]: e["ligand"] for e in ccd_val["entries"]})

    def transform(batch):
        return (restrict_non_single(batch, ccd_paths[batch["case"]])
                if args.torsion == "single_bond" else batch)

    first = transform(train._load_one(train.entries[0]))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    expert = build_diffdock_expert(expert_dir, "cuda")
    fuse = ConsensusFuse(input_dim=first["e3_data"]["features"].shape[-1])
    contract = {"chemistry": args.chemistry, "torsion": args.torsion,
                "adaptation": args.adaptation, "seed": args.seed}
    model = ExperimentalConsensus(expert, fuse)
    trainer = ExperimentalConsensusTrainer(
        model, PairedTorsionFlow(), device="cuda", experiment_contract=contract,
        lr=1e-4, activation_clamp_mode="irrep_norm")
    val_batches = [transform(batch) for batch in val]
    # Model/adapter construction and first-time featurization consume random
    # numbers. Reset here so every arm sees identical cluster ordering, flow
    # couplings and dropout masks for a given experimental seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    deadline = started + args.seconds - 90
    history = []
    best, stale, completed = float("inf"), 0, 0
    for epoch in range(1, 26):
        losses = []
        for batch in train:
            if time.monotonic() >= deadline:
                break
            losses.append(trainer.train_step(**transform(batch))["loss_total"])
        if not losses:
            break
        completed = epoch
        validation = mean_validation(trainer, val_batches, args.seed + 50000)
        row = {"epoch": epoch, "step": trainer.global_step,
               "train_loss": float(np.mean(losses)), "val_loss": validation}
        history.append(row)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        trainer.save_checkpoint(args.out / "latest.pt", epoch, best)
        if validation < best - 1e-5:
            best, stale = validation, 0
            trainer.save_checkpoint(args.out / "best.pt", epoch, best)
        else:
            stale += 1
        print(json.dumps(row), flush=True)
        # Twenty is the normal cap; 21-25 are allowed only while validation improves.
        if stale >= 4 and epoch >= 5:
            break
        if epoch >= 20 and stale:
            break
        if time.monotonic() >= deadline:
            break
    summary = {**contract, "completed_epochs": completed,
               "steps": trainer.global_step, "best_val_loss": best,
               "elapsed_seconds": time.monotonic() - started,
               "checkpoint": str((args.out / "best.pt").resolve()),
               "history": history}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    if completed < 1 or not (args.out / "best.pt").is_file():
        raise RuntimeError("arm produced no valid checkpoint")


if __name__ == "__main__":
    main()
