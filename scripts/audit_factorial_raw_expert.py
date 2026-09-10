"""Isolate frozen raw expert proposals on the factorial's paired paths."""

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
from tricoopdiff.consensus import ConsensusCoopDiff, ConsensusFuse, BRANCHES
from tricoopdiff.expert_factory import build_diffdock_expert


RUN = Path("/tmp/coopdiff/results/overnight_root_cause_v3_run04")
INDEX = Path("/tmp/coopdiff/results/overnight_root_cause_v3_plan/indexes")


def main():
    expert_dir = "/tmp/coopdiff/checkpoints/score_model"
    score_args = _load_score_model_args(expert_dir)
    ccd_index = json.loads((INDEX / "ternary_val_ccd.json").read_text())
    ccd_paths = {e["case"]: e["ligand"] for e in ccd_index["entries"]}
    loaders = {}
    for chemistry in ("old", "ccd"):
        loaders[chemistry] = TernaryDataLoader(
            INDEX / f"ternary_val_{chemistry}.json", score_args,
            shuffle=False, device="cuda",
            cache_dir=f"/tmp/coopdiff/results/overnight_root_cause_v3_plan/cache_{chemistry}")
    first = loaders["old"]._load_one(loaders["old"].entries[0])
    expert = build_diffdock_expert(expert_dir, "cuda")
    model = ConsensusCoopDiff(
        expert, ConsensusFuse(input_dim=first["e3_data"]["features"].shape[-1])).to("cuda").eval()
    flow = PairedTorsionFlow()
    rows = []
    with torch.no_grad():
        for chemistry in ("old", "ccd"):
            base_batches = list(loaders[chemistry])
            for torsion in ("legacy", "single_bond"):
                for original in base_batches:
                    batch = copy.deepcopy(original)
                    if torsion == "single_bond":
                        batch = restrict_non_single(batch, ccd_paths[batch["case"]])
                    e3, poi, ligand = (batch[k] for k in
                                       ("e3_data", "poi_data", "ligand_data"))
                    for value in (0.1, 0.5, 0.9):
                        digest = hashlib.sha256(
                            f"factorial-eval-v1:{batch['case']}:{value}".encode()).digest()
                        seed = int.from_bytes(digest[:4], "little") % (2**31)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        coupling = flow.sample_coupling(e3["coords"], poi["coords"], ligand)
                        t = torch.tensor([value], device="cuda")
                        lig_pos, poi_pos, body, target = flow.path(
                            ligand, poi["coords"], coupling, t)
                        dt = t.new_full((1,), 1 / flow.inference_steps)
                        condition = flow.conditioning(t, dt, e3["coords"], poi["coords"])
                        sigma = flow.t_to_sigma(t, t, t)[0]
                        try:
                            output = model(
                                e3, {**poi, "coords": poi_pos},
                                {**ligand, "coords": lig_pos}, t, t,
                                r_e3_to_poi=body, diffusion_state=condition,
                                receptor_crop_cutoff=sigma * 3 + 20)
                        except (FloatingPointError, RuntimeError) as error:
                            rows.append({"chemistry": chemistry, "torsion": torsion,
                                         "case": batch["case"], "time": value,
                                         "valid": False, "error": str(error)})
                            continue
                        scales = flow.scales(e3["coords"], poi["coords"])
                        record = {"chemistry": chemistry, "torsion": torsion,
                                  "case": batch["case"], "time": value,
                                  "valid": True}
                        for index, branch in enumerate(BRANCHES):
                            proposal = output["binary_proposals"][branch]
                            record[f"{branch}_tr_mse"] = float((
                                proposal["tr"] - target[branch]["tr"] / scales[index]
                            ).square().mean())
                            record[f"{branch}_rot_mse"] = float((
                                proposal["rot"] - target[branch]["rot"] / np.pi
                            ).square().mean())
                            record[f"{branch}_rot_energy"] = float(
                                proposal["rot"].square().mean())
                        rows.append(record)
    groups = []
    for chemistry in ("old", "ccd"):
        for torsion in ("legacy", "single_bond"):
            attempted = [r for r in rows if r["chemistry"] == chemistry
                         and r["torsion"] == torsion]
            selected = [r for r in attempted if r["valid"]]
            names = [k for k in selected[0] if k.endswith(("_mse", "_energy"))]
            groups.append({"chemistry": chemistry, "torsion": torsion,
                           "attempted": len(attempted), "finite": len(selected),
                           **{k: float(np.mean([r[k] for r in selected])) for k in names}})
    output = {"scope": "frozen expert raw normalized proposals before consensus; same 22 validation cases and fixed case/time couplings as checkpoint evaluation; no LoRA",
              "groups": groups, "rows": rows}
    (RUN / "raw_expert_factorial.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
