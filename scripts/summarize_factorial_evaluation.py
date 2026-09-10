"""Paired case-bootstrap contrasts for the fixed factorial evaluation."""

import json
from pathlib import Path

import numpy as np


RUN = Path("/tmp/coopdiff/results/overnight_root_cause_v3_run04")


def main():
    payload = json.loads((RUN / "fixed_validation_evaluation.json").read_text())
    rows = payload["rows"]
    index = {(r["chemistry"], r["torsion"], r["adaptation"], r["seed"],
              r["case"], r["time"]): r for r in rows}
    cases = sorted({r["case"] for r in rows})
    contrasts = [
        ("ccd_chemistry_legacy_frozen", ("old", "legacy", "frozen", 1729),
         ("ccd", "legacy", "frozen", 1729)),
        ("single_bond_old_frozen", ("old", "legacy", "frozen", 1729),
         ("old", "single_bond", "frozen", 1729)),
        ("ccd_single_bond_frozen", ("old", "legacy", "frozen", 1729),
         ("ccd", "single_bond", "frozen", 1729)),
        ("lora_old_legacy", ("old", "legacy", "frozen", 1729),
         ("old", "legacy", "lora", 1729)),
        ("lora_old_single", ("old", "single_bond", "frozen", 1729),
         ("old", "single_bond", "lora", 1729)),
        ("lora_ccd_legacy_seed1729", ("ccd", "legacy", "frozen", 1729),
         ("ccd", "legacy", "lora", 1729)),
        ("lora_ccd_legacy_seed2718", ("ccd", "legacy", "frozen", 2718),
         ("ccd", "legacy", "lora", 2718)),
        ("lora_ccd_single", ("ccd", "single_bond", "frozen", 1729),
         ("ccd", "single_bond", "lora", 1729)),
    ]
    output = []
    rng = np.random.default_rng(20260909)
    for name, baseline, treatment in contrasts:
        per_case = []
        for case in cases:
            values = []
            for time in (0.1, 0.5, 0.9):
                a = index[baseline + (case, time)]
                b = index[treatment + (case, time)]
                values.append({
                    "total": b["loss_total"] - a["loss_total"],
                    "translation": (b["components"]["loss_e3l_tr"] +
                                    b["components"]["loss_pol_tr"] -
                                    a["components"]["loss_e3l_tr"] -
                                    a["components"]["loss_pol_tr"]),
                    "rotation": (b["components"]["loss_e3l_rot"] +
                                 b["components"]["loss_pol_rot"] -
                                 a["components"]["loss_e3l_rot"] -
                                 a["components"]["loss_pol_rot"]),
                    "torsion": (b["components"]["loss_score_tor"] -
                                a["components"]["loss_score_tor"]),
                })
            per_case.append({key: float(np.mean([v[key] for v in values]))
                             for key in values[0]})
        result = {"name": name, "baseline": baseline, "treatment": treatment,
                  "cases": len(per_case), "delta_treatment_minus_baseline": {}}
        samples = rng.integers(0, len(per_case), size=(10000, len(per_case)))
        for component in per_case[0]:
            values = np.array([v[component] for v in per_case])
            bootstrap = values[samples].mean(1)
            result["delta_treatment_minus_baseline"][component] = {
                "mean": float(values.mean()),
                "case_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                "improved_cases": int((values < 0).sum()),
            }
        output.append(result)
    (RUN / "paired_contrasts.json").write_text(json.dumps({
        "scope": "22-case hierarchical unit; each case averages t=.1,.5,.9 first; intervals quantify this selected internal cohort, not benchmark generalization",
        "contrasts": output}, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
