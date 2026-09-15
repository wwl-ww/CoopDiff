"""Create deterministic paired old/CCD indexes for the overnight factorial."""

import hashlib
import json
from pathlib import Path


ROOT_CAUSE = Path("/tmp/coopdiff/results/root_cause_v2")
OUT = Path("/tmp/coopdiff/results/overnight_root_cause_v3_plan/indexes")


def main():
    manifest = json.loads((ROOT_CAUSE / "ccd_restoration_manifest.json").read_text())
    restored = {(r["split"], r["case"]): r for r in manifest["records"]
                if r["status"] == "restored"}
    original = {
        split: json.loads(Path(f"/tmp/coopdiff/data/{split}.json").read_text())
        for split in ("ternary_train", "ternary_val")
    }
    OUT.mkdir(parents=True, exist_ok=False)
    selected = {}
    for split, count in (("ternary_train", 128), ("ternary_val", 22)):
        candidates = [e for e in original[split]["entries"]
                      if (split, e["case"]) in restored]
        candidates.sort(key=lambda e: hashlib.sha256(
            f"factorial-v1:{split}:{e['case']}".encode()).hexdigest())
        entries = candidates[:count]
        selected[split] = [e["case"] for e in entries]
        for chemistry in ("old", "ccd"):
            materialized = []
            for entry in entries:
                item = dict(entry)
                if chemistry == "ccd":
                    item["ligand"] = restored[(split, entry["case"])]["entry"]["ligand"]
                materialized.append(item)
            payload = {
                "entries": materialized,
                "clusters": [{"cluster_id": e["case"], "items": [e["case"]],
                              "representative": [e["case"]]} for e in materialized],
                "metadata": {"protocol": "factorial-v1 deterministic paired cohort",
                             "chemistry": chemistry, "split": split},
                "rejected": [],
            }
            (OUT / f"{split}_{chemistry}.json").write_text(json.dumps(payload, indent=2))
    (OUT / "selection.json").write_text(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
