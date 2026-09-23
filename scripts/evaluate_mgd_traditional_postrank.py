"""Evaluate fixed, label-free traditional post-rank rules on MGD archives."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def methods():
    return {
        "r0_balanced_contact_minus_clash": lambda g: min(g[4], g[10]) - g[5] - g[11],
        "contact_min": lambda g: min(g[4], g[10]),
        "contact_mean": lambda g: (g[4] + g[10]) / 2,
        "contact_sum_minus_clash_sum": lambda g: g[4] + g[10] - g[5] - g[11],
        "contact_mean_minus_clash_mean": lambda g: (g[4] + g[10]) / 2 - (g[5] + g[11]) / 2,
        "contact_min_minus_clash_max": lambda g: min(g[4], g[10]) - max(g[5], g[11]),
        "distance_target_35_35_5": lambda g: -(
            abs(g[0] - 3.5) + abs(g[6] - 3.5) + abs(g[12] - 5.0)
        ),
        "distance_target_35_35": lambda g: -(abs(g[0] - 3.5) + abs(g[6] - 3.5)),
        "contact_min_distance_target": lambda g: min(g[4], g[10]) - 0.25 * (
            abs(g[0] - 3.5) + abs(g[6] - 3.5)
        ),
    }


def coordinate_features(ligand, receptor):
    """Compact classical interface terms from saved candidate coordinates."""
    distances = torch.cdist(ligand.float(), receptor.float())
    nearest = distances.min(dim=1).values
    return {
        "soft_contact": float(torch.exp(-((nearest - 3.5) / 1.5) ** 2).mean()),
        "pair_contact": float(torch.exp(-((distances - 4.0) / 2.0) ** 2).mean()),
        "clash": float(torch.sigmoid((2.0 - nearest) / 0.35).mean()),
        "contact6": float((nearest < 6).float().mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rules = methods()
    coordinate_rules = {
        "coordinate_pair_contact_minus_clash": lambda e3, poi: min(
            e3["pair_contact"], poi["pair_contact"]
        ) - e3["clash"] - poi["clash"],
        "coordinate_nearest_contact_minus_clash": lambda e3, poi: min(
            e3["contact6"], poi["contact6"]
        ) - e3["clash"] - poi["clash"],
        "coordinate_soft_contact_minus_clash": lambda e3, poi: min(
            e3["soft_contact"], poi["soft_contact"]
        ) - e3["clash"] - poi["clash"],
    }
    selected = {name: [] for name in rules}
    scores = {name: [] for name in rules}
    scores.update({name: [] for name in coordinate_rules})
    scores.update({"rank_consensus_geometry_coordinate": []})
    selected.update({name: [] for name in coordinate_rules})
    selected.update({"rank_consensus_geometry_coordinate": []})
    random_means, oracles = [], []
    cases = 0
    for archive_path in sorted(args.archives.glob("*.pt")):
        label_path = args.labels / f"{archive_path.stem}.json"
        if not label_path.exists():
            continue
        archive = torch.load(archive_path, map_location="cpu", weights_only=False)
        labels = json.loads(label_path.read_text())
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if labels.get("archive_sha256") != digest:
            raise ValueError(f"archive/label hash mismatch: {archive_path.name}")
        by_pool = {item["pool_index"]: item for item in labels["labels"]}
        rows = []
        for pair in archive["pairs"]:
            item = by_pool.get(pair["pool_index"])
            if pair["valid"] and item is not None:
                geometry = [float(value) for value in pair["final_features"]["geometry"]]
                state = pair["final"]
                e3 = coordinate_features(state["ligand"], archive["input_graphs"]["e3_data"]["coords"][0])
                poi = coordinate_features(state["ligand"], archive["input_graphs"]["poi_data"]["coords"][0])
                rows.append({"pool_index": pair["pool_index"], "geometry": geometry,
                             "e3": e3, "poi": poi, "dockq": float(item["final"]["dockq"])})
        if not rows:
            continue
        cases += 1
        values = [row["dockq"] for row in rows]
        random_means.append(sum(values) / len(values))
        oracles.append(max(values))
        for name, rule in rules.items():
            best = max(rows, key=lambda row: (rule(row["geometry"]), -row["pool_index"]))
            selected[name].append(
                {"case": archive["case"], "pool_index": best["pool_index"],
                 "score": rule(best["geometry"]), "dockq": best["dockq"]}
            )
            scores[name].append(best["dockq"])
        for name, rule in coordinate_rules.items():
            best = max(rows, key=lambda row: (rule(row["e3"], row["poi"]), -row["pool_index"]))
            selected[name].append(
                {"case": archive["case"], "pool_index": best["pool_index"],
                 "score": rule(best["e3"], best["poi"]), "dockq": best["dockq"]}
            )
            scores[name].append(best["dockq"])

        # Borda-style consensus is a standard rank aggregation operation. The
        # component rules are fixed above; no labels are consulted here.
        component_names = ["r0_balanced_contact_minus_clash", "distance_target_35_35",
                           "contact_min_distance_target", "coordinate_pair_contact_minus_clash",
                           "coordinate_nearest_contact_minus_clash"]
        totals = {row["pool_index"]: 0.0 for row in rows}
        for component in component_names:
            if component in rules:
                values_by_id = {row["pool_index"]: rules[component](row["geometry"]) for row in rows}
            else:
                values_by_id = {row["pool_index"]: coordinate_rules[component](row["e3"], row["poi"])
                                for row in rows}
            ordered = sorted(values_by_id, key=lambda index: (values_by_id[index], -index), reverse=True)
            n = len(ordered)
            for rank, index in enumerate(ordered):
                totals[index] += (n - rank) / n
        best = max(rows, key=lambda row: (totals[row["pool_index"]], -row["pool_index"]))
        selected["rank_consensus_geometry_coordinate"].append(
            {"case": archive["case"], "pool_index": best["pool_index"],
             "score": totals[best["pool_index"]], "dockq": best["dockq"]}
        )
        scores["rank_consensus_geometry_coordinate"].append(best["dockq"])

    def summary(values):
        ordered = sorted(values)
        return {
            "cases": len(values),
            "mean_final_top1_dockq": sum(values) / len(values),
            "median": ordered[len(values) // 2],
            "p75": ordered[int(0.75 * (len(values) - 1))],
            "hit_0.23": sum(value >= 0.23 for value in values),
            "hit_0.49": sum(value >= 0.49 for value in values),
        }

    report = {
        "protocol": "mgd94_final_postrank_traditional_heuristics_v2",
        "native_or_labels_used_for_selection": False,
        "cases": cases,
        "random_final_mean": sum(random_means) / len(random_means),
        "oracle_final_mean": sum(oracles) / len(oracles),
        "oracle_hit_0.23": sum(value >= 0.23 for value in oracles),
        "oracle_hit_0.49": sum(value >= 0.49 for value in oracles),
        "methods": {name: summary(values) for name, values in scores.items()},
        "selected": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["methods"], indent=2))


if __name__ == "__main__":
    main()
