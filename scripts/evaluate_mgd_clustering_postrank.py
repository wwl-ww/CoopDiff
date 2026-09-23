"""Evaluate fixed pose/contact clustering selectors on archived MGD poses."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cluster_medoid(rows, distances, threshold):
    if len(rows) == 1:
        return rows[0]
    tree = linkage(squareform(distances, checks=False), method="average")
    groups = fcluster(tree, t=threshold, criterion="distance")
    counts = np.bincount(groups)[1:]
    largest = np.flatnonzero(counts == counts.max()) + 1
    # Tie between largest clusters: use the cluster with the best mean distance
    # to its medoid, then pool index for complete determinism.
    candidates = []
    for group in largest:
        ids = np.flatnonzero(groups == group)
        medoid = ids[np.argmin(distances[np.ix_(ids, ids)].sum(axis=1))]
        candidates.append((distances[np.ix_(ids, ids)].sum(), rows[medoid]["pool_index"], medoid))
    return rows[min(candidates)[2]]


def pose_distances(rows):
    vectors = np.stack([row["poi"].numpy().reshape(-1) for row in rows])
    return np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2) / np.sqrt(vectors.shape[1] / 3)


def contact_distances(rows):
    e3 = rows[0]["e3_reference"]
    fingerprints = []
    for row in rows:
        poi = row["poi"]
        # C-alpha interface contact graph in the fixed receptor frame.
        contact = (torch.cdist(e3.float(), poi.float()) < 8.0).flatten().numpy()
        fingerprints.append(contact)
    fingerprints = np.stack(fingerprints)
    intersection = fingerprints @ fingerprints.T
    sizes = fingerprints.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    return 1.0 - intersection / np.maximum(union, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selectors = {
        "pose_cluster": lambda rows: cluster_medoid(rows, pose_distances(rows), 4.0),
        "contact_cluster": lambda rows: cluster_medoid(rows, contact_distances(rows), 0.35),
    }
    results = {name: [] for name in selectors}
    results["geometry_filter50_pose_cluster"] = []
    results["geometry_filter50_contact_cluster"] = []
    random_means, oracles = [], []
    for archive_path in sorted(args.archives.glob("*.pt")):
        label_path = args.labels / f"{archive_path.stem}.json"
        if not label_path.exists():
            continue
        archive = torch.load(archive_path, map_location="cpu", weights_only=False)
        labels = json.loads(label_path.read_text())
        if labels.get("archive_sha256") != hashlib.sha256(archive_path.read_bytes()).hexdigest():
            raise ValueError(f"archive/label hash mismatch: {archive_path.name}")
        by_pool = {item["pool_index"]: item for item in labels["labels"]}
        e3 = archive["input_graphs"]["e3_data"]["coords"][0]
        rows = []
        for pair in archive["pairs"]:
            item = by_pool.get(pair["pool_index"])
            if not pair["valid"] or item is None:
                continue
            state = pair["final"]
            geometry = [float(x) for x in pair["final_features"]["geometry"]]
            rows.append({"pool_index": pair["pool_index"], "poi": state["poi"],
                         "e3_reference": e3, "geometry_score": min(geometry[4], geometry[10])
                         - 0.5 * (geometry[5] + geometry[11]),
                         "dockq": float(item["final"]["dockq"])})
        if not rows:
            continue
        values = [row["dockq"] for row in rows]
        random_means.append(sum(values) / len(values))
        oracles.append(max(values))
        for name, selector in selectors.items():
            best = selector(rows)
            results[name].append({"case": archive["case"], "pool_index": best["pool_index"], "dockq": best["dockq"]})
        ranked = sorted(rows, key=lambda row: (-row["geometry_score"], row["pool_index"]))
        filtered = ranked[: max(1, len(ranked) // 2)]
        for name, selector in (("geometry_filter50_pose_cluster", selectors["pose_cluster"]),
                               ("geometry_filter50_contact_cluster", selectors["contact_cluster"])):
            best = selector(filtered)
            results[name].append({"case": archive["case"], "pool_index": best["pool_index"], "dockq": best["dockq"]})

    def summary(items):
        values = [item["dockq"] for item in items]
        return {"cases": len(values), "mean_final_top1_dockq": sum(values) / len(values),
                "hit_0.23": sum(x >= 0.23 for x in values), "hit_0.49": sum(x >= 0.49 for x in values)}

    report = {"protocol": "mgd94_final_postrank_clustering_v1", "native_or_labels_used_for_selection": False,
              "random_final_mean": sum(random_means) / len(random_means),
              "oracle_final_mean": sum(oracles) / len(oracles),
              "methods": {name: summary(items) for name, items in results.items()}, "selected": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["methods"], indent=2))


if __name__ == "__main__":
    main()
