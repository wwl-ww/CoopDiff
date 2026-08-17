#!/usr/bin/env python
"""
Orientation-distinguishability probe for SE3GeometryEncoder (Cause 2 hypothesis).

Hypothesis: the SE3 geometry encoder
    per-node MLP Linear(6,64)->SiLU->Linear(64,64)  ->  mean+max pool  ->  Linear(128,64)
is permutation-invariant and may be ORIENTATION-BLIND for the POI rotation score.
The translation score needs only the first moment (POI centroid vs E3) which mean
pooling captures; the rotation score needs the relative ORIENTATION of the POI
point cloud, a second-order (covariance) property.

This script:
  1. Builds a fixed E3 point set (type 0), a fixed anisotropic POI point set
     (type 1), and a few PROTAC nodes (type 2) -- matching build_tribrain_graph
     semantics (node_coords = cat[E3, POI, PROTAC], node_types = 0/1/2).
  2. Rotates the POI subset about its OWN centroid (exactly how add_noise_se3
     applies rot_update: poi_centered = poi_pos - poi_center).
  3. Feeds the result through the real SE3GeometryEncoder and measures whether
     h_geo distinguishes rotations, using translation as a positive control and
     the POI covariance upper-triangle as a gold-standard orientation descriptor.

Self-contained: loads tricoopdiff/tribrain.py directly (no package __init__), CPU-only.
"""

import importlib.util
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = "/data/run01/scwb792/mwj/CoopDiff"
TRIBRAIN_PATH = os.path.join(PROJECT_ROOT, "tricoopdiff", "tribrain.py")

# ---------------------------------------------------------------------------


def load_tribrain():
    spec = importlib.util.spec_from_file_location("tribrain_probe", TRIBRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tribrain = load_tribrain()
SE3GeometryEncoder = tribrain.SE3GeometryEncoder


def make_clouds(seed=0):
    """Anisotropic E3 / POI point clouds + a small PROTAC cluster (all float32)."""
    rng = np.random.default_rng(seed)
    n_e3, n_poi, n_lig = 80, 80, 20

    def anisotropic(n, eigvals, center, orient_seed):
        R = Rotation.random(random_state=orient_seed).as_matrix()
        cov = R @ np.diag(eigvals) @ R.T
        return rng.multivariate_normal(center, cov, size=n).astype(np.float32)

    e3 = anisotropic(n_e3, [12.0, 5.0, 2.0], np.zeros(3), 100)
    poi0 = anisotropic(n_poi, [14.0, 6.0, 1.5], np.array([22.0, 0.0, 0.0]), 200)
    lig = rng.normal(loc=np.array([11.0, 0.0, 0.0]), scale=2.0, size=(n_lig, 3)).astype(np.float32)
    poi_center = poi0.mean(axis=0).astype(np.float32)
    return e3, poi0, lig, poi_center


def build_nodes(e3, poi, lig):
    node_coords = np.concatenate([e3, poi, lig], axis=0)
    node_types = np.concatenate([
        np.zeros(len(e3), dtype=np.int64),
        np.ones(len(poi), dtype=np.int64),
        2 * np.ones(len(lig), dtype=np.int64),
    ])
    return node_coords, node_types


def encode_parts(encoder, node_coords, node_types):
    """Replicate SE3GeometryEncoder.forward, returning mean/max/full embeddings."""
    coords = torch.from_numpy(node_coords)
    types = torch.from_numpy(node_types)
    e3_mask = types == 0
    e3_center = coords[e3_mask].mean(dim=0)          # [3]
    rel = coords - e3_center                          # [N,3]
    onehot = torch.zeros(len(types), 3, dtype=torch.float32)
    onehot[torch.arange(len(types)), types] = 1.0
    feat = torch.cat([rel, onehot], dim=-1)           # [N,6]
    with torch.no_grad():
        h = encoder.node_net(feat)                    # [N,64]
    mean = h.mean(dim=0)                              # [64]
    mx = h.max(dim=0).values                          # [64]
    full = encoder.combine(torch.cat([mean, mx])).detach().numpy()  # [64] == h_geo
    return mean.numpy(), mx.numpy(), full


def _triu(M):
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    return M[iu]


def _dist(H):
    return np.linalg.norm(H[:, None, :] - H[None, :, :], axis=-1)


def _angles(rots):
    n = len(rots)
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a[i, j] = a[j, i] = (rots[i].inv() * rots[j]).magnitude()
    return a


def _report(name, dH, dGold, target):
    """Correlate pairwise h_geo distances with pairwise target distances."""
    pe, sp = pearsonr(dH, target), spearmanr(dH, target)
    pe_g, sp_g = pearsonr(dGold, target), spearmanr(dGold, target)
    print(f"  {name:<18} pearson={pe[0]:+.4f} spearman={sp[0]:+.4f}   "
          f"[gold descriptor: pearson={pe_g[0]:+.4f}]")
    return pe[0]


def _linear_r2(H, Y):
    """Ridge-free least-squares linear readout R^2 (per dim mean)."""
    M = H.shape[0]
    A = np.concatenate([H, np.ones((M, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    Yhat = A @ coef
    ss_res = ((Y - Yhat) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return r2


def rotation_probe(encoder, e3, poi0, lig, poi_center, M=300, seed=0):
    rots = Rotation.random(M, random_state=seed)
    centered = poi0 - poi_center
    D = encoder.geo_dim
    Hf = np.zeros((M, D)); Hmean = np.zeros((M, D)); Hmax = np.zeros((M, D))
    Gold = np.zeros((M, 6))       # POI covariance upper-triangle
    Rotv = np.zeros((M, 3))       # so(3) rotvec target
    for k, Rk in enumerate(rots):
        poi_k = (poi_center + Rk.apply(centered)).astype(np.float32)
        nc, nt = build_nodes(e3, poi_k, lig)
        mean, mx, full = encode_parts(encoder, nc, nt)
        Hf[k], Hmean[k], Hmax[k] = full, mean, mx
        C = np.cov(poi_k - poi_k.mean(axis=0), rowvar=False)
        Gold[k] = C[np.triu_indices(3)]
        Rotv[k] = Rk.as_rotvec()
    ang = _angles(rots)
    tgt = _triu(ang)
    print("ROTATION probe (POI rotated about its own centroid):")
    print(f"  h_geo full : mean std={Hf.std(0).mean():.5f}  mean |h|={np.linalg.norm(Hf,axis=1).mean():.3f}")
    r_full = _report("h_geo (mean+max)", _triu(_dist(Hf)), _triu(_dist(Gold)), tgt)
    r_mean = _report("mean-pool only", _triu(_dist(Hmean)), _triu(_dist(Gold)), tgt)
    r_max = _report("max-pool only", _triu(_dist(Hmax)), _triu(_dist(Gold)), tgt)
    r2_full = _linear_r2(Hf, Rotv)
    r2_mean = _linear_r2(Hmean, Rotv)
    r2_max = _linear_r2(Hmax, Rotv)
    print(f"  linear readout h_geo->rotvec  R^2 (x,y,z,mean): "
          f"{r2_full.round(4)} mean={r2_full.mean():+.4f}")
    print(f"  linear readout mean-pool->rotvec R^2 mean={r2_mean.mean():+.4f}   "
          f"max-pool->rotvec R^2 mean={r2_max.mean():+.4f}")
    return r_full, r2_full.mean()


def translation_probe(encoder, e3, poi0, lig, poi_center, M=300, seed=1):
    rng = np.random.default_rng(seed)
    # random directions, log-uniform magnitudes in [0.3, 5] A
    dirs = rng.normal(size=(M, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mags = 10 ** rng.uniform(np.log10(0.3), np.log10(5.0), size=M)
    tvec = (dirs * mags[:, None]).astype(np.float32)
    D = encoder.geo_dim
    Hf = np.zeros((M, D))
    Gold = np.zeros((M, 3))       # gold = translation vector itself
    for k in range(M):
        poi_k = (poi0 + tvec[k]).astype(np.float32)
        nc, nt = build_nodes(e3, poi_k, lig)
        _, _, full = encode_parts(encoder, nc, nt)
        Hf[k] = full
        Gold[k] = tvec[k]
    tgt = _triu(_dist(tvec))
    print("TRANSLATION probe (POI shifted, orientation fixed) -- positive control:")
    print(f"  h_geo full : mean std={Hf.std(0).mean():.5f}  mean |h|={np.linalg.norm(Hf,axis=1).mean():.3f}")
    r_full = _report("h_geo (mean+max)", _triu(_dist(Hf)), _triu(_dist(Gold)), tgt)
    r2_full = _linear_r2(Hf, tvec)
    print(f"  linear readout h_geo->translation  R^2 (x,y,z,mean): "
          f"{r2_full.round(4)} mean={r2_full.mean():+.4f}")
    return r_full, r2_full.mean()


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    seeds = [0, 1, 2, 3, 4]
    print("=" * 80)
    print("Cause-2 probe: is SE3GeometryEncoder.h_geo orientation-blind?")
    print("=" * 80)
    rot_corr, rot_r2, tr_corr, tr_r2 = [], [], [], []
    for s in seeds:
        print(f"\n--- encoder seed {s} ---")
        e3, poi0, lig, poi_center = make_clouds(seed=0)
        torch.manual_seed(1000 + s)  # distinct deterministic init per seed
        encoder = SE3GeometryEncoder(geo_dim=64)
        rc, rr2 = rotation_probe(encoder, e3, poi0, lig, poi_center, M=300, seed=s)
        tc, tr2 = translation_probe(encoder, e3, poi0, lig, poi_center, M=300, seed=s)
        rot_corr.append(rc); rot_r2.append(rr2); tr_corr.append(tc); tr_r2.append(tr2)

    print("\n" + "=" * 80)
    print("SUMMARY over 5 encoder seeds")
    print(f"  rotation  h_geo distance-vs-angle pearson : {np.mean(rot_corr):+.4f} +- {np.std(rot_corr):.4f}")
    print(f"  rotation  linear readout h_geo->rotvec R^2: {np.mean(rot_r2):+.4f} +- {np.std(rot_r2):.4f}")
    print(f"  translation h_geo distance-vs-shift pearson: {np.mean(tr_corr):+.4f} +- {np.std(tr_corr):.4f}")
    print(f"  translation linear readout h_geo->t R^2   : {np.mean(tr_r2):+.4f} +- {np.std(tr_r2):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
