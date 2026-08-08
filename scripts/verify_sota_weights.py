#!/usr/bin/env python
"""
Quick verification: load SOTA epoch_1000.pth and run 1-2 cases with 10 seeds.
Expected: DockQ should match official results (~0.706 avg).

Usage:
    python scripts/verify_sota_weights.py
"""

import os, sys, time, tempfile
from copy import deepcopy
import numpy as np
import scipy.spatial as spa
import torch
import dgl
from scipy.special import softmax
from mmengine.config import Config

# ── Setup paths ──────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'DeepTernary'))

from deepternary.models.geometry_utils import random_rotation_translation
from deepternary.models.process_mols import (
    get_geometry_graph_ring, get_lig_graph_revised, get_rec_graph,
    get_receptor_inference, read_molecule, rigid_transform_Kabsch_3D,
)
from deepternary.models.pdb_utils import merge_pdbs, set_new_chain, set_new_coords, write_pdb
from deepternary.models.correct import correct_ligand
from deepternary.models.ternary_pdb import get_pocket_and_mask
from deepternary.models.rotate_utils import kabsch, rotate_and_translate
from deepternary.models.logger import log as dt_log
from deepternary.models.path import IDEAL_PATH
from biopandas.pdb import PandasPdb
from mmpretrain.apis.model import get_model

# ── Config ────────────────────────────────────────────────────────────────
# Use the SOTA config from public/ + the SOTA checkpoint
CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'DeepTernary/output/checkpoints/PROTAC/protac_test_config.py')
CHECKPOINT_PATH = os.path.join(_PROJECT_ROOT, 'DeepTernary/output/checkpoints/PROTAC/epoch_1000.pth')
PDB_PATH = os.path.join(_PROJECT_ROOT, 'DeepTernary/data/PDB_sdf')
UNBOUND_FOLDER = os.path.join(_PROJECT_ROOT, 'DeepTernary/output/protac22')

FIX_TWO_ENDS = False  # Match official config
CORRECT_LIGAND = True
SEEDS = 10  # Quick test with 10 seeds
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cases to test: one easy (high DockQ), one hard (low DockQ)
TEST_CASES = [
    '5T35_D_A_759',  # Official best DockQ: 0.9516, audit: 0.7035
    '6BN7_B_C_RN3',  # Official best DockQ: 0.7018, audit: 0.4449
]

# ── Helpers ───────────────────────────────────────────────────────────────
def RMSD(pred, gt):
    diff = pred - gt
    if isinstance(pred, torch.Tensor):
        return torch.sqrt(torch.sum(diff ** 2, dim=1).mean()).item()
    elif isinstance(pred, np.ndarray):
        return np.sqrt(np.sum(diff ** 2, axis=1).mean())
    raise NotImplementedError

def auto_download_ideal_sdf(lig_name, case_name=None):
    # First check per-case cache in protac22 (already exists for all 22 cases)
    if case_name:
        case_cache = os.path.join(UNBOUND_FOLDER, case_name, 'ligand_ideal.sdf')
        if os.path.exists(case_cache):
            return case_cache
    # Then check shared cache
    ideal_sdf_path = os.path.join(IDEAL_PATH, f'{lig_name}_ideal.sdf')
    if os.path.exists(ideal_sdf_path):
        return ideal_sdf_path
    # Try download (may fail behind firewall)
    try:
        import requests
        os.makedirs(IDEAL_PATH, exist_ok=True)
        url = f'https://files.rcsb.org/ligands/download/{lig_name}_ideal.sdf'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(ideal_sdf_path, 'wb') as f:
            f.write(response.content)
        return ideal_sdf_path
    except Exception as e:
        print(f'  [WARN] Cannot get ideal SDF for {lig_name}: {e}')
        return None

# ── Main prediction loop (adapted from predict_cpu.py) ─────────────────────
@torch.no_grad()
def predict_one(name, cfg, seed_num=40):
    print(f'\n=== {name} ===')
    model = cfg.nn_model
    ds_cfg = cfg.test_dataloader.dataset

    lig_path = os.path.join(UNBOUND_FOLDER, name, 'ligand.pdb')
    p1_path = os.path.join(UNBOUND_FOLDER, name, 'unbound_protein1.pdb')
    p2_path = os.path.join(UNBOUND_FOLDER, name, 'unbound_protein2.pdb')

    lig_origin = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p1_path)
    p1_graph_origin = get_rec_graph(recs, recs_coords, c_alpha_coords, n_coords, c_coords,
                                    use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
                                    surface_max_neighbors=ds_cfg.surface_max_neighbors,
                                    surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
                                    surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
                                    c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p2_path)
    p2_graph_origin = get_rec_graph(recs, recs_coords, c_alpha_coords, n_coords, c_coords,
                                    use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
                                    surface_max_neighbors=ds_cfg.surface_max_neighbors,
                                    surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
                                    surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
                                    c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors)
    protein2_coords_gt = deepcopy(p2_graph_origin.ndata['x'])

    results = []
    for seed in range(seed_num):
        lig = deepcopy(lig_origin)
        p1_graph = deepcopy(p1_graph_origin)
        p2_graph = deepcopy(p2_graph_origin)

        lig, lig_graph = get_lig_graph_revised(
            lig, name=name, max_neighbors=ds_cfg.lig_max_neighbors,
            ideal_path=auto_download_ideal_sdf(name.split('_')[-1], case_name=name),
            use_random_coords=False, seed=seed,
            use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius)
        geometry_graph = get_geometry_graph_ring(lig)

        lig_coords_gt = deepcopy(lig_graph.ndata['x'])

        # Unbound pocket
        from deepternary.models.process_mols import get_lig_coords as _get_lig_coords
        lig1_coords = _get_lig_coords(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig1.pdb'))
        lig2_coords = _get_lig_coords(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig2.pdb'))

        def replace_to_unbound_coords(whole_coords, part_coords):
            cdist = torch.cdist(whole_coords, part_coords)
            update_mask = cdist.min(dim=1)[0] < 1
            whole_coords[update_mask] = part_coords[cdist.argmin(dim=1)[update_mask]]
            return whole_coords, update_mask

        lig_coords_gt, update_mask1 = replace_to_unbound_coords(lig_coords_gt, lig1_coords)
        lig_coords_gt, update_mask2 = replace_to_unbound_coords(lig_coords_gt, lig2_coords)
        pocket_mask = update_mask1 | update_mask2

        p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
            lig_coords_gt, p1_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
        p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
            lig_coords_gt, p2_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)

        p1lig_lig_pocket_mask = p1lig_lig_pocket_mask & pocket_mask
        p1lig_lig_pocket_coords = lig_coords_gt[p1lig_lig_pocket_mask]
        p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)

        p2lig_lig_pocket_mask = p2lig_lig_pocket_mask & pocket_mask
        p2lig_lig_pocket_coords_origin = lig_coords_gt[p2lig_lig_pocket_mask]

        data = dict(
            lig_graph=lig_graph, rec_graph=p1_graph, rec2_graph=p2_graph,
            geometry_graph=geometry_graph,
            complex_name=[name], rec2_coords=[protein2_coords_gt],
            p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
            p1lig_p1_pocket_coords=[p1lig_p1_pocket_coords],
            p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
            p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
            p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
            p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords_origin],
            p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
            p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords_origin],
        )

        # Random move lig
        rot_T, rot_b = random_rotation_translation(translation_distance=5)
        lig_coords_to_move = data['lig_graph'].ndata['new_x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        data['lig_graph'].ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        data['p1lig_lig_pocket_coords'][0] = (rot_T @ (data['p1lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b
        data['p2lig_lig_pocket_coords'][0] = (rot_T @ (data['p2lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b

        # Random move p2
        p2_rot_T, p2_rot_b = random_rotation_translation(translation_distance=5)
        p2_coord_to_move = data['rec2_graph'].ndata['x']
        p2_mean = p2_coord_to_move.mean(dim=0, keepdims=True)
        data['rec2_graph'].ndata['x'] = (p2_rot_T @ (p2_coord_to_move - p2_mean).T).T + p2_rot_b
        data['p2lig_p2_pocket_coords'][0] = (p2_rot_T @ (data['p2lig_p2_pocket_coords'][0] - p2_mean).T).T + p2_rot_b
        data['rec2_coords_input'] = [data['rec2_graph'].ndata['x']]

        with torch.no_grad():
            model_outputs = model(**data, mode='predict')[0]

        prediction = model_outputs['ligs_coords_pred']
        p2_to_p1_rotation = model_outputs['rotation_2']
        p2_to_p1_translation = model_outputs['translation_2']
        p2_rmsd_pred = model_outputs['p2_rmsd_pred'].item()
        lig_coords_gt_t = data['lig_graph'].ndata['x'].to(prediction.device)

        # Ligand correction
        if CORRECT_LIGAND:
            prediction = correct_ligand(prediction, lig_graph.ndata['new_x'], lig)
            prediction = torch.from_numpy(prediction).float()

        # Save predicted complex for DockQ
        tmp_dir = tempfile.mkdtemp()
        lig_pred_path = os.path.join(tmp_dir, f'lig_pred_{seed}.pdb')
        pred_lig = set_new_coords(lig_path, prediction.cpu().numpy(), is_ligand=True)
        pred_lig = set_new_chain(pred_lig, 'A', is_ligand=True)
        write_pdb(pred_lig, lig_pred_path)

        p2_gt = data['rec2_coords'][0].to(prediction.device)
        p2_pred = rotate_and_translate(data['rec2_coords_input'][0].to(prediction.device),
                                        p2_to_p1_rotation, p2_to_p1_translation)
        p2_rot, p2_trans, _ = kabsch(p2_gt, p2_pred)

        p1_path = os.path.join(UNBOUND_FOLDER, name, 'protein1.pdb')
        p2_path = os.path.join(UNBOUND_FOLDER, name, 'protein2.pdb')
        p2_pred_file = os.path.join(tmp_dir, f'p2_pred_{seed}.pdb')
        p2_pdb = PandasPdb().read_pdb(p2_path)
        p2_gt_coords = p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
        p2_pred_coords = rotate_and_translate(p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy())
        p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_pred_coords
        p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)
        complex_save_path = os.path.join(tmp_dir, f'complex_pred_{seed}.pdb')
        merge_pdbs(pdb_files=[p1_path, p2_pred_file, lig_pred_path], path=complex_save_path)

        # DockQ
        from DockQ.dockq_util import cal_dockq
        try:
            fnat, irms, Lrms, DockQ = cal_dockq(
                complex_save_path,
                os.path.join(UNBOUND_FOLDER, name, 'gt_complex.pdb'))
        except Exception as e:
            print(f'  Seed {seed}: DockQ error: {e}')
            fnat, irms, Lrms, DockQ = 0., 0., 0., 0.

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        lig_rmsd = RMSD(prediction, lig_coords_gt_t)
        sm_rmsd = RMSD(prediction[p1lig_lig_pocket_mask | p2lig_lig_pocket_mask],
                       lig_coords_gt_t[p1lig_lig_pocket_mask | p2lig_lig_pocket_mask])

        results.append(dict(
            fnat=fnat, irms=irms, Lrms=Lrms, dockq=DockQ,
            pred_p2_rmsd=p2_rmsd_pred, lig_rmsd=lig_rmsd, sm_rmsd=sm_rmsd,
        ))
        print(f'  Seed {seed}: DockQ={DockQ:.4f}  p2_rmsd={p2_rmsd_pred:.2f}  lig_rmsd={lig_rmsd:.2f}')

    results = sorted(results, key=lambda x: x['pred_p2_rmsd'])
    dockqs = [r['dockq'] for r in results]
    print(f'\n  Best DockQ: {max(dockqs):.4f}  Avg DockQ: {np.mean(dockqs):.4f}  '
          f'Top-1 DockQ: {dockqs[0]:.4f}')
    return results


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Config:     {CONFIG_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device:     {DEVICE}")
    print(f"Seeds:      {SEEDS}")
    print(f"Cases:      {TEST_CASES}")

    if not os.path.exists(CONFIG_PATH):
        sys.exit(f"Config not found: {CONFIG_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        sys.exit(f"Checkpoint not found: {CHECKPOINT_PATH}")

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.noise_initial = 0
    cfg.device = DEVICE
    cfg.model.device = DEVICE
    cfg.tmp_dir = tempfile.TemporaryDirectory()

    print("\nLoading model...")
    model = get_model(cfg, pretrained=CHECKPOINT_PATH, device=DEVICE)
    model.to(DEVICE)
    model.device = DEVICE
    model.eval()
    cfg.nn_model = model
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}\n")

    total_start = time.time()
    for name in TEST_CASES:
        t0 = time.time()
        results = predict_one(name, cfg, seed_num=SEEDS)
        dockqs = [r['dockq'] for r in results]
        print(f"\n  → {name}: Best DockQ={max(dockqs):.4f}  Avg DockQ={np.mean(dockqs):.4f}  Time={time.time()-t0:.1f}s")

    print(f"\nTotal time: {time.time()-total_start:.1f}s")

    # Compare with official
    print("\n" + "="*70)
    print("COMPARISON (best DockQ)")
    print("="*70)
    official = {
        '5T35_D_A_759': 0.9516,
        '6BN7_B_C_RN3': 0.7018,
    }
    audit_old = {
        '5T35_D_A_759': 0.7035,
        '6BN7_B_C_RN3': 0.4449,
    }
    for name in TEST_CASES:
        print(f"  {name}:")
        print(f"    Official SOTA:  {official[name]:.4f}")
        print(f"    Old audit:      {audit_old[name]:.4f}")
        print(f"    → Should match official if weights correct")

    cfg.tmp_dir.cleanup()
