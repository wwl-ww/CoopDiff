#!/usr/bin/env python
"""
Controlled A/B test: run BOTH the official predict_cpu.py logic AND our
audit_run_predictions.py logic on the same case, with the SAME random seeds,
to determine if the DockQ gap is from code differences or something else.

Each seed uses an explicitly seeded numpy RNG to ensure reproducibility.
"""
import os, sys, time, tempfile
from copy import deepcopy
import numpy as np
import torch

_PROJECT_ROOT = '/data/run01/scwb792/mwj/CoopDiff'
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'DeepTernary'))

import dgl
from mmengine.config import Config
from mmpretrain.apis.model import get_model

from deepternary.models.geometry_utils import random_rotation_translation
from deepternary.models.process_mols import (
    get_geometry_graph_ring, get_lig_graph_revised, get_rec_graph,
    get_receptor_inference, read_molecule,
)
from deepternary.models.pdb_utils import merge_pdbs, set_new_chain, set_new_coords, write_pdb
from deepternary.models.correct import correct_ligand
from deepternary.models.ternary_pdb import get_pocket_and_mask
from deepternary.models.rotate_utils import kabsch, rotate_and_translate
from deepternary.models.path import IDEAL_PATH
from biopandas.pdb import PandasPdb

# ── Config ────────────────────────────────────────────────────────────────
CONFIG = os.path.join(_PROJECT_ROOT, 'DeepTernary/output/checkpoints/PROTAC/protac_test_config.py')
CHECKPOINT = os.path.join(_PROJECT_ROOT, 'DeepTernary/output/checkpoints/PROTAC/epoch_1000.pth')
UNBOUND_FOLDER = os.path.join(_PROJECT_ROOT, 'DeepTernary/output/protac22')
CORRECT_LIGAND = True
SEEDS = 40
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_CASE = '6BOY_B_C_RN6'

# ── Helpers ───────────────────────────────────────────────────────────────
def RMSD(pred, gt):
    diff = pred - gt
    if isinstance(pred, torch.Tensor):
        return torch.sqrt(torch.sum(diff ** 2, dim=1).mean()).item()
    elif isinstance(pred, np.ndarray):
        return np.sqrt(np.sum(diff ** 2, axis=1).mean())
    raise NotImplementedError

def auto_download_ideal_sdf(lig_name):
    ideal_sdf_path = f'{IDEAL_PATH}/{lig_name}_ideal.sdf'
    if os.path.exists(ideal_sdf_path):
        return ideal_sdf_path
    return None

def get_lig_coords(lig_path):
    lig = read_molecule(lig_path, sanitize=True, remove_hs=True)
    conf = lig.GetConformer()
    true_lig_coords = conf.GetPositions()
    return torch.from_numpy(true_lig_coords).float()

def replace_to_unbound_coords(whole_coords, part_coords):
    cdist = torch.cdist(whole_coords, part_coords)
    update_mask = cdist.min(dim=1)[0] < 1
    whole_coords[update_mask] = part_coords[cdist.argmin(dim=1)[update_mask]]
    return whole_coords, update_mask


# ── Official predict_cpu.py logic (EXACT replica of predict_one_unbound) ───
@torch.no_grad()
def predict_official_logic(name, cfg, rng_state_per_seed=None):
    """
    Replicates predict_cpu.py predict_one_unbound() EXACTLY.
    rng_state_per_seed: list of (rng_state, seed) tuples for reproducibility
    """
    print(f'\n=== OFFICIAL predict_cpu.py logic: {name} ===')
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
    for si, seed in enumerate(range(SEEDS)):
        # Restore numpy RNG state for reproducibility
        if rng_state_per_seed is not None:
            np.random.set_state(rng_state_per_seed[si])

        lig = deepcopy(lig_origin)
        p1_graph = deepcopy(p1_graph_origin)
        p2_graph = deepcopy(p2_graph_origin)

        # EXACT predict_cpu.py lines 354-362 (FIX_TWO_ENDS=False path)
        lig, lig_graph = get_lig_graph_revised(
            lig, name=name, max_neighbors=ds_cfg.lig_max_neighbors,
            ideal_path=auto_download_ideal_sdf(name.split('_')[-1]),
            use_random_coords=False,
            seed=seed,
            use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius)
        geometry_graph = get_geometry_graph_ring(lig)

        lig_coords_gt = deepcopy(lig_graph.ndata['x'])

        # Unbound pocket
        lig1_coords = get_lig_coords(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig1.pdb'))
        lig2_coords = get_lig_coords(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig2.pdb'))
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

        # Random move lig (uses numpy global RNG!)
        rot_T, rot_b = random_rotation_translation(translation_distance=5)
        lig_coords_to_move = data['lig_graph'].ndata['new_x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        data['lig_graph'].ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        data['p1lig_lig_pocket_coords'][0] = (rot_T @ (data['p1lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b
        data['p2lig_lig_pocket_coords'][0] = (rot_T @ (data['p2lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b

        # Random move p2
        protein2_rot_T, protein2_rot_b = random_rotation_translation(translation_distance=5)
        protein2_coord_to_move = data['rec2_graph'].ndata['x']
        protein2_mean_to_remove = protein2_coord_to_move.mean(dim=0, keepdims=True)
        data['rec2_graph'].ndata['x'] = (protein2_rot_T @ (protein2_coord_to_move - protein2_mean_to_remove).T).T + protein2_rot_b
        data['p2lig_p2_pocket_coords'][0] = (protein2_rot_T @ (data['p2lig_p2_pocket_coords'][0] - protein2_mean_to_remove).T).T + protein2_rot_b
        data['rec2_coords_input'] = [data['rec2_graph'].ndata['x']]

        with torch.no_grad():
            model_outputs = model(**data, mode='predict')[0]

        prediction = model_outputs['ligs_coords_pred']
        p2_to_p1_rotation = model_outputs['rotation_2']
        p2_to_p1_translation = model_outputs['translation_2']
        p2_rmsd_pred = model_outputs['p2_rmsd_pred'].item()
        lig_coords_gt_t = data['lig_graph'].ndata['x'].to(prediction.device)

        if CORRECT_LIGAND:
            prediction = correct_ligand(prediction, lig_graph.ndata['new_x'], lig)
            prediction = torch.from_numpy(prediction).float()

        # Save complex for DockQ
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
            seed=seed, fnat=fnat, irms=irms, Lrms=Lrms, dockq=DockQ,
            pred_p2_rmsd=p2_rmsd_pred, lig_rmsd=lig_rmsd, sm_rmsd=sm_rmsd,
        ))
        print(f'  [OFFICIAL] Seed {seed}: DockQ={DockQ:.4f}  p2_rmsd={p2_rmsd_pred:.2f}  lig_rmsd={lig_rmsd:.2f}')

    results = sorted(results, key=lambda x: x['pred_p2_rmsd'])
    dockqs = [r['dockq'] for r in results]
    print(f'\n  OFFICIAL logic — Best DockQ: {max(dockqs):.4f}  Mean DockQ: {np.mean(dockqs):.4f}')
    return results


# ── Our audit logic (adapted from predict_one_case with FIX_TWO_ENDS=False) ─
@torch.no_grad()
def predict_audit_logic(name, cfg, rng_state_per_seed=None):
    """
    Replicates our audit_run_predictions.py predict_one_case() with FIX_TWO_ENDS=False.
    """
    print(f'\n=== AUDIT logic (ours): {name} ===')
    model = cfg.nn_model
    ds_cfg = cfg.test_dataloader.dataset

    lig_path = os.path.join(UNBOUND_FOLDER, name, 'ligand.pdb')
    p1_path = os.path.join(UNBOUND_FOLDER, name, 'unbound_protein1.pdb')
    p2_path = os.path.join(UNBOUND_FOLDER, name, 'unbound_protein2.pdb')
    merge_p1_path = os.path.join(UNBOUND_FOLDER, name, 'protein1.pdb')
    merge_p2_path = os.path.join(UNBOUND_FOLDER, name, 'protein2.pdb')

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
    for si, seed in enumerate(range(SEEDS)):
        if rng_state_per_seed is not None:
            np.random.set_state(rng_state_per_seed[si])

        lig = deepcopy(lig_origin)
        p1_graph = deepcopy(p1_graph_origin)
        p2_graph = deepcopy(p2_graph_origin)

        # Our audit code (FIX_TWO_ENDS=False path, with ideal_path passed)
        lig_name = name.split('_')[-1]
        ideal_path = auto_download_ideal_sdf(lig_name)
        lig, lig_graph = get_lig_graph_revised(
            lig, name=name, max_neighbors=ds_cfg.lig_max_neighbors,
            use_random_coords=False, seed=seed,
            use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius,
            ideal_path=ideal_path)
        geometry_graph = get_geometry_graph_ring(lig)

        lig_coords_gt = deepcopy(lig_graph.ndata['x'])

        lig1_coords = get_lig_coords(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig1.pdb'))
        lig2_coords = get_lig_coords(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig2.pdb'))
        lig_coords_gt, update_mask1 = replace_to_unbound_coords(lig_coords_gt, lig1_coords)
        lig_coords_gt, update_mask2 = replace_to_unbound_coords(lig_coords_gt, lig2_coords)
        pocket_mask = update_mask1 | update_mask2

        p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = \
            get_pocket_and_mask(lig_coords_gt, p1_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
        p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = \
            get_pocket_and_mask(lig_coords_gt, p2_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)

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
        protein2_rot_T, protein2_rot_b = random_rotation_translation(translation_distance=5)
        protein2_coord_to_move = data['rec2_graph'].ndata['x']
        protein2_mean_to_remove = protein2_coord_to_move.mean(dim=0, keepdims=True)
        data['rec2_graph'].ndata['x'] = (protein2_rot_T @ (protein2_coord_to_move - protein2_mean_to_remove).T).T + protein2_rot_b
        data['p2lig_p2_pocket_coords'][0] = (protein2_rot_T @ (data['p2lig_p2_pocket_coords'][0] - protein2_mean_to_remove).T).T + protein2_rot_b
        data['rec2_coords_input'] = [data['rec2_graph'].ndata['x']]

        with torch.no_grad():
            model_outputs = model(**data, mode='predict')[0]

        prediction = model_outputs['ligs_coords_pred']
        p2_to_p1_rotation = model_outputs['rotation_2']
        p2_to_p1_translation = model_outputs['translation_2']
        p2_rmsd_pred = model_outputs['p2_rmsd_pred'].item()
        lig_coords_gt_dev = data['lig_graph'].ndata['x'].to(prediction.device)

        if CORRECT_LIGAND:
            prediction = correct_ligand(prediction, lig_graph.ndata['new_x'], lig)
            prediction = torch.from_numpy(prediction).float()

        # Save complex for DockQ
        tmp_dir = tempfile.mkdtemp()
        lig_pred_path = os.path.join(tmp_dir, f'lig_pred_{seed}.pdb')
        pred_lig_pdb = set_new_coords(lig_path, prediction.cpu().numpy(), is_ligand=True)
        pred_lig_pdb = set_new_chain(pred_lig_pdb, 'A', is_ligand=True)
        write_pdb(pred_lig_pdb, lig_pred_path)

        p2_gt = data['rec2_coords'][0].to(prediction.device)
        p2_pred = rotate_and_translate(data['rec2_coords_input'][0].to(prediction.device),
                                        p2_to_p1_rotation, p2_to_p1_translation)
        p2_rot, p2_trans, _ = kabsch(p2_gt, p2_pred)

        p2_pred_file = os.path.join(tmp_dir, f'p2_pred_{seed}.pdb')
        p2_pdb = PandasPdb().read_pdb(merge_p2_path)
        p2_gt_coords = p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
        p2_pred_coords = rotate_and_translate(p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy())
        p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_pred_coords
        p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)
        complex_save_path = os.path.join(tmp_dir, f'complex_pred_{seed}.pdb')
        merge_pdbs(pdb_files=[merge_p1_path, p2_pred_file, lig_pred_path], path=complex_save_path)

        # DockQ
        from DockQ.dockq_util import cal_dockq
        try:
            fnat, irms, Lrms, DockQ = cal_dockq(
                complex_save_path,
                os.path.join(UNBOUND_FOLDER, name, 'gt_complex.pdb'))
        except Exception as e:
            print(f'  [AUDIT] DockQ failed for {name} seed {seed}: {e}')
            fnat, irms, Lrms, DockQ = 0.0, 0.0, 0.0, 0.0

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        lig_rmsd = RMSD(prediction, lig_coords_gt_dev)
        pocket_mask_combined = p1lig_lig_pocket_mask | p2lig_lig_pocket_mask
        sm_rmsd = RMSD(prediction[pocket_mask_combined], lig_coords_gt_dev[pocket_mask_combined])

        results.append(dict(
            seed=seed, fnat=fnat, irms=irms, Lrms=Lrms, dockq=DockQ,
            pred_p2_rmsd=p2_rmsd_pred, lig_rmsd=lig_rmsd, sm_rmsd=sm_rmsd,
        ))
        print(f'  [AUDIT]    Seed {seed}: DockQ={DockQ:.4f}  p2_rmsd={p2_rmsd_pred:.2f}  lig_rmsd={lig_rmsd:.2f}')

    results = sorted(results, key=lambda x: x['pred_p2_rmsd'])
    dockqs = [r['dockq'] for r in results]
    print(f'\n  AUDIT logic — Best DockQ: {max(dockqs):.4f}  Mean DockQ: {np.mean(dockqs):.4f}')
    return results


if __name__ == '__main__':
    print(f"Config: {CONFIG}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")
    print(f"Test case: {TEST_CASE}")
    print(f"Seeds: {SEEDS}")

    cfg = Config.fromfile(CONFIG)
    cfg.model.noise_initial = 0
    cfg.tmp_dir = tempfile.TemporaryDirectory()
    cfg.device = DEVICE
    cfg.model.device = DEVICE

    print("\nLoading model...")
    model = get_model(cfg, pretrained=CHECKPOINT, device=DEVICE)
    model.to(DEVICE)
    model.device = DEVICE
    model.eval()
    cfg.nn_model = model
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}\n")

    # Pre-save numpy RNG states so BOTH runs use the SAME random perturbations
    master_seed = 42
    np.random.seed(master_seed)
    rng_states = [np.random.get_state() for _ in range(SEEDS)]

    # Run OFFICIAL logic first
    t0 = time.time()
    official_results = predict_official_logic(TEST_CASE, cfg, rng_state_per_seed=rng_states)
    official_time = time.time() - t0

    # Run AUDIT logic second (same RNG states)
    t0 = time.time()
    audit_results = predict_audit_logic(TEST_CASE, cfg, rng_state_per_seed=rng_states)
    audit_time = time.time() - t0

    # Comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON for {TEST_CASE}:")
    print(f"  Official SOTA (paper):     best DockQ = 0.7108")
    print(f"  Our SOTA audit (prior):    best DockQ = 0.3211")
    official_dockqs = [r['dockq'] for r in official_results]
    audit_dockqs = [r['dockq'] for r in audit_results]
    print(f"  THIS RUN official logic:   best={max(official_dockqs):.4f} mean={np.mean(official_dockqs):.4f}")
    print(f"  THIS RUN audit logic:      best={max(audit_dockqs):.4f} mean={np.mean(audit_dockqs):.4f}")
    print(f"  Timing: official={official_time:.1f}s  audit={audit_time:.1f}s")

    # Per-seed comparison
    print(f"\n  Per-seed DockQ comparison:")
    all_match = True
    for i in range(SEEDS):
        o = official_results[i]
        a = audit_results[i]
        dockq_match = abs(o['dockq'] - a['dockq']) < 0.001
        if not dockq_match:
            all_match = False
        status = "✓" if dockq_match else "✗ DIFF"
        print(f"    Seed {o['seed']:02d}: official={o['dockq']:.4f}  audit={a['dockq']:.4f}  {status}")

    if all_match:
        print(f"\n  *** ALL SEEDS MATCH — code is functionally identical ***")
    else:
        print(f"\n  *** SEEDS DIFFER — there IS a code/logic difference ***")
    print(f"{'='*70}")

    cfg.tmp_dir.cleanup()
