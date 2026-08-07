"""
TriCoopDiff inference script.

Generates PROTAC ternary complex structures via multi-walker reverse diffusion.

Usage:
    python scripts/inference.py \
        --e3_pdb data/protac_cases/5t35/e3.pdb \
        --poi_pdb data/protac_cases/5t35/poi.pdb \
        --lig_sdf data/protac_cases/5t35/protac.sdf \
        --checkpoint checkpoints/tricoopdiff_ft.pt \
        --outdir results/5t35 \
        --n_walkers 32 \
        --steps 20

Architecture:
    1. Build two disconnected subgraphs (E3-L, POI-L) + full ternary graph
    2. TriBrain processes ternary graph -> FiLM params + s_SE3
    3. Frozen DiffDock experts run with per-layer FiLM modulation
    4. Scores merged by direct addition -> reverse SDE step
    5. Repeat for `steps` steps, for `n_walkers` independent trajectories
    6. TriReflect post-processing -> final ensemble
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="TriCoopDiff ternary complex inference")

    # Required inputs
    p.add_argument("--e3_pdb", required=True, help="E3 ligase PDB file")
    p.add_argument("--poi_pdb", required=True, help="POI target PDB file")
    p.add_argument("--lig_sdf", required=True, help="PROTAC ligand SDF file")

    # Model
    p.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    p.add_argument("--diffdock_weights", default=None,
                   help="Path to DiffDock pretrained weights (default: auto-detect)")

    # Inference
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--n_walkers", type=int, default=32, help="Number of independent walkers")
    p.add_argument("--steps", type=int, default=20, help="Reverse diffusion steps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Optional: prior knowledge
    p.add_argument("--pocket_guide", action="store_true",
                   help="Enable pocket-guided diffusion (requires warhead info)")
    p.add_argument("--e3_pocket_residues", default=None,
                   help="E3 pocket residue indices (comma-separated)")
    p.add_argument("--poi_pocket_residues", default=None,
                   help="POI pocket residue indices (comma-separated)")

    # Optional: ablation flags
    p.add_argument("--no_film", action="store_true",
                   help="Disable FiLM (L0 ablation)")
    p.add_argument("--no_trireflect", action="store_true",
                   help="Skip TriReflect post-processing")
    p.add_argument("--single_rate", action="store_true",
                   help="Use single-rate diffusion (ablation C)")

    # Output
    p.add_argument("--save_trajectory", action="store_true",
                   help="Save full trajectory (not just final structures)")
    p.add_argument("--name", default=None, help="Run name (default: auto-generated)")

    return p.parse_args()


def main():
    args = parse_args()

    # ----------------------------------------------------------------
    # 1. Setup
    # ----------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    run_name = args.name or Path(args.e3_pdb).stem
    device = torch.device(args.device)
    print(f"[TriCoopDiff] inference | run: {run_name} | device: {device}")
    print(f"  walkers: {args.n_walkers} | steps: {args.steps}")

    # ----------------------------------------------------------------
    # 2. Load model
    # ----------------------------------------------------------------
    print("[1/6] Loading model...")
    model, diffusion = load_model(
        args.checkpoint,
        args.diffdock_weights,
        device,
        no_film=args.no_film,
        single_rate=args.single_rate,
    )
    model.eval()

    # ----------------------------------------------------------------
    # 3. Parse input structures
    # ----------------------------------------------------------------
    print("[2/6] Parsing input structures...")
    e3_data = parse_protein(args.e3_pdb)
    poi_data = parse_protein(args.poi_pdb)
    ligand_data = parse_ligand(args.lig_sdf)

    # Validate
    assert e3_data['coords'].shape[0] > 0, "E3 has no residues"
    assert poi_data['coords'].shape[0] > 0, "POI has no residues"
    assert ligand_data['coords'].shape[0] > 0, "PROTAC has no atoms"
    print(f"  E3: {e3_data['coords'].shape[0]} residues")
    print(f"  POI: {poi_data['coords'].shape[0]} residues")
    print(f"  PROTAC: {ligand_data['coords'].shape[0]} atoms, "
          f"{ligand_data.get('n_rotatable_bonds', '?')} rotatable bonds")

    # ----------------------------------------------------------------
    # 4. Initialize system at t=1
    # ----------------------------------------------------------------
    print("[3/6] Initializing ternary system...")
    from tricoopdiff.diffusion import TernaryInitializer
    initializer = TernaryInitializer()

    # E3 fixed at origin
    e3_center = e3_data['coords'].mean(dim=0, keepdim=True)
    e3_coords = e3_data['coords'] - e3_center

    # Initialize PROTAC near E3 pocket
    pocket_center = e3_coords.mean(dim=0, keepdim=True)  # fallback: E3 center
    if args.e3_pocket_residues:
        indices = [int(x) for x in args.e3_pocket_residues.split(",")]
        pocket_center = e3_coords[indices].mean(dim=0, keepdim=True)

    # Prepare batched data tensors [n_walkers, N, 3]
    init_ligand = initializer.initialize_protac(
        pocket_center.repeat(args.n_walkers, 1),
        ligand_data['coords'],
    )
    init_poi = initializer.initialize_poi(
        torch.zeros(args.n_walkers, 3, device=device),
        poi_data['coords'],
    )

    # ----------------------------------------------------------------
    # 5. Reverse diffusion
    # ----------------------------------------------------------------
    print(f"[4/6] Running reverse diffusion ({args.steps} steps × {args.n_walkers} walkers)...")
    schedule = diffusion.get_inference_schedule(args.steps)
    trajectory = run_reverse_diffusion(
        model, diffusion, schedule,
        e3_data, poi_data, ligand_data,
        init_ligand, init_poi,
        device, args.n_walkers,
        save_trajectory=args.save_trajectory,
        pocket_guide=args.pocket_guide,
        pocket_center=pocket_center if args.pocket_guide else None,
    )

    # ----------------------------------------------------------------
    # 6. TriReflect post-processing
    # ----------------------------------------------------------------
    ligand_ensemble = [t['ligand_coords'] for t in trajectory['final']]
    poi_ensemble = [t['poi_coords'] for t in trajectory['final']]

    if not args.no_trireflect:
        print("[5/6] TriReflect post-processing...")
        from tricoopdiff.reflect import tricoop_reflect
        reflect_results = []
        for i, (lig_coords, poi_coords) in enumerate(zip(ligand_ensemble, poi_ensemble)):
            # Concatenate all atoms for clash detection
            all_coords = np.concatenate([
                e3_coords.cpu().numpy(),
                poi_coords.cpu().numpy(),
                lig_coords.cpu().numpy(),
            ], axis=0)
            atomic_numbers = np.concatenate([
                np.full(e3_coords.shape[0], 6),  # Cα proxy
                np.full(poi_coords.shape[0], 6),
                ligand_data.get('atomic_numbers',
                    np.full(lig_coords.shape[0], 6)),
            ])
            result = tricoop_reflect(all_coords, atomic_numbers, do_minimization=True)
            reflect_results.append(result)

        n_valid = sum(r['is_valid'] for r in reflect_results)
        print(f"  Valid structures: {n_valid}/{len(reflect_results)}")
    else:
        reflect_results = None

    # ----------------------------------------------------------------
    # 7. Save outputs
    # ----------------------------------------------------------------
    print("[6/6] Saving results...")
    save_results(
        args.outdir, run_name,
        e3_coords, ligand_ensemble, poi_ensemble,
        trajectory, reflect_results,
        args.save_trajectory,
    )

    print(f"\nDone. Results saved to: {args.outdir}")
    print(f"  {len(ligand_ensemble)} structures generated")


# ===================================================================
# Model Loading
# ===================================================================

def _make_timestep_emb(embedding_dim=32):
    from utils.diffusion_utils import GaussianFourierProjection
    return GaussianFourierProjection(embedding_size=embedding_dim)


def load_model(checkpoint_path, diffdock_weights, device, no_film=False, single_rate=False):
    """Load TriCoopDiff model and diffusion process."""
    from tricoopdiff import TriCoopDiff
    from tricoopdiff.tribrain import TriBrain
    from tricoopdiff.expert import FrozenDiffDock
    from tricoopdiff.diffusion import MultiRateDiffusion

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Determine config from checkpoint or use defaults
    cfg = ckpt.get('config', {})
    ns = cfg.get('ns', 32)
    nv = cfg.get('nv', 4)
    num_layers = cfg.get('num_transformer_layers', 3)
    num_conv_layers = cfg.get('num_conv_layers', 6)
    node_dim = cfg.get('node_dim', 128)

    # Build TriBrain
    tribrain = TriBrain(
        node_dim=node_dim,
        ns=ns,
        nv=nv,
        num_transformer_layers=num_layers,
        num_conv_layers_per_expert=num_conv_layers,
    )

    # Build frozen expert (requires DiffDock in path)
    diffdock_root = str(Path(__file__).parent.parent / "DiffDock")
    sys.path.insert(0, diffdock_root)

    try:
        from models.cg_model import CGModel
    except ImportError:
        print("ERROR: Cannot import DiffDock. Ensure DiffDock/ is in the project root.")
        print("  git clone https://github.com/gcorso/DiffDock.git")
        sys.exit(1)

    # Build CGModel with default DiffDock-L config
    def _build_cg_model():
        return CGModel(
            t_to_sigma=lambda tr, rot, tor: (
                torch.tensor(0.1) * (30.0 / 0.1) ** tr,
                torch.tensor(0.1) * (1.6 / 0.1) ** rot,
                torch.tensor(0.2) * (2.0 / 0.2) ** tor,
            ),
            device=device,
            timestep_emb_func=_make_timestep_emb(32),
            ns=ns, nv=nv,
            num_conv_layers=num_conv_layers,
            scale_by_sigma=True,
            separate_noise_schedule=False,
        )

    cg_model = _build_cg_model()

    # Load DiffDock pretrained weights if available
    if diffdock_weights and os.path.exists(diffdock_weights):
        cg_model.load_state_dict(torch.load(diffdock_weights, map_location=device), strict=False)

    frozen_expert = FrozenDiffDock(cg_model, num_conv_layers=num_conv_layers)

    # Build full model (with ablation flags)
    model = TriCoopDiff(
        frozen_expert, tribrain, num_conv_layers=num_conv_layers,
        use_film=not no_film,
    )

    # Load TriBrain weights from checkpoint
    if 'tribrain_state_dict' in ckpt:
        model.tribrain.load_state_dict(ckpt['tribrain_state_dict'])
    else:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    model.to(device)

    # Build diffusion
    diffusion = MultiRateDiffusion(
        inference_steps=20,
        **{k: v for k, v in cfg.get('diffusion', {}).items()}
    )

    # Override for single-rate ablation
    if single_rate:
        diffusion.lambda_q_threshold = 0.0
        diffusion.lambda_se3_threshold = 0.0

    return model, diffusion


# ===================================================================
# Input Parsing
# ===================================================================

def parse_protein(pdb_path: str) -> dict:
    """
    Parse protein PDB to residue-level features and coordinates.

    Uses BioPython for robust parsing. Falls back to manual Cα extraction.
    """
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]

        coords = []
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':  # standard residue
                    try:
                        ca = residue['CA']
                        coords.append(ca.get_coord())
                    except KeyError:
                        continue

        coords = np.array(coords)
    except Exception:
        # Fallback: extract Cα atoms manually
        coords = _extract_ca_fallback(pdb_path)

    coords = torch.from_numpy(coords).float()
    n_residues = coords.shape[0]

    # Minimal residue features: one-hot amino acid + placeholder
    # Full feature extraction should use DiffDock's process_mols
    features = torch.zeros(n_residues, 1)  # placeholder

    return {
        'coords': coords,
        'features': features,
        'n_residues': n_residues,
        'pdb_path': pdb_path,
    }


def parse_ligand(sdf_path: str) -> dict:
    """
    Parse ligand SDF to atom-level features, coordinates, and bond graph.

    Uses RDKit for robust molecular parsing.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Cannot parse ligand: {sdf_path}")

    # Generate 3D coordinates if needed
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    # Basic atom features
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

    # Rotatable bond count
    from rdkit.Chem import rdMolDescriptors
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

    coords = torch.from_numpy(coords).float()
    features = torch.from_numpy(atomic_numbers).float().unsqueeze(-1)

    return {
        'coords': coords,
        'features': features,
        'atomic_numbers': atomic_numbers,
        'n_atoms': coords.shape[0],
        'n_rotatable_bonds': n_rotatable,
        'sdf_path': sdf_path,
    }


def _extract_ca_fallback(pdb_path: str) -> np.ndarray:
    """Extract Cα coordinates from PDB without BioPython."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords) if coords else np.zeros((1, 3))


# ===================================================================
# Reverse Diffusion
# ===================================================================

def run_reverse_diffusion(
    model,
    diffusion,
    schedule: dict,
    e3_data: dict,
    poi_data: dict,
    ligand_data: dict,
    init_ligand: torch.Tensor,
    init_poi: torch.Tensor,
    device: torch.device,
    n_walkers: int,
    save_trajectory: bool = False,
    pocket_guide: bool = False,
    pocket_center: torch.Tensor = None,
) -> dict:
    """
    Run multi-walker reverse diffusion from t=1 to t=0.

    Returns:
        dict with 'final': list of final states per walker
                  'trajectory': list of all intermediate states (if save_trajectory)
    """
    B = n_walkers
    steps = len(schedule['tr_schedule']) - 1

    # Initialize state
    ligand_pos = init_ligand.to(device)  # [B, N_lig, 3]
    poi_pos = init_poi.to(device)        # [B, N_poi, 3]

    traj = [] if save_trajectory else None

    e3_coords = e3_data['coords'].to(device)  # [N_e3, 3]

    for step_idx in range(steps):
        # Current diffusion times
        t = torch.full((B,), schedule['tr_schedule'][step_idx], device=device)
        t_se3 = torch.full((B,), schedule['se3_tr_schedule'][step_idx], device=device)
        dt = schedule['tr_schedule'][step_idx + 1] - schedule['tr_schedule'][step_idx]
        dt_se3 = schedule['se3_tr_schedule'][step_idx + 1] - schedule['se3_tr_schedule'][step_idx]

        # Prepare batch data (B=1 for now, iterate over walkers)
        # For production: batch all walkers through the model
        all_s_q = {'tr': [], 'rot': [], 'tor': []}
        all_s_se3 = []

        for b in range(B):
            batch_data = _prepare_single_input(
                e3_coords, e3_data['features'],
                poi_pos[b], poi_data['features'],
                ligand_pos[b], ligand_data['features'],
                device,
            )

            with torch.no_grad():
                output = model(
                    batch_data['e3'],
                    batch_data['poi'],
                    batch_data['ligand'],
                    t=t[b:b+1],
                    t_se3=t_se3[b:b+1],
                    pocket_guide=pocket_center[b:b+1] if pocket_guide else None,
                )

            all_s_q['tr'].append(output['s_q']['tr'])
            all_s_q['rot'].append(output['s_q']['rot'])
            all_s_q['tor'].append(output['s_q']['tor'])
            all_s_se3.append(output['s_se3'])

        # Stack scores
        tr_score = torch.cat(all_s_q['tr'], dim=0)
        rot_score = torch.cat(all_s_q['rot'], dim=0)
        tor_score = torch.cat(all_s_q['tor'], dim=0) if all_s_q['tor'][0].numel() > 0 else None
        se3_score = torch.cat(all_s_se3, dim=0)

        # Reverse SDE step
        tr_t = torch.full((B,), schedule['tr_schedule'][step_idx], device=device)
        rot_t = torch.full((B,), schedule['rot_schedule'][step_idx], device=device)
        tor_t = torch.full((B,), schedule['tor_schedule'][step_idx], device=device)
        se3_tr_t = torch.full((B,), schedule['se3_tr_schedule'][step_idx], device=device)
        se3_rot_t = torch.full((B,), schedule['se3_rot_schedule'][step_idx], device=device)

        dt_tr = torch.full((B,), abs(dt), device=device)
        dt_rot = torch.full((B,), abs(dt), device=device)
        dt_tor = torch.full((B,), abs(dt), device=device)
        dt_se3_tr = torch.full((B,), abs(dt_se3), device=device)
        dt_se3_rot = torch.full((B,), abs(dt_se3), device=device)

        add_noise = step_idx < steps - 1  # no noise at final step

        ligand_pos = diffusion.reverse_step_protac(
            ligand_pos, tr_score, rot_score,
            tor_score if tor_score is not None else torch.zeros(B, 0, device=device),
            tr_t, rot_t, tor_t, dt_tr, dt_rot, dt_tor,
            add_noise=add_noise,
        )

        poi_pos = diffusion.reverse_step_se3(
            poi_pos, se3_score, se3_tr_t, se3_rot_t,
            dt_se3_tr, dt_se3_rot, add_noise=add_noise,
        )

        if save_trajectory:
            traj.append({
                'step': step_idx,
                't': t[0].item(),
                'ligand_coords': ligand_pos.clone(),
                'poi_coords': poi_pos.clone(),
            })

        if step_idx % 5 == 0 or step_idx == steps - 1:
            print(f"  Step {step_idx+1}/{steps} | t={t[0].item():.3f}")

    final_states = []
    for b in range(B):
        final_states.append({
            'walker_id': b,
            'ligand_coords': ligand_pos[b].cpu(),
            'poi_coords': poi_pos[b].cpu(),
        })

    return {
        'final': final_states,
        'trajectory': traj if save_trajectory else [],
    }


def _prepare_single_input(e3_coords, e3_feat, poi_coords, poi_feat, lig_coords, lig_feat, device):
    """Package single-complex data for model forward."""
    return {
        'e3': {
            'coords': e3_coords.unsqueeze(0).to(device),
            'features': e3_feat.unsqueeze(0).to(device) if e3_feat.dim() == 2 else e3_feat.to(device),
        },
        'poi': {
            'coords': poi_coords.unsqueeze(0).to(device) if poi_coords.dim() == 2 else poi_coords.to(device),
            'features': poi_feat.unsqueeze(0).to(device) if poi_feat.dim() == 2 else poi_feat.to(device),
        },
        'ligand': {
            'coords': lig_coords.unsqueeze(0).to(device) if lig_coords.dim() == 2 else lig_coords.to(device),
            'features': lig_feat.unsqueeze(0).to(device) if lig_feat.dim() == 2 else lig_feat.to(device),
        },
    }


# ===================================================================
# Output
# ===================================================================

def save_results(outdir, run_name, e3_coords, ligand_ensemble, poi_ensemble,
                 trajectory, reflect_results, save_trajectory):
    """Save generated structures as PDB files and metadata as JSON."""

    # Save each walker as PDB
    for i, (lig, poi) in enumerate(zip(ligand_ensemble, poi_ensemble)):
        pdb_path = os.path.join(outdir, f"{run_name}_walker{i:03d}.pdb")
        _write_ternary_pdb(pdb_path, e3_coords, poi, lig, run_name)

    # Save summary
    summary = {
        'run_name': run_name,
        'n_walkers': len(ligand_ensemble),
        'n_valid': sum(r['is_valid'] for r in reflect_results) if reflect_results else None,
        'output_dir': outdir,
    }

    if reflect_results:
        summary['clash_stats'] = {
            'mean_n_clashes': float(np.mean([r['n_clashes'] for r in reflect_results])),
            'max_n_clashes': int(np.max([r['n_clashes'] for r in reflect_results])),
        }

    with open(os.path.join(outdir, f"{run_name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save trajectory (only if requested)
    if save_trajectory and trajectory:
        traj_dir = os.path.join(outdir, f"{run_name}_trajectory")
        os.makedirs(traj_dir, exist_ok=True)
        for frame in trajectory:
            step = frame['step']
            for b, (lig, poi) in enumerate(zip(
                frame['ligand_coords'], frame['poi_coords']
            )):
                if b == 0:  # save first walker trajectory only to save space
                    pdb_path = os.path.join(traj_dir, f"step{step:03d}.pdb")
                    _write_ternary_pdb(pdb_path, e3_coords, poi.cpu(), lig.cpu(), run_name)
                    break


def _write_ternary_pdb(pdb_path, e3_coords, poi_coords, lig_coords, name):
    """Write a ternary complex as a multi-chain PDB file."""
    with open(pdb_path, 'w') as f:
        atom_idx = 1

        # Chain A: E3
        for i, pos in enumerate(e3_coords.cpu().numpy() if hasattr(e3_coords, 'cpu') else e3_coords):
            f.write(f"ATOM  {atom_idx:5d}  CA  ALA A{atom_idx:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1

        # Chain B: POI
        for i, pos in enumerate(poi_coords.cpu().numpy() if hasattr(poi_coords, 'cpu') else poi_coords):
            f.write(f"ATOM  {atom_idx:5d}  CA  ALA B{atom_idx:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1

        # Chain C: PROTAC
        lig_np = lig_coords.cpu().numpy() if hasattr(lig_coords, 'cpu') else np.array(lig_coords)
        for i, pos in enumerate(lig_np):
            f.write(f"ATOM  {atom_idx:5d}  C   LIG C{atom_idx:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1

        f.write("END\n")


if __name__ == "__main__":
    main()
