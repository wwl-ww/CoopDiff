#!/usr/bin/env python
"""Validate dual_native on real input molecules without ESM or model downloads."""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tricoopdiff.diffdock_compat  # noqa: F401
from torch_geometric.data import HeteroData
from tricoopdiff.diffdock.datasets.process_mols import get_lig_graph_with_matching
from tricoopdiff.initialization import (
    INITIAL_NOISE_STD_PROPORTION, TernaryInitializer, sample_binary_pose,
)
from tricoopdiff.rigid import apply_rigid, invert_rigid


def read_case(directory):
    """Preserve indexed SDF atoms and use PDB Cα coordinates for both receptors."""
    from Bio.PDB import PDBParser

    receptors = []
    for name in ("protein1", "protein2"):
        structure = PDBParser(QUIET=True).get_structure(name, directory / f"{name}.pdb")
        receptors.append(torch.tensor(np.array([
            residue['CA'].coord for residue in structure.get_residues() if 'CA' in residue
        ]), dtype=torch.float64))
    mol = Chem.SDMolSupplier(str(directory / 'ligand.sdf'), removeHs=False)[0]
    if mol is None:
        raise ValueError(f"invalid ligand in {directory}")
    graph = HeteroData()
    get_lig_graph_with_matching(mol, graph, popsize=None, maxiter=None,
                               matching=False, keep_original=False,
                               num_conformers=1, remove_hs=True)
    data = {
        'mol': Chem.RemoveHs(mol).ToBinary(),
        'edge_index': graph['ligand', 'ligand'].edge_index,
        'edge_mask': graph['ligand'].edge_mask,
        'mask_rotate': graph['ligand'].mask_rotate,
    }
    return *receptors, data


def validate_case(directory, seeds):
    a, b, data = read_case(directory)
    results = []
    for seed in seeds:
        print(f"[geometry] {directory.name}, seed={seed}", flush=True)
        ligand, poi, body, metadata = TernaryInitializer().initialize(a, b, data, seed)
        sides = metadata['sides']
        ra, rb = (torch.tensor(sides[s]['rotation'], dtype=a.dtype) for s in ('e3', 'poi'))
        ta, tb = (torch.tensor(sides[s]['translation'], dtype=a.dtype) for s in ('e3', 'poi'))
        shared = apply_rigid(ligand[0], *invert_rigid(ra, ta))
        local_b = apply_rigid(shared, rb, tb)
        central_a = apply_rigid(a, *invert_rigid(ra, ta))
        errors = {
            'e3_cdist_max_error': (torch.cdist(a, ligand[0]) - torch.cdist(central_a, shared)).abs().max().item(),
            'poi_cdist_max_error': (torch.cdist(b, local_b) - torch.cdist(poi[0], ligand[0])).abs().max().item(),
            'body_frame_max_error': ((poi - poi.mean(1, keepdim=True)) @ body - (b - b.mean(0))).abs().max().item(),
            'rotation_orthogonality_error': (body[0].T @ body[0] - torch.eye(3)).abs().max().item(),
        }
        assert all(error < 1e-5 for error in errors.values()), errors
        assert torch.det(body[0]) > 0
        results.append({
            'seed': seed, **errors, 'initialization': metadata,
            'minimum_protein_ca_distance': torch.cdist(a, poi[0]).min().item(),
            'n_ligand_atoms': ligand.shape[1], 'n_rotatable_bonds': int(data['edge_mask'].sum()),
        })
    return {'case': directory.name, 'seeds': results}


def official_parity(source):
    """Execute the real upstream initializer function on identical rigid inputs.

    ETKDG/shared torsion are tested separately; no_torsion=True here isolates
    the binary rigid kernel, with each receptor centered as upstream requires.
    """
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'randomize_position')
    module = ast.Module(body=[function], type_ignores=[])
    scope = {'np': np, 'torch': torch, 'R': Rotation}
    exec(compile(module, str(source), 'exec'), scope)
    errors = []
    for seed in range(16):
        torch.manual_seed(90)
        ligand = torch.randn(16, 3)
        receptor = torch.randn(45, 3) * (5 + seed)
        receptor -= receptor.mean(0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rotation, translation, _ = sample_binary_pose(ligand, receptor)
        predicted = apply_rigid(ligand, rotation, translation)
        data = {'ligand': SimpleNamespace(pos=ligand.clone()),
                'receptor': SimpleNamespace(pos=receptor)}
        np.random.seed(seed)
        torch.manual_seed(seed)
        scope['randomize_position'](
            [data], no_torsion=True, no_random=False, tr_sigma_max=19,
            initial_noise_std_proportion=INITIAL_NOISE_STD_PROPORTION,
        )
        error = (predicted - data['ligand'].pos).abs().max().item()
        assert error < 1e-5, (seed, error)
        errors.append(error)
    return {'source': str(source.resolve()),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'n_seeds': len(errors), 'max_error': max(errors),
            'scope': 'rigid kernel, shared conformer fixed, no_torsion=True, CPU float32'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dirs', nargs='+', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 7, 42])
    parser.add_argument('--official-sampling', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {'init_mode': 'dual_native', 'passed': True,
              'cases': [validate_case(path, args.seeds) for path in args.case_dirs]}
    if args.official_sampling:
        report['official_rigid_parity'] = official_parity(args.official_sampling)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"PASS: {len(report['cases'])} cases × {len(args.seeds)} seeds; {args.out}")


if __name__ == '__main__':
    main()
