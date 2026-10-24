"""Compare learned and independently observed binary anchor roles on PROTACs."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assembly_pipeline import (candidate_import_paths, load_case,
    model_for, read_payload)
from scripts.protac_dual_anchor_init import _binary_ligand
from scripts.train_entry import _load_score_model_args
from tricoopdiff.initialization.candidates import (
    generate_etkdg_ligand_at_hdock_interface, load_assembly_candidate)
from tricoopdiff.initialization.protac import (
    match_external_binary_anchors, validate_anchor_graph_identity)
from tricoopdiff.metrics.dockq import _featurized_ca_coords, compute_fullatom_dockq


def binary_roles(entry, ligand, ccd_dir):
    """Read warhead atoms and pockets from binary complexes, never ternary targets."""
    directory = Path(entry['protein1']).parent
    full = Chem.MolFromMolFile(entry['ligand'], removeHs=True, sanitize=True)
    if full is None:
        raise ValueError('invalid PROTAC chemical graph')
    full.RemoveAllConformers()
    fragments = [_binary_ligand(directory/f'unbound_lig{i}.pdb', ccd_dir)
                 for i in (1, 2)]
    mapping = match_external_binary_anchors(full, *fragments,
        min_coverage=.75, geometry_limit=1)
    if mapping is None:
        raise ValueError('binary warheads cannot be mapped to the PROTAC graph')
    validate_anchor_graph_identity(mapping, ligand['mol'])
    roles = {}
    for side, branch in enumerate(('e3_ligand', 'poi_ligand'), 1):
        ligand_mask = np.zeros(full.GetNumAtoms(), dtype=bool)
        ligand_mask[list(mapping[f'anchor{side}_atoms'])] = True
        protein_ca = _featurized_ca_coords(entry[f'protein{side}'])
        fragment_xyz = fragments[side-1].GetConformer().GetPositions()
        protein_mask = cKDTree(fragment_xyz).query(protein_ca)[0] < 8.
        if protein_mask.sum() < 1:
            raise ValueError(f'protein {side} binary fragment has no C-alpha contacts')
        roles[branch] = {'ligand': ligand_mask, 'protein': protein_mask}
    return {'contract': 'external_binary_anchor_roles_v1', 'roles': roles,
            'binary_core_coverage': mapping['binary_core_coverage'],
            'native_ternary_coordinates_used': False}


def align_pose_atom_records(pose, protein2, scratch):
    """Drop source hydrogens absent from a heavy-atom binary pose."""
    pose, protein2 = Path(pose), Path(protein2)
    def key(line):
        return line[22:27], line[17:20], line[12:16]

    pose_lines = pose.read_text().splitlines()
    source_lines = protein2.read_text().splitlines()
    posed = {key(line) for line in pose_lines
             if line.startswith('ATOM') and line[21] == 'B'}
    source = {key(line) for line in source_lines if line.startswith('ATOM')}
    common = posed & source
    if not common or len(common) != len(posed):
        raise ValueError('binary POI pose contains atoms absent from its monomer template')
    if common == source:
        return pose, protein2
    filtered = scratch/'protein2_matching_atoms.pdb'
    filtered.write_text('\n'.join(line for line in source_lines
        if line.startswith('ATOM') and key(line) in common)+'\nEND\n')
    return pose, filtered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--hdock-bank', type=Path)
    sources.add_argument('--anchor-candidate-root', type=Path)
    parser.add_argument('--ccd-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--score-model-dir', type=Path, required=True)
    parser.add_argument('--deepternary-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.index.resolve().parents[2]
    entries = json.loads(args.index.read_text())['entries']
    if args.case:
        entries = [entry for entry in entries if entry['case'] in args.case]
    score_args = _load_score_model_args(args.score_model_dir)
    opt = Namespace(cache_dir=args.cache_dir, feature_cache_source=None,
                    cached_only=True, device=args.device, objective='drifting',
                    motion_contract=None, weights_from=None, node_dim=128)
    payload, _ = read_payload(args.checkpoint, args.device)
    model = None
    rows = []
    for raw in entries:
        entry = {key: str((root/value).resolve()) if key in
                 ('protein1', 'protein2', 'ligand') else value
                 for key, value in raw.items()}
        case = entry['case']
        try:
            # Mapping is checked before feature-cache admission so excluded
            # compounds report the actual binary-anchor failure.
            full = Chem.MolFromMolFile(entry['ligand'], removeHs=True, sanitize=True)
            if full is None:
                raise ValueError('invalid PROTAC chemical graph')
            condition = binary_roles(entry, {'mol': full}, args.ccd_dir)
            e3, poi, ligand, input_dim = load_case(opt, entry, score_args)
            if model is None:
                model = model_for(opt, input_dim, payload).eval()
            scratch = args.output/case
            scratch.mkdir(exist_ok=True)
            source = ((args.hdock_bank/case/'model_1.pdb') if args.hdock_bank else
                      (args.anchor_candidate_root/case/'candidate_1_proteins.pdb'))
            ligand_pose = (None if args.hdock_bank else
                           args.anchor_candidate_root/case/'candidate_1.sdf')
            posed = scratch/'source_pose.pdb'
            if not posed.exists():
                posed.symlink_to(source.resolve())
            posed, p1, p2 = candidate_import_paths(entry, scratch, posed, e3, poi)
            posed, p2 = align_pose_atom_records(posed, p2, scratch)
            candidate = load_assembly_candidate(posed, ligand_pose,
                e3['coords'], poi['coords'], ligand, candidate_id=case,
                rank=1, protein1_path=p1, protein2_path=p2)
            if args.hdock_bank:
                candidate = generate_etkdg_ligand_at_hdock_interface(
                    e3, ligand, candidate, seed=73)
            target = root/'DeepTernary/output/protac22'/case/'gt_complex.pdb'
            variants = {'start': candidate}
            with torch.inference_mode():
                variants['predicted'] = model.generate_candidate(e3, poi, ligand, candidate)
                variants['supplied'] = model.generate_candidate(
                    e3, poi, ligand, candidate, conditions=condition)
            scores = {}
            for name, pose in variants.items():
                metric = compute_fullatom_dockq(
                    pose['ligand'][0].detach().cpu().numpy(),
                    pose['poi'][0].detach().cpu().numpy(),
                    e3['coords'][0].detach().cpu().numpy(),
                    entry['protein1'], entry['protein2'], entry['ligand'],
                    target, args.deepternary_root,
                    metric_protein1_pdb=str(root/raw['metric_protein1']),
                    metric_protein2_pdb=str(root/raw['metric_protein2']),
                    pred_poi_rotation=pose['rotation'][0].detach().cpu().numpy())
                rmsd = (pose['ligand']-ligand['coords']).square().sum(-1).mean().sqrt()
                scores[name] = {'dockq': metric['dockq'], 'ligand_rmsd': float(rmsd)}
            row = {'case': case, 'scores': scores,
                   'protein_anchor_counts': [int(condition['roles'][branch]['protein'].sum())
                       for branch in ('e3_ligand', 'poi_ligand')],
                   'ligand_anchor_counts': [int(condition['roles'][branch]['ligand'].sum())
                       for branch in ('e3_ligand', 'poi_ligand')],
                   'binary_core_coverage': condition['binary_core_coverage']}
            rows.append(row)
            print(json.dumps(row), flush=True)
        except Exception as exc:
            row = {'case': case, 'error': f'{type(exc).__name__}: {exc}'}
            rows.append(row)
            print(json.dumps(row), flush=True)
        (args.output/'summary.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
