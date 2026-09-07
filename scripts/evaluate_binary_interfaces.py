"""Independent DiffDock docking of each receptor against the full ligand.

Consumes explicit, versioned feature caches. Native coordinates are used only
for metrics; no ternary assembly, FiLM, LoRA, Fuse or flow conversion is used.
Interface RMSD is an atom-indexed CA-contact diagnostic, not DockQ or a
chemically annotated warhead RMSD. Samples are unranked (no confidence model).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import tricoopdiff.diffdock_compat  # noqa: F401
from tricoopdiff.diffusion import InterfaceDiffusion
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.graphs import build_subgraph_e3l, crop_receptor_to_ligand
from tricoopdiff.initialization import sample_binary_pose, validate_ligand_graph
from tricoopdiff.model import _set_time_embeddings
from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles


def fresh_ligand(ligand, receptor, seed):
    """Chemical-graph-only ETKDG, uniform torsions and native blind rigid prior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mol = ligand['mol']
    mol = Chem.Mol(mol)
    bonds, masks = validate_ligand_graph(
        mol, ligand['edge_index'], ligand['edge_mask'], ligand['mask_rotate'])
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(np.random.randint(0, 2**31 - 1))
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError('ETKDG failed; no native conformer fallback')
    n = ligand['features'].shape[1]
    pos = torch.as_tensor(mol.GetConformer().GetPositions()[:n],
                          dtype=receptor.dtype)
    if len(bonds):
        pos = modify_conformer_torsion_angles(
            pos, bonds, masks, np.random.uniform(-np.pi, np.pi, len(bonds)))
    pos = pos.to(receptor.device)
    rotation, translation, metadata = sample_binary_pose(pos, receptor)
    return (pos @ rotation.T + translation).unsqueeze(0), metadata


def interface_metrics(predicted, native, receptor, cutoff=8.0):
    """No ligand fitting: the receptor is fixed throughout independent docking."""
    native_dist = torch.cdist(native[0], receptor[0])
    predicted_dist = torch.cdist(predicted[0], receptor[0])
    contact = native_dist < cutoff
    mask = contact.any(dim=1)
    errors = (predicted - native).square().sum(-1)[0]
    return {
        'ligand_rmsd': errors.mean().sqrt().item(),
        'interface_atom_rmsd': errors[mask].mean().sqrt().item() if mask.any() else None,
        'interface_atom_count': int(mask.sum()),
        'ca_contact_recall': (float(((predicted_dist < cutoff) & contact).sum()
                                    / contact.sum()) if contact.any() else None),
    }


@torch.no_grad()
def dock(expert, process, receptor, ligand, seed, steps):
    pos, init = fresh_ligand(ligand, receptor['coords'][0], seed)
    initial = pos.clone()
    schedule = process.get_inference_schedule(steps)
    times = schedule['tr_schedule']
    # DiffDock v1.1 defaults: length-20 schedule, actual_steps=19, final dt=t.
    actual_steps = steps - 1
    trace = []
    for i in range(actual_steps):
        t = torch.as_tensor(times[i], device=pos.device, dtype=pos.dtype).reshape(1)
        dt = (t - float(times[i + 1]) if i < actual_steps - 1 else t.clone())
        tr_sigma, _, _ = process.t_to_sigma(t, t, t)
        cutoff = tr_sigma[0] * 3 + 20 if expert.model.dynamic_max_cross else None
        crop = crop_receptor_to_ligand(
            receptor['coords'][0], receptor['features'][0], pos[0],
            rec_edge_index=receptor.get('edge_index'), cutoff=cutoff,
            min_residues=1, min_residues_with_edges=8, require_internal_edges=True)
        graph = build_subgraph_e3l(
            crop['coords'], crop['features'], pos[0], ligand['features'][0],
            rec_edge_index=crop['edge_index'], lig_edge_index=ligand['edge_index'],
            lig_edge_attr=ligand['edge_attr'], lig_edge_mask=ligand['edge_mask'],
            lig_mask_rotate=ligand['mask_rotate'])
        _set_time_embeddings(graph, t, expert.model)
        tr, rot, tor = expert(graph)[:3]
        if not all(torch.isfinite(x).all() for x in (tr, rot, tor)):
            return None, initial, {'failed_step': i, 'reason': 'nonfinite_native_score',
                                   'trace': trace, 'initialization': init}
        rigid, target = process.reverse_step_protac(
            pos, tr, rot, tor, t, t, t, dt, dt, dt, add_noise=i < actual_steps - 1)
        pos = process.reverse_step_torsion(
            rigid, tor.reshape(1, -1), t, dt, ligand['edge_index'],
            ligand['edge_mask'], ligand['mask_rotate'],
            add_noise=i < actual_steps - 1, rigid_target_pos=target)
        if not torch.isfinite(pos).all():
            return None, initial, {'failed_step': i, 'reason': 'nonfinite_coordinates'}
        trace.append({'step': i, 't': float(t.item()),
                      **interface_metrics(pos, ligand['coords'], receptor['coords'])})
    return pos, initial, {'trace': trace, 'initialization': init}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', action='append', required=True,
                        help='Explicit case .pt feature cache; repeat for multiple cases')
    parser.add_argument('--out', required=True)
    parser.add_argument('--expert-model-dir', required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 2:
        parser.error('samples >= 1 and steps >= 2 required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        parser.error('this experiment is restricted to CUDA_VISIBLE_DEVICES=3')
    if os.environ.get('NAN_CLAMP'):
        parser.error('native baseline requires NAN_CLAMP unset')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    expert = build_diffdock_expert(args.expert_model_dir, 'cuda').eval()
    if expert.model.no_torsion:
        raise ValueError('this baseline requires the standard torsion-enabled expert')
    assert not any(p.requires_grad for p in expert.parameters())
    assert not any('lora' in n for n, _ in expert.named_parameters())
    process = InterfaceDiffusion(inference_steps=args.steps)
    metadata = {
        **vars(args), 'expert': expert.provenance,
        'commit': subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parent.parent), 'rev-parse', 'HEAD'],
            text=True).strip(),
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'protocol': 'independent full-ligand blind docking; native SDE; no learned coupling',
        'metric': 'indexed RMSD in fixed receptor frame; interface=within 8A of native CA',
        'ranking': 'unranked; best-of-N is oracle, not confidence-ranked top-1',
        'native_source': 'https://github.com/gcorso/DiffDock/blob/v1.1/utils/sampling.py',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    results = []
    for cache in args.cache:
        payload = torch.load(cache, map_location='cuda', weights_only=False)
        if payload.get('cache_version') != 5 or not payload.get('source_signature'):
            raise ValueError(f'{cache}: expected signed feature cache v5')
        for component in ('e3_data', 'poi_data', 'ligand_data'):
            for name in ('coords', 'features'):
                value = payload[component][name]
                if value.ndim == 2:
                    payload[component][name] = value.unsqueeze(0)
        ligand = payload['ligand_data']
        case = Path(cache).stem
        for branch in ('e3_data', 'poi_data'):
            receptor = payload[branch]
            for sample in range(args.samples):
                seed = args.seed + sample
                pos, initial, detail = dock(expert, process, receptor, ligand, seed, args.steps)
                row = {'case': case, 'branch': branch, 'sample': sample, 'seed': seed,
                       'valid': pos is not None, 'cache': cache,
                       'source_signature': payload['source_signature'],
                       'initial': interface_metrics(initial, ligand['coords'], receptor['coords']),
                       'final': (interface_metrics(pos, ligand['coords'], receptor['coords'])
                                 if pos is not None else None), **detail}
                results.append(row)
                (output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
                torch.save({'initial': initial.cpu(), 'final': pos.cpu() if pos is not None else None},
                           output / f'{case}_{branch}_{sample}.pt')
                print(json.dumps({k: row[k] for k in ('case', 'branch', 'sample', 'valid', 'final')}), flush=True)


if __name__ == '__main__':
    main()
