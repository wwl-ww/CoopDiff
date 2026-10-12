#!/usr/bin/env python
"""Measure Matcha anchor-switch sensitivity using local weights and cached cases.

The boundary experiment deliberately deforms one CA to an anchor-selection tie.
It is a stress test, not evidence that native proteins routinely cross that tie.
At identical coordinates, fixed versus reselected anchors isolate frame choice.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tricoopdiff.diffdock_compat  # noqa: E402,F401
from tricoopdiff.runtime import atomic_json, sha256_file


def anchors_and_frame(coords):
    """Match FrozenMatcha.prepare_case/read_pair on the supplied tensor dtype."""
    centered = coords - coords.mean(0)
    first = int(centered.square().sum(-1).argmax())
    e1 = centered[first] / centered[first].norm()
    perpendicular = centered - (centered @ e1)[:, None] * e1
    second = int(perpendicular.square().sum(-1).argmax())
    if perpendicular[second].norm() < 1e-3:
        raise ValueError('degenerate protein anchor frame')
    return (first, second), frame_for(coords, (first, second))


def frame_for(coords, anchors):
    centered = coords - coords.mean(0)
    e1 = centered[anchors[0]] / centered[anchors[0]].norm()
    e2 = centered[anchors[1]] - (centered[anchors[1]] @ e1) * e1
    e2 = e2 / e2.norm()
    return torch.stack((e1, e2, torch.linalg.cross(e1, e2)), -1)


def boundary_pair(coords, epsilon):
    """Move the runner-up radially; bisect the first actual anchor switch."""
    original = coords.double()
    centered = original - original.mean(0)
    order = centered.square().sum(-1).argsort(descending=True)
    first, runner = map(int, order[:2])
    direction = centered[runner] / centered[runner].norm()

    def shifted(amount):
        value = original.clone()
        value[runner] += amount * direction
        return value

    low, high = 0., max(float(centered[first].norm() - centered[runner].norm()) * 2, epsilon * 4)
    for _ in range(40):
        if anchors_and_frame(shifted(high))[0][0] != first:
            break
        high *= 2
    else:
        raise ValueError('could not bracket an anchor switch')
    for _ in range(60):
        middle = (low + high) / 2
        if anchors_and_frame(shifted(middle))[0][0] == first:
            low = middle
        else:
            high = middle
    boundary = (low + high) / 2
    # Inference uses float32; verify the switch survives its rounding.
    minus = shifted(boundary - epsilon).to(coords)
    plus = shifted(boundary + epsilon).to(coords)
    a, fa = anchors_and_frame(minus)
    b, fb = anchors_and_frame(plus)
    if a == b:
        raise ValueError('anchor switch lost to rounding; increase --epsilon')
    cosine = ((torch.trace(fa.T @ fb).double() - 1) / 2).clamp(-1, 1)
    return minus, plus, a, b, {
        'runner_atom': runner, 'boundary_single_ca_shift_angstrom': boundary,
        'boundary_setup_rmsd_angstrom': boundary / len(coords) ** .5,
        'pair_rmsd_angstrom': float((minus - plus).square().sum(-1).mean().sqrt()),
        'pair_max_atom_displacement_angstrom': float((minus - plus).norm(dim=-1).max()),
        'anchors_minus': list(a), 'anchors_plus': list(b),
        'frame_rotation_degrees': float(torch.acos(cosine) * 180 / torch.pi),
        'canonical_coords_rmsd_angstrom': float(((minus - minus.mean(0)) @ fa
            - (plus - plus.mean(0)) @ fb).square().sum(-1).mean().sqrt()),
    }


def difference(first, second):
    a, b = first.detach().double().flatten(), second.detach().double().flatten()
    delta = a - b
    return {'rms': float(delta.square().mean().sqrt()) if delta.numel() else 0.,
            'max_abs': float(delta.abs().max()) if delta.numel() else 0.,
            'relative_l2': float(delta.norm() / a.norm().clamp_min(1e-12)),
            'reference_l2': float(a.norm())}


def evidence_difference(first, second):
    return {key: difference(a, b) for key, a, b in zip(
        ('translation_head_div10', 'rotation_head', 'torsion_head', 'node_scalars', 'node_vectors'),
        first, second)}


def load_case(cache_path):
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    metadata_path = cache_path.parent / 'matcha_metadata' / cache_path.name
    metadata = torch.load(metadata_path, map_location='cpu', weights_only=False)
    result = []
    for key, meta in zip(('e3_data', 'poi_data', 'ligand_data'), metadata['metadata']):
        part = copy.deepcopy(cached[key])
        for field in ('coords', 'features', 'local_geometry'):
            if field in part and part[field].ndim == 2:
                part[field] = part[field].unsqueeze(0)
        part['matcha'] = meta
        result.append(part)
    return result, metadata['identity']['expert']


def variant(data, role, coords, anchors):
    changed = list(data)
    part = data[role]
    changed[role] = {**part, 'coords': coords[None],
                     'matcha': {**part['matcha'], 'anchors': anchors}}
    return changed


def controller_read(model, flow, data, time):
    e3, poi, ligand = data
    t = ligand['coords'].new_tensor([time])
    dt = t.new_tensor([min(.05, time)])
    body = torch.eye(3, dtype=t.dtype)[None]
    output = flow.forward_state(model, e3, poi, ligand, ligand['coords'],
                                poi['coords'], body, t, dt)
    velocities = flow.velocities(flow.normalized_predictions(output, t), e3['coords'], poi['coords'])
    moved_ligand, moved_poi, _ = flow.step(ligand['coords'], poi['coords'], body,
                                           velocities, dt, ligand)
    return velocities, moved_ligand, moved_poi


def controller_difference(first, second):
    a, al, ap = first
    b, bl, bp = second
    report = {f'{branch}_{kind}': difference(a[branch][kind], b[branch][kind])
              for branch in ('e3_ligand', 'poi_ligand') for kind in ('tr', 'rot')}
    report['torsion'] = difference(a['torsion'], b['torsion'])
    report['one_step_ligand_rmsd_angstrom'] = float((al - bl).square().sum(-1).mean().sqrt())
    report['one_step_poi_rmsd_angstrom'] = float((ap - bp).square().sum(-1).mean().sqrt())
    return report


def rollout(model, flow, data, steps):
    e3, poi, ligand = data
    lig, prot = ligand['coords'], poi['coords']
    body = torch.eye(3, dtype=lig.dtype)[None]
    for t, dt in flow.reverse_times(lig, steps):
        output = flow.forward_state(model, e3, poi, ligand, lig, prot, body, t, dt)
        velocity = flow.velocities(flow.normalized_predictions(output, t), e3['coords'], poi['coords'])
        lig, prot, body = flow.step(lig, prot, body, velocity, dt, ligand)
    return lig, prot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--source-dir', default='results/matcha_qualification_source')
    parser.add_argument('--weights-dir', default='results/matcha_qualification_weights')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-weights', choices=('raw', 'ema'), default='raw')
    parser.add_argument('--cases', help='Comma-separated cases; otherwise scan all cached metadata')
    parser.add_argument('--max-cases', type=int, default=3, help='Select smallest first-anchor boundary shifts')
    parser.add_argument('--max-protein-nodes', type=int, default=400)
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Single-CA offset on each side of a forced tie, Angstrom')
    parser.add_argument('--times', default='1,.5,.05')
    parser.add_argument('--rollout-steps', type=int, default=0, help='Optional endpoint comparison, starts at native geometry')
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    times = [float(value) for value in args.times.split(',')]
    if (args.epsilon <= 0 or not np.isfinite(args.epsilon) or args.max_cases < 1
            or args.max_protein_nodes < 3 or args.rollout_steps < 0 or args.threads < 1
            or any(not 0 < time <= 1 for time in times)):
        parser.error('invalid epsilon, limits, rollout steps, threads or time')
    torch.set_num_threads(args.threads)
    torch.manual_seed(73)
    cache_dir = Path(args.cache_dir)
    paths = ([cache_dir / f'{case}.pt' for case in args.cases.split(',')]
             if args.cases else sorted((cache_dir / 'matcha_metadata').glob('*.pt')))
    if not args.cases:
        paths = [cache_dir / path.name for path in paths]
    candidates, excluded = [], []
    for path in paths:
        data, identity = load_case(path)
        for role in (0, 1):
            coords = data[role]['coords'][0]
            if len(coords) > args.max_protein_nodes:
                excluded.append({'case': path.stem, 'role': role, 'reason': 'node_limit'})
                continue
            minus, plus, a, b, geometry = boundary_pair(coords, args.epsilon)
            candidates.append({'case': path.stem, 'role': role, 'data': data, 'identity': identity,
                               'minus': minus, 'plus': plus, 'a': a, 'b': b, 'geometry': geometry,
                               'cache_sha256': sha256_file(path),
                               'metadata_sha256': sha256_file(cache_dir / 'matcha_metadata' / path.name)})
    candidates.sort(key=lambda row: row['geometry']['boundary_single_ca_shift_angstrom'])
    selected = candidates[:args.max_cases]
    if not selected:
        raise ValueError('no eligible cases')
    report = {'format': 'matcha_anchor_sensitivity_v1', 'config': vars(args), 'device': 'cpu',
              'torch_version': torch.__version__, 'script_sha256': sha256_file(__file__),
              'eligible_protein_count': len(candidates), 'excluded': excluded,
              'geometry_scan': [{key: row[key] for key in ('case', 'role', 'geometry')} for row in candidates],
              'experiments': [],
              'scope': 'Native-pose local sensitivity and artificially constructed CA tie boundaries; not docking accuracy.'}
    report['source_identity'] = {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        'sha256': {str(path.relative_to(ROOT)): sha256_file(path)
                   for path in sorted((ROOT / 'tricoopdiff').rglob('*.py'))},
    }
    for row in selected:
        print(json.dumps({'case': row['case'], 'role': row['role'], **row['geometry']}), flush=True)
    if args.geometry_only:
        atomic_json(report, args.out)
        return
    from tricoopdiff.models.expert import FrozenMatcha, validate_expert_base_identity
    from tricoopdiff.models.assembly import TernaryAssembly, TernaryAssemblyModel
    from tricoopdiff.sdpi_flow import SDPIFlow
    expert = FrozenMatcha(args.source_dir, args.weights_dir, device='cpu').eval()
    report['expert_identity'] = expert.provenance
    model, flow = None, SDPIFlow()
    if args.checkpoint:
        from tricoopdiff.train import validate_assembly_checkpoint
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        validate_assembly_checkpoint(payload)
        validate_expert_base_identity(payload, expert)
        controller = TernaryAssembly(**payload['model_config'])
        key = 'trifuse_state_dict' if args.checkpoint_weights == 'raw' else 'ema_trifuse_state_dict'
        controller.load_state_dict(payload[key], strict=True)
        model = TernaryAssemblyModel(expert, controller).eval()
        report['checkpoint'] = {'path': args.checkpoint, 'sha256': sha256_file(args.checkpoint),
                                'global_step': payload['global_step'], 'weights': args.checkpoint_weights}
    with torch.no_grad():
        for row in selected:
            if row['identity'] != expert.provenance:
                raise ValueError(f"{row['case']}: cached Matcha metadata belongs to another expert")
            data, role = row['data'], row['role']
            variants = [variant(data, role, row['minus'], row['a']),
                        variant(data, role, row['plus'], row['a']),
                        variant(data, role, row['plus'], row['b'])]
            item = {key: row[key] for key in ('case', 'role', 'geometry', 'cache_sha256', 'metadata_sha256')}
            item['boundary_reads'] = []
            for time in times:
                reads = [expert.read_pair(parts[role], parts[2], torch.tensor([time])) for parts in variants]
                comparisons = {'geometry_only': evidence_difference(reads[0], reads[1]),
                               'frame_only_same_coords': evidence_difference(reads[1], reads[2]),
                               'combined': evidence_difference(reads[0], reads[2])}
                entry = {'time': time, 'unconditioned_expert': comparisons}
                if model is not None:
                    actions = [controller_read(model, flow, parts, time) for parts in variants]
                    entry['controller'] = {
                        name: controller_difference(actions[a], actions[b])
                        for name, a, b in (('geometry_only', 0, 1), ('frame_only_same_coords', 1, 2), ('combined', 0, 2))}
                item['boundary_reads'].append(entry)
                print(json.dumps({'case': row['case'], 'role': role, 'time': time,
                    'frame_only_scalar_relative_l2': comparisons['frame_only_same_coords']['node_scalars']['relative_l2'],
                    'geometry_only_scalar_relative_l2': comparisons['geometry_only']['node_scalars']['relative_l2']}), flush=True)
            coords = data[role]['coords'][0]
            original_anchors = tuple(data[role]['matcha']['anchors'])
            base = expert.read_pair(data[role], data[2], torch.tensor([.5]))
            item['ordinary_perturbations'] = []
            generator = torch.Generator().manual_seed(73)
            for amplitude in (1e-4, 1e-3, 1e-2, .1):
                noise = torch.randn(coords.shape, generator=generator)
                noise -= noise.mean(0)
                noise *= amplitude / noise.square().sum(-1).mean().sqrt()
                changed = coords + noise
                changed_anchors, _ = anchors_and_frame(changed)
                current = variant(data, role, changed, changed_anchors)
                fixed = variant(data, role, changed, original_anchors)
                read = expert.read_pair(current[role], current[2], torch.tensor([.5]))
                fixed_read = expert.read_pair(fixed[role], fixed[2], torch.tensor([.5]))
                item['ordinary_perturbations'].append({'requested_rmsd_angstrom': amplitude,
                    'actual_rmsd_angstrom': float((changed - coords).square().sum(-1).mean().sqrt()),
                    'anchor_switched': changed_anchors != original_anchors,
                    'anchors': list(changed_anchors), 'combined': evidence_difference(base, read),
                    'frame_only_same_coords': evidence_difference(fixed_read, read)})
            if model is not None and args.rollout_steps:
                endpoints = [rollout(model, flow, parts, args.rollout_steps) for parts in variants]
                item['rollout'] = {name: {
                    'ligand_rmsd_angstrom': float((endpoints[a][0] - endpoints[b][0]).square().sum(-1).mean().sqrt()),
                    'poi_rmsd_angstrom': float((endpoints[a][1] - endpoints[b][1]).square().sum(-1).mean().sqrt())}
                    for name, a, b in (('geometry_only', 0, 1), ('frame_only_same_coords', 1, 2), ('combined', 0, 2))}
            report['experiments'].append(item)
            atomic_json(report, args.out)
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
