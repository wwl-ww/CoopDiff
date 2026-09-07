"""Replay selected failed seeds and score every state, without changing weights."""

import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tricoopdiff.diffdock_compat  # noqa: E402,F401

import numpy as np
import torch

from evaluation_common import load_index_entry, resolve_entry_paths
from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from tricoopdiff.deepternary_metrics import compute_fullatom_dockq, _fit_rigid
from tricoopdiff.sampler import sample_single


def geometry(ligand, poi, e3, reference_ligand, reference_poi):
    """Rigid pose errors refer to input templates, not symmetry-corrected poses."""
    rotation, translation = _fit_rigid(poi, reference_poi)
    lig_rotation, lig_translation = _fit_rigid(ligand, reference_ligand)
    ligand_poi = np.linalg.norm(ligand[:, None] - poi[None], axis=-1)
    ligand_e3 = np.linalg.norm(ligand[:, None] - e3[None], axis=-1)
    proteins = np.linalg.norm(e3[:, None] - poi[None], axis=-1)
    return {
        'ligand_com_error_A': float(np.linalg.norm(ligand.mean(0) - reference_ligand.mean(0))),
        'ligand_pose_rmsd_A': float(np.sqrt(np.mean(np.sum((ligand-reference_ligand)**2, axis=-1)))),
        'ligand_aligned_rmsd_A': float(np.sqrt(np.mean(np.sum(
            (ligand @ lig_rotation + lig_translation-reference_ligand)**2, axis=-1)))),
        'poi_com_error_to_input_A': float(np.linalg.norm(poi.mean(0)-reference_poi.mean(0))),
        'poi_rotation_to_input_deg': float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))),
        'poi_aligned_rmsd_A': float(np.sqrt(np.mean(np.sum((poi @ rotation + translation-reference_poi)**2, axis=-1)))),
        'min_e3_ca_ligand_A': float(ligand_e3.min()),
        'min_poi_ca_ligand_A': float(ligand_poi.min()),
        'e3_ca_ligand_pairs_8A': int((ligand_e3 < 8).sum()),
        'poi_ca_ligand_pairs_8A': int((ligand_poi < 8).sum()),
        'protein_ca_pairs_3A': int((proteins < 3).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--protac-root', required=True)
    parser.add_argument('--deepternary-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data_root = Path(args.data_root)
    runner = EvaluationRunner(root, args.checkpoint, deepternary_root=args.deepternary_root,
                              deterministic=args.deterministic)
    specifications = [
        ('protac', '5T35_D_A_759', 0, 'fixed first seed'),
        ('protac', '5T35_D_A_759', 18, 'oracle-best of completed 40 seeds'),
        ('mgd', '3JRQ_B_A_A8S', 0, 'fixed first case'),
        ('mgd', '6Q0R_B_D_O6M', 0, 'best completed MGD case at stop time'),
    ]
    if args.limit is not None:
        if args.limit < 1:
            parser.error('limit must be positive')
        specifications = specifications[:args.limit]
    summary = {'checkpoint_sha256': runner.checkpoint_sha256,
               'reproducibility': runner.reproducibility,
               'deterministic_algorithms': args.deterministic,
               'scope': 'selected post-hoc failure analysis, not an unbiased benchmark',
               'notes': 'CA contacts are geometry diagnostics. Unbound POI errors use prealigned input templates.',
               'runs': []}
    for group, case, seed, selection in specifications:
        index = data_root / ('protac22_unbound.json' if group == 'protac' else 'mgd94.json')
        paths = resolve_entry_paths(load_index_entry(index, case), root)
        native_root = Path(args.protac_root) if group == 'protac' else data_root / 'release/pdbs'
        cache = data_root / f'featurize_{group}'
        e3, poi, ligand, dim = featurize_complex(
            case, str(paths['protein1']), str(paths['protein2']), str(paths['ligand']),
            runner.score_model_args, 'cuda', cache_dir=str(cache))
        runner._ensure_model(dim)
        initial = []
        head_probes = {'tr': [], 'rot': []}
        handles = []
        for component in head_probes:
            def record_head(module, inputs, output, component=component):
                raw = float(inputs[0][0, 0].detach())
                magnitude = float(output.reshape(-1)[0].detach())
                head_probes[component].append({
                    'raw_vector_norm': raw, 'signed_magnitude': magnitude,
                    'normalization_gain': abs(magnitude) / max(raw, 1e-8),
                })
            handles.append(getattr(runner.expert.model, f'{component}_final_layer').register_forward_hook(record_head))
        initialize = runner.initializer.initialize

        def capture(*values, **kwargs):
            result = initialize(*values, **kwargs)
            initial.append(tuple(t[0].detach().cpu().numpy().copy() for t in result[:2]))
            return result

        try:
            with patch.object(runner.initializer, 'initialize', side_effect=capture):
                result = sample_single(
                    runner.model, runner.diffusion, runner.initializer, e3, poi, ligand,
                    device='cuda', steps=20, seed=seed, return_trajectory=True, return_trace=True)
        finally:
            for handle in handles:
                handle.remove()
        if not result['valid'] or len(result['trajectory']) != 20:
            raise RuntimeError(f'incomplete replay for {case}/{seed}')
        states = initial + result['trajectory']
        reference_ligand = ligand['coords'][0].cpu().numpy()
        reference_poi = poi['coords'][0].cpu().numpy()
        e3_coords = e3['coords'][0].cpu().numpy()
        metric_paths = dict(protein1_pdb=paths['protein1'], protein2_pdb=paths['protein2'],
                            ligand_pdb=paths['ligand'], native_complex_pdb=native_root/case/'gt_complex.pdb',
                            deepternary_root=args.deepternary_root)
        if group == 'protac':
            metric_paths.update(metric_protein1_pdb=native_root/case/'protein1.pdb',
                                metric_protein2_pdb=native_root/case/'protein2.pdb')
        rows = []
        for step, (ligand_coords, poi_coords) in enumerate(states):
            row = dict(step=step, **geometry(ligand_coords, poi_coords, e3_coords,
                                           reference_ligand, reference_poi))
            row.update(compute_fullatom_dockq(
                ligand_coords, poi_coords, e3_coords, **metric_paths))
            rows.append(row)
        benchmark_paths = list((root/'results/epoch20_benchmarks'/group).rglob(f'{case}.json'))
        assert len(benchmark_paths) == 1
        benchmark = json.loads(benchmark_paths[0].read_text())
        expected = next(w['dockq'] for w in benchmark['walkers'] if w['seed'] == seed)
        replay_error = abs(rows[-1]['dockq']-expected)
        controls = []
        for repeat in range(2):
            with patch.object(runner.initializer, 'initialize', side_effect=capture):
                plain = sample_single(
                    runner.model, runner.diffusion, runner.initializer,
                    copy.deepcopy(e3), copy.deepcopy(poi), copy.deepcopy(ligand),
                    device='cuda', steps=20, seed=seed, return_trace=False)
            initial_error = max(float(np.max(np.abs(a-b))) for a, b in zip(initial[0], initial[-1]))
            if initial_error > 1e-6:
                raise AssertionError(f'same seed initial coordinates changed: {initial_error}')
            if plain['initialization']['torsion_updates'] != result['initialization']['torsion_updates']:
                raise AssertionError('same seed produced different initial torsions')
            metrics = compute_fullatom_dockq(
                plain['ligand_coords'], plain['poi_coords'], e3_coords, **metric_paths)
            controls.append({'repeat': repeat, 'dockq': metrics['dockq'],
                             'initial_max_coordinate_error_A': initial_error,
                             'ligand_rmsd_to_trace_A': float(np.sqrt(np.mean(np.sum(
                                 (plain['ligand_coords']-states[-1][0])**2, axis=-1)))),
                             'poi_rmsd_to_trace_A': float(np.sqrt(np.mean(np.sum(
                                 (plain['poi_coords']-states[-1][1])**2, axis=-1))))})
        payload = {'case': case, 'group': group, 'seed': seed, 'selection': selection,
                   'reproducibility': runner.reproducibility,
                   'initialization': result['initialization'], 'trace': result['trace'],
                   'states': rows, 'final_dockq_replay_error': replay_error,
                   'historical_dockq': expected, 'historical_replay_matches_1e3': replay_error <= 1e-3,
                   'untraced_repeats': controls}
        payload['head_probes'] = head_probes
        destination = out / f'{case}_seed{seed}'
        destination.with_suffix('.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
        np.savez_compressed(destination.with_suffix('.npz'),
                            ligand=np.stack([s[0] for s in states]),
                            poi=np.stack([s[1] for s in states]), e3=e3_coords,
                            reference_ligand=reference_ligand, reference_poi=reference_poi)
        summary['runs'].append({k: payload[k] for k in ('case', 'group', 'seed', 'selection', 'final_dockq_replay_error')})
        summary['runs'][-1].update(initial=rows[0], final=rows[-1], untraced_repeats=controls,
                                  best_step=max(rows, key=lambda r:r['dockq'])['step'])
        (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(f"DONE {case} seed={seed}: DockQ {rows[0]['dockq']:.5f} -> {rows[-1]['dockq']:.5f}", flush=True)


if __name__ == '__main__':
    main()
