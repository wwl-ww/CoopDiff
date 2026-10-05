"""Paired fixed-state evaluation of protected binary and learned local motion.

Uses production validate_step. Native coordinates are scoring-only. The four
validation cases are a small development set, not a final benchmark.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import train_entry as entry
from tricoopdiff.geometry.rigid import proper_rigid_fit
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.initialization.endpoints import case_seed, ligand_symmetries, reachable_ligand_target
from tricoopdiff.metrics.dockq import compute_fullatom_dockq
from tricoopdiff.runtime import atomic_json, atomic_torch_save, configure_inference_reproducibility
from tricoopdiff.train import validate_consensus_checkpoint

TRAIN_CASES = ['1AFQ_B_C_0FG', '1ALZ_A_B_EOH', '1BHX_B_F_R56', '1HKD_A_B_GYP',
               '4HIH_C_D_RAM', '1CJT_A_B_DAD', '1JLL_B_D_COA', '1D8D_B_P_FII']
VAL_CASES = ['1GKA_A_B_AXT', '7ALB_A_B_QBQ', '1JCQ_A_B_739', '1TN6_B_C_FII']


def state_hash(state):
    return hashlib.sha256(b''.join(state[key].detach().cpu().contiguous().numpy().tobytes()
                                  for key in ('ligand', 'poi', 'body_frame'))).hexdigest()


def component_ids(ld):
    count = ld['coords'].shape[1]
    edges = ld['edge_index'].detach().cpu().T.tolist()
    adjacent = [set() for _ in range(count)]
    for a, b in edges:
        adjacent[a].add(b)
    unseen, groups = set(range(count)), []
    while unseen:
        pending, group = [min(unseen)], set()
        while pending:
            atom = pending.pop()
            if atom in group:
                continue
            group.add(atom)
            pending.extend(adjacent[atom] - group)
        unseen -= group
        groups.append(sorted(group))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'results/model41_cross_system_pilot')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--weights', choices=['baseline', 'raw', 'ema'], default='baseline')
    parser.add_argument('--split', choices=['validation', 'train', 'both'], default='validation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--output-label')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.weights != 'baseline' and args.checkpoint is None:
        parser.error('learned evaluation requires --checkpoint')
    args.root.mkdir(parents=True, exist_ok=True)
    out = args.root / 'paired_evaluation' / (args.output_label or args.weights)
    out.mkdir(parents=True, exist_ok=True)
    reproducibility = configure_inference_reproducibility(deterministic=True)
    device = torch.device(args.device)
    train_index, val_index = [ROOT / 'results/model45_partition' / name for name in ('ternary_train.json', 'ternary_val.json')]
    cache = ROOT.parent / 'CoopDiff/results/featurize_cache_proposal_oracle'
    expert = ROOT.parent / 'CoopDiff/checkpoints/score_model'
    native = ROOT / 'DeepTernary/data/TernaryDB/pdbs'
    build_args = entry.parse_args(['--train_data', str(train_index), '--val_data', str(val_index),
        '--output_dir', str(out), '--training-arm', 'A1', '--no-random-flip-proteins',
        '--epochs', '5', '--batch_size', '1', '--seed', '73', '--lr', '1e-4', '--node_dim', '128',
        '--expert_model_dir', str(expert), '--cache_dir', str(cache), '--cache-only',
        '--native-root', str(native), '--nan-clamp', '100', '--device', args.device])
    entries = {item['case']: item for path in (train_index, val_index) for item in entry._load_entries(path)}
    chosen = ((VAL_CASES if args.split != 'train' else []) + (TRAIN_CASES if args.split != 'validation' else []))
    if args.cases:
        if not set(args.cases).issubset(chosen):
            raise ValueError('selected cases must belong to the requested split')
        chosen = [case for case in chosen if case in args.cases]
    score_args = entry._load_score_model_args(expert)
    entry.seed_training(73, device)
    data = {}
    for case in chosen:
        item = entries[case]
        ed, pd, ld, dim = entry.featurize_complex(case, item['protein1'], item['protein2'], item['ligand'],
            score_args, args.device, cache_dir=cache, cache_read_only=True, cache_only=True)
        data[case] = (ed, pd, ld, dim)
    model, flow, trainer = entry.build_trainer(build_args, device, input_dim=data[chosen[0]][3])
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        validate_consensus_checkpoint(checkpoint)
        assert checkpoint['model_config'] == model.trifuse.config
        model.trifuse.load_state_dict(checkpoint['trifuse_state_dict'], strict=True)
        trainer.ema = checkpoint['ema_trifuse_state_dict']
    model.eval()
    initializer = TernaryInitializer(no_torsion=model.expert.model.no_torsion)
    init_path = args.root / ('validation_initial_states.pt' if args.split != 'train' else 'train_initial_states.pt')
    states = torch.load(init_path, map_location='cpu', weights_only=False) if init_path.exists() else {}
    for case in chosen:
        if case not in states:
            ed, pd, ld, _ = data[case]
            ligand, poi, body, _ = initializer.initialize(ed['coords'][0], pd['coords'][0], ld, seed=case_seed(case, 0))
            state = {key: value.detach().cpu() for key, value in zip(('ligand', 'poi', 'body_frame'), (ligand, poi, body))}
            states[case] = {'initial_state': state, 'seed': case_seed(case, 0), 'state_hash': state_hash(state)}
        assert states[case]['state_hash'] == state_hash(states[case]['initial_state']), 'saved initial state hash mismatch'
    atomic_torch_save(states, init_path)
    print(json.dumps({'event': 'initial_states_ready', 'path': str(init_path), 'cases': list(states)}), flush=True)
    local = model.trifuse.local_compatibility
    if args.weights == 'baseline':
        def disabled(self, graph, scales, no_torsion=False):
            width = 12 + (0 if no_torsion else int(graph['lig_edge_mask'].sum()))
            coords = graph['node_coords']
            return {'vector': coords.new_zeros(width), 'local_cost': coords.new_zeros(()),
                    'diagnostics': {'pair_cost': coords.new_zeros(()), 'triplet_cost': coords.new_zeros(()),
                                    'cutoff': self.cutoff, 'mobility': self.mobility}}
        local.forward = MethodType(disabled, local)
    protocol = {'source_commit': '70f1e92', 'reproducibility': reproducibility, 'weights': args.weights, 'split': args.split,
                'checkpoint': str(args.checkpoint) if args.checkpoint else None,
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                'global_step': checkpoint.get('global_step') if checkpoint else 0,
                'cases': chosen, 'actual_initial_states': str(init_path), 'case_rng': 'case_seed(case,0), explicit initializer seed',
                'evaluation': 'production trainer.validate_step with coupling_override; one initial state per case',
                'validation_scope': 'four development cases; not a held-out final benchmark',
                'local_disabled': args.weights == 'baseline', 'repeat': args.repeat,
                'validation_rng': 'case_seed(case,0) reset immediately before each validate_step'}
    atomic_json(protocol, out / 'protocol.json')
    summary = []
    for repeat_index, case in [(repeat, case) for repeat in range(args.repeat) for case in chosen]:
        result_path = out / (f'{case}.json' if args.repeat == 1 else f'{case}_repeat{repeat_index}.json')
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous['initial_state_hash'] != states[case]['state_hash'] or previous['protocol']['checkpoint_sha256'] != protocol['checkpoint_sha256']:
                raise ValueError('refusing to overwrite a differently paired evaluation')
            summary.append(previous)
            continue
        started = time.monotonic()
        ed, pd, ld, _ = data[case]
        initial = {key: value.to(device) for key, value in states[case]['initial_state'].items()}
        full = entry.fullatom_targets(entries[case], ed, pd)
        context = entry.validation_metric_context(entries[case], native)
        _, projection, candidates = reachable_ligand_target(ld, initial['ligand'], initial['poi'], pd['coords'],
            no_torsion=model.expert.model.no_torsion, symmetries=ligand_symmetries(ld), return_candidates=True)
        candidates = torch.as_tensor(candidates, device=device, dtype=ld['coords'].dtype)
        full_e3, full_poi = [torch.as_tensor(full[key], device=device, dtype=ld['coords'].dtype) for key in ('e3', 'poi')]
        true_pairs = {}
        for name, left, right in (('e3_poi', full_e3, full_poi), ('e3_ligand', full_e3, ld['coords'][0]), ('poi_ligand', full_poi, ld['coords'][0])):
            true_pairs[name] = (torch.cdist(left, right) < 5.).nonzero(as_tuple=True)
        groups = component_ids(ld)
        n = ld['coords'].shape[1]
        adjacency = torch.zeros(n, n, device=device, dtype=torch.double)
        edge = ld['edge_index']
        adjacency[edge[0], edge[1]] = 1.
        adjacency[edge[1], edge[0]] = 1.
        adjacency.fill_diagonal_(0.)
        eye = torch.eye(n, device=device, dtype=torch.double)
        communication = torch.linalg.solve(eye + torch.diag(adjacency.sum(-1)) - adjacency, eye).to(ld['coords'])
        trace = []
        def observe(graph, output=None, dt=0.):
            coords, types = graph['node_coords'].detach(), graph['node_types']
            e3, poi, ligand = [coords[types == kind] for kind in (0, 1, 2)]
            pp = torch.cdist(e3, poi)
            pl = [torch.cdist(ligand, protein) for protein in (e3, poi)]
            touches = [(distance < local.cutoff).any(-1) for distance in pl]
            occupancies = []
            for distance in pl:
                radius = distance / local.cutoff
                count = ((1-radius).clamp_min(0).pow(4) * (1+4*radius)).sum(-1)
                occupancies.append(count / (1+count))
            bridge = .5 * (occupancies[0] * (communication @ occupancies[1]) + occupancies[1] * (communication @ occupancies[0])).sum()
            rotation, translation, _ = proper_rigid_fit(pd['coords'], poi[None])
            actual_full_poi = full_poi @ rotation[0].T + translation[0]
            recall = {}
            for name, left, right in (('e3_poi', full_e3, actual_full_poi), ('e3_ligand', full_e3, ligand), ('poi_ligand', actual_full_poi, ligand)):
                a, b = true_pairs[name]
                recall[name + '_native_contact_recall'] = float(((left[a]-right[b]).square().sum(-1) < 25.).float().mean()) if len(a) else None
            record = {'pp_support_pairs': int((pp < local.cutoff).sum()),
                      'pp_support_fraction': float((pp < local.cutoff).float().mean()),
                      'e3_ligand_support_pairs': int((pl[0] < local.cutoff).sum()),
                      'poi_ligand_support_pairs': int((pl[1] < local.cutoff).sum()),
                      'e3_ligand_atom_coverage': float(touches[0].float().mean()),
                      'poi_ligand_atom_coverage': float(touches[1].float().mean()),
                      'same_component_bridged_support': any(bool(touches[0][g].any() and touches[1][g].any()) for g in groups),
                      'effective_bridge_gate_sum': float(bridge), **recall,
                      'poi_native_rmsd_angstrom': float((poi-pd['coords'][0]).square().sum(-1).mean().sqrt()),
                      'ligand_projected_rmsd_angstrom': float((candidates-ligand).square().sum(-1).mean(-1).min().sqrt())}
            if output is not None:
                correction = output['local_correction'].detach()
                record.update(local_action_norm=float(correction.norm()),
                              local_rigid_norms=correction[:12].reshape(4,3).norm(dim=-1).tolist(),
                              local_torsion_abs_max=float(correction[12:].abs().max()) if len(correction)>12 else 0.,
                              local_cost=float(output['local_cost'].detach()), dt=dt)
            return record
        initial_graph = {'node_coords': torch.cat((ed['coords'][0], initial['poi'][0], initial['ligand'][0])),
                         'node_types': torch.cat([torch.full((len(value),), kind, device=device) for kind, value in enumerate((ed['coords'][0],initial['poi'][0],initial['ligand'][0]))])}
        with torch.no_grad():
            initial_metrics = {**observe(initial_graph), **compute_fullatom_dockq(initial['ligand'][0].cpu().numpy(), initial['poi'][0].cpu().numpy(), ed['coords'][0].cpu().numpy(), **context)}
        def hook(module, inputs, output):
            with torch.no_grad():
                trace.append(observe(inputs[0], output, float(inputs[2][0, 5])))
        handle = model.trifuse.register_forward_hook(hook)
        terminal_metrics = []
        original_compute_loss = trainer.compute_loss
        def capture_terminal(**batch):
            result = original_compute_loss(**batch)
            if 'predicted_ligand' in result:
                terminal_graph = {**initial_graph, 'node_coords': torch.cat((ed['coords'][0],
                    result['predicted_poi'][0], result['predicted_ligand'][0]))}
                terminal_metrics.append(observe(terminal_graph))
            return result
        trainer.compute_loss = capture_terminal
        try:
            entry.seed_training(case_seed(case, 0), device)
            final = trainer.validate_step(case=case, e3_data=ed, poi_data=pd, ligand_data=ld, source=0,
                fullatom_targets=full, coupling_override={'initial_state': initial}, dockq_context=context,
                weights='raw' if args.weights == 'baseline' else args.weights)
        finally:
            handle.remove()
            trainer.compute_loss = original_compute_loss
        if terminal_metrics:
            final.update(terminal_metrics[-1])
        projection = json.loads(json.dumps(projection, default=lambda value: value.item()))
        record = {'case': case, 'repeat_index': repeat_index, 'split': 'validation' if case in VAL_CASES else 'train',
                  'initial_state_hash': states[case]['state_hash'], 'seed': states[case]['seed'],
                  'protocol': protocol, 'initial': initial_metrics, 'final': final, 'projection': projection,
                  'trajectory': trace, 'elapsed_seconds': time.monotonic()-started}
        if args.weights != 'baseline':
            baseline_path = args.root / 'paired_evaluation/baseline' / f'{case}.json'
            if not baseline_path.exists():
                raise FileNotFoundError(f'paired baseline missing for {case}')
            baseline = json.loads(baseline_path.read_text())
            assert baseline['initial_state_hash'] == record['initial_state_hash'], 'unpaired actual initial states'
            record['paired_deltas'] = {key: final[key] - baseline['final'][key] for key in (
                'dockq', 'poi_native_rmsd_angstrom', 'ligand_projected_rmsd_angstrom',
                'e3_ligand_contact_mae_angstrom', 'poi_ligand_contact_mae_angstrom')}
            absent = all(row['pp_support_pairs'] == 0 and not row['same_component_bridged_support']
                         for row in baseline['trajectory'])
            if absent:
                record['zero_support_invariant_pass'] = (all(row['local_action_norm'] == 0. for row in trace)
                    and abs(record['paired_deltas']['dockq']) < 1e-9
                    and abs(record['paired_deltas']['poi_native_rmsd_angstrom']) < 1e-5
                    and abs(record['paired_deltas']['ligand_projected_rmsd_angstrom']) < 1e-5)
        atomic_json(record, result_path)
        summary.append(record)
        atomic_json(summary, out / 'summary.json')
        print(json.dumps({'event':'case_complete','case':case,'weights':args.weights,
                          'dockq':final['dockq'],'ligand_rmsd':final['ligand_projected_rmsd_angstrom'],
                          'initial_hash':states[case]['state_hash'],'seconds':record['elapsed_seconds']}), flush=True)
    atomic_json(summary, out / 'summary.json')


if __name__ == '__main__':
    main()
