#!/usr/bin/env python
"""Compare two checkpoints on identical states from paired exact rollouts."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from compare_rollout_trajectories import trajectory_rows
from local_velocity_audit import ARTIFACTS, build, load_batch
from train_entry import _load_score_model_args
from tricoopdiff.joint_endpoint_experiment import _proper_rigid_fit
from tricoopdiff.reproducibility import configure_inference_reproducibility


def load_report(folder):
    report = json.loads((folder / 'report.json').read_text())
    if report['status'] != 'complete' or report['weights'] != 'raw':
        raise ValueError('paired state audit requires complete raw exact reports')
    return report


def load_state(folder, case, step, device):
    values = np.load(folder / case / 'coordinates.npz')
    if step == 0:
        ligand, poi = values['initial_ligand'], values['initial_poi']
    else:
        ligand = values['trajectory_ligand'][step - 1:step]
        poi = values['trajectory_poi'][step - 1:step]
    return (torch.as_tensor(ligand, device=device),
            torch.as_tensor(poi, device=device))


def state_body(native_poi, current_poi):
    rotation, _, rmsd = _proper_rigid_fit(native_poi, current_poi)
    if bool((rmsd > 1e-3).any()):
        raise ValueError(f'POI state is not rigid: RMSD={rmsd.tolist()}')
    return rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-a', type=Path, required=True)
    parser.add_argument('--source-b', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=(5, 10, 15, 19))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.steps)) != len(args.steps) or any(not 0 <= s < 20 for s in args.steps):
        parser.error('steps must be unique integers in [0, 19]')
    configure_inference_reproducibility(True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    report_a, report_b = load_report(args.source_a), load_report(args.source_b)
    for key in ('cases', 'seed', 'steps'):
        if report_a[key] != report_b[key]:
            raise ValueError(f'unpaired report field: {key}')

    plan = ARTIFACTS / 'results/overnight_root_cause_v3_plan'
    entries = {entry['case']: entry for entry in json.loads(
        (plan / 'indexes/ternary_val_ccd.json').read_text())['entries']}
    expert_dir = ARTIFACTS / 'checkpoints/score_model'
    score_args = _load_score_model_args(str(expert_dir))
    batches = {case: load_batch(entries[case], plan, score_args)[0]
               for case in report_a['cases']}
    sources = {'a': args.source_a, 'b': args.source_b}
    checkpoints = {'a': Path(report_a['checkpoint']), 'b': Path(report_b['checkpoint'])}
    rows = []
    for model_name, checkpoint in checkpoints.items():
        trainer, _ = build(checkpoint, expert_dir)
        trainer.model.eval()
        context = nullcontext()
        with context, torch.no_grad():
            for source_name, folder in sources.items():
                for case, batch in batches.items():
                    native = {
                        'ligand': batch['ligand_data']['coords'],
                        'poi': batch['poi_data']['coords'],
                    }
                    for step in args.steps:
                        ligand, poi = load_state(folder, case, step, 'cuda')
                        body = state_body(native['poi'], poi)
                        time = ligand.new_tensor([1 - .05 * step])
                        dt = ligand.new_tensor([.05])
                        output = trainer._forward_state(
                            batch['e3_data'], batch['poi_data'], batch['ligand_data'],
                            ligand, poi, body, time, dt)
                        normalized = trainer.diffusion.normalized_predictions(output, time)
                        velocities = trainer.diffusion.velocities(
                            normalized, batch['e3_data']['coords'], batch['poi_data']['coords'])
                        following = trainer.diffusion.step(
                            ligand, poi, body, velocities, dt, batch['ligand_data'])
                        row = {'model': model_name, 'source': source_name,
                               'case': case, 'seed': report_a['seed'],
                               'step': step, 'time': float(time.item()), 'entities': {}}
                        for entity, before, after in (
                                ('ligand', ligand, following[0]),
                                ('poi', poi, following[1])):
                            metric = trajectory_rows(
                                [before[0].cpu().numpy(), after[0].cpu().numpy()],
                                native[entity][0].cpu().numpy())[0]
                            row['entities'][entity] = metric
                        rows.append(row)
        del trainer
        torch.cuda.empty_cache()
        print(f'checkpoint {model_name} complete', flush=True)
    output = {
        'scope': ('Raw checkpoints evaluated on identical saved states. Metrics describe '
                  'one finite solver step toward native, not raw-head finite differences.'),
        'source_a': str(args.source_a), 'source_b': str(args.source_b),
        'checkpoint_a': str(checkpoints['a']), 'checkpoint_b': str(checkpoints['b']),
        'seed': report_a['seed'], 'steps': args.steps, 'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
