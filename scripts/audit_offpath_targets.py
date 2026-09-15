#!/usr/bin/env python
"""Compare original coupling targets with direct endpoint targets off path."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
import tricoopdiff.diffdock_compat  # noqa: F401,E402
from audit_velocity_directions import vector_metrics  # noqa: E402
from evaluate_rollout_pilot import build  # noqa: E402
from factorial_arm import restrict_non_single  # noqa: E402
from train_entry import TernaryDataLoader, _load_score_model_args  # noqa: E402


CASES = {
    '3CXH_C_E_6PH', '6U9D_Q_R_TPP', '4I0P_D_H_PGE', '5W7C_A_C_FTT',
    '6HS4_A_B_GON', '4IL6_C_I_LMT', '7L0F_G_H_GSP', '3ZCB_A_B_ATP',
}


def endpoint_e3_target(ligand, native, time):
    """Rigid velocity that maps the current ligand best-fit pose to native at t=0."""
    current = ligand[0].detach().cpu().double().numpy()
    target = native[0].detach().cpu().double().numpy()
    current_center = current.mean(0)
    target_center = target.mean(0)
    rotation, _ = Rotation.align_vectors(
        target - target_center, current - current_center)
    value = float(time.item())
    return {
        'tr': ligand.new_tensor((current_center - target_center) / value)[None],
        'rot': ligand.new_tensor(-rotation.as_rotvec() / value)[None],
    }


def model_velocity(trainer, batch, ligand, poi, body, time, dt):
    output = trainer._forward_state(
        batch['e3_data'], batch['poi_data'], batch['ligand_data'],
        ligand, poi, body, time, dt)
    normalized = trainer.diffusion.normalized_predictions(output, time)
    return trainer.diffusion.velocities(
        normalized, batch['e3_data']['coords'], batch['poi_data']['coords'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--paths', type=int, default=2)
    parser.add_argument('--seed', type=int, default=13001)
    args = parser.parse_args()

    index_dir = Path('/tmp/coopdiff/results/overnight_root_cause_v3_plan/indexes')
    index_path = index_dir / 'ternary_val_ccd.json'
    entries = json.loads(index_path.read_text())['entries']
    ligand_paths = {entry['case']: entry['ligand'] for entry in entries}
    loader = TernaryDataLoader(
        index_path, _load_score_model_args('/tmp/coopdiff/checkpoints/score_model'),
        shuffle=False, device='cuda',
        cache_dir='/tmp/coopdiff/results/overnight_root_cause_v3_plan/cache_ccd')
    batches = []
    for batch in loader:
        if batch['case'] in CASES:
            batches.append(restrict_non_single(batch, ligand_paths[batch['case']]))
        if len(batches) == args.cases:
            break

    model, trainer = build(args.checkpoint)
    flow = trainer.diffusion
    rows = []
    with trainer.ema_parameters(), torch.no_grad():
        model.eval()
        for path_index in range(args.paths):
            np.random.seed(args.seed + path_index)
            torch.manual_seed(args.seed + path_index)
            for batch in batches:
                coupling = flow.sample_coupling(
                    batch['e3_data']['coords'], batch['poi_data']['coords'],
                    batch['ligand_data'],
                    no_torsion=bool(model.expert.model.no_torsion))
                time = torch.tensor([1.0], device='cuda')
                dt = time.new_tensor([0.05])
                ligand, poi, body, _ = flow.path(
                    batch['ligand_data'], batch['poi_data']['coords'], coupling, time)
                for step in range(1, 16):
                    velocity = model_velocity(
                        trainer, batch, ligand, poi, body, time, dt)
                    ligand, poi, body = flow.step(
                        ligand, poi, body, velocity, dt, batch['ligand_data'])
                    time = time - dt
                    if step not in (5, 10, 15):
                        continue
                    _, _, _, original = flow.path(
                        batch['ligand_data'], batch['poi_data']['coords'], coupling, time)
                    teacher_ligand, _, _, _ = flow.path(
                        batch['ligand_data'], batch['poi_data']['coords'], coupling, time)
                    endpoint = endpoint_e3_target(
                        ligand, batch['ligand_data']['coords'], time)
                    teacher_endpoint = endpoint_e3_target(
                        teacher_ligand, batch['ligand_data']['coords'], time)
                    predicted = model_velocity(
                        trainer, batch, ligand, poi, body, time, dt)['e3_ligand']
                    rows.append({
                        'case': batch['case'], 'path_index': path_index,
                        'time': float(time.item()),
                        'offpath_center_error_A': float(torch.linalg.vector_norm(
                            ligand.mean(1) - batch['ligand_data']['coords'].mean(1))),
                        'original_vs_endpoint': {
                            component: vector_metrics(
                                original['e3_ligand'][component], endpoint[component])
                            for component in ('tr', 'rot')},
                        'teacher_original_vs_endpoint': {
                            component: vector_metrics(
                                original['e3_ligand'][component],
                                teacher_endpoint[component])
                            for component in ('tr', 'rot')},
                        'prediction_vs_original': {
                            component: vector_metrics(
                                predicted[component], original['e3_ligand'][component])
                            for component in ('tr', 'rot')},
                        'prediction_vs_endpoint': {
                            component: vector_metrics(
                                predicted[component], endpoint[component])
                            for component in ('tr', 'rot')},
                    })
                print(batch['case'], path_index, flush=True)

    summary = {}
    for comparison in ('teacher_original_vs_endpoint', 'original_vs_endpoint',
                       'prediction_vs_original', 'prediction_vs_endpoint'):
        summary[comparison] = {}
        for component in ('tr', 'rot'):
            values = [row[comparison][component] for row in rows]
            summary[comparison][component] = {
                'mean_cosine': float(np.mean([value['cosine'] for value in values])),
                'median_projection_ratio': float(np.median([
                    value['signed_projection_ratio'] for value in values])),
                'mean_error_norm': float(np.mean([
                    value['error_norm'] for value in values])),
            }
    payload = {
        'scope': 'E3-ligand rigid target audit on model-generated off-path states',
        'checkpoint': str(args.checkpoint.resolve()),
        'cases': [batch['case'] for batch in batches],
        'summary': summary,
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
