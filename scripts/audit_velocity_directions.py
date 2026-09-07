"""Compare checkpoint velocities with exact SDPI path tangents on validation cases."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from train_entry import (TernaryDataLoader, _load_score_model_args,
                         build_trainer, parse_args, seed_training)
from tricoopdiff.reproducibility import configure_inference_reproducibility


def vector_metrics(predicted, target):
    """Return direction and scale diagnostics without hiding zero targets."""
    predicted = predicted.detach().double().reshape(-1)
    target = target.detach().double().reshape(-1)
    pred_norm = torch.linalg.vector_norm(predicted)
    target_norm = torch.linalg.vector_norm(target)
    dot = torch.dot(predicted, target)
    cosine = (float(dot / (pred_norm * target_norm))
              if pred_norm > 0 and target_norm > 0 else None)
    projection = float(dot / target_norm.square()) if target_norm > 0 else None
    error = torch.linalg.vector_norm(predicted - target)
    return {
        'prediction_norm': float(pred_norm), 'target_norm': float(target_norm),
        'cosine': cosine, 'signed_projection_ratio': projection,
        'error_norm': float(error),
    }


def coordinate_rms(actual, expected):
    return float((actual - expected).square().mean().sqrt())


def probe(trainer, batch, coupling, value, local_dt):
    """Evaluate one fixed path state, including an actual reverse-step check."""
    flow, model = trainer.diffusion, trainer.model
    time = torch.tensor([value], device=trainer.device)
    state, targets = flow.path(
        batch['ligand_data'], batch['poi_data']['coords'], coupling, time)
    dt = time.new_tensor([min(local_dt, value)])
    conditioning = flow.conditioning(
        time, dt, batch['e3_data']['coords'], batch['poi_data']['coords'])
    tr_sigma, _, _ = flow.t_to_sigma(time, time, time)
    crop = (tr_sigma * 3 + 20
            if bool(getattr(model.expert.model, 'dynamic_max_cross', False)) else None)
    output = model(
        batch['e3_data'], {**batch['poi_data'], 'coords': state['poi']},
        {**batch['ligand_data'], 'coords': state['a']}, t_e3l=time, t_pol=time,
        ligand_b_coords=state['b'], r_e3_to_poi=state['body'], diffusion_state=conditioning,
        receptor_crop_cutoff=crop)
    normalized = flow.normalized_predictions(output, time)
    predicted = flow.velocities(
        normalized, batch['e3_data']['coords'], batch['poi_data']['coords'])
    target_velocity = targets
    components = {}
    for branch in targets:
        for component in predicted[branch]:
            components[f'{branch}.{component}'] = vector_metrics(
                predicted[branch][component], targets[branch][component])

    expected, _ = flow.path(batch['ligand_data'], batch['poi_data']['coords'],
                         coupling, time - dt)
    predicted_step = flow.step(
        state, predicted, dt, batch['ligand_data'])
    oracle_step = flow.step(
        state, target_velocity, dt, batch['ligand_data'])
    step_errors = {}
    for name, index in (('ligand', 'a'), ('ligand_b', 'b'), ('poi', 'poi')):
        zero = coordinate_rms(state[index], expected[index])
        actual = coordinate_rms(predicted_step[index], expected[index])
        oracle = coordinate_rms(oracle_step[index], expected[index])
        step_errors[name] = {
            'zero_velocity_rms_A': zero,
            'predicted_velocity_rms_A': actual,
            'oracle_euler_rms_A': oracle,
            'improvement_over_zero_A': zero - actual,
        }
    return {'case': batch['case'], 'time': value, 'dt': float(dt),
            'components': components, 'local_step': step_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--paths', type=int, default=2)
    parser.add_argument('--seed', type=int, default=41000)
    parser.add_argument('--local-dt', type=float, default=.01)
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    args = parser.parse_args()
    if args.limit < 1 or args.paths < 1 or not 0 < args.local_dt <= .1:
        parser.error('limit/paths must be positive and local-dt in (0, .1]')
    index = json.loads(Path(args.index).read_text())
    if index.get('metadata', {}).get('split') != 'val':
        raise ValueError('velocity audit requires an explicitly labeled validation index')
    reproducibility = configure_inference_reproducibility(True)
    device = torch.device('cuda')
    seed_training(args.seed, device)
    config = parse_args([
        '--train_data', args.index, '--val_data', args.index,
        '--output_dir', args.out, '--expert_model_dir', args.expert_model_dir,
        '--device', 'cuda', '--lr', '1e-4'])
    loader = TernaryDataLoader(
        args.index, _load_score_model_args(args.expert_model_dir), shuffle=False,
        device='cuda', cache_dir=args.cache_dir)
    batches = [loader._load_one(entry) for entry in loader.entries[:args.limit]]
    model, flow, trainer = build_trainer(
        config, device, input_dim=batches[0]['e3_data']['features'].shape[-1])
    trainer.set_phase(1)
    trainer.load_checkpoint(args.checkpoint)
    model.eval()
    records = []
    with torch.no_grad():
        for path_index in range(args.paths):
            seed_training(args.seed + path_index, device)
            for batch in batches:
                coupling = flow.sample_coupling(
                    batch['e3_data']['coords'], batch['poi_data']['coords'],
                    batch['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
                for value in (.1, .5, .9):
                    row = probe(trainer, copy.deepcopy(batch), copy.deepcopy(coupling),
                                value, args.local_dt)
                    row['path_index'] = path_index
                    records.append(row)
                    print(row['case'], path_index, value, flush=True)
    summary = {}
    names = list(records[0]['components'])
    for name in names:
        values = [row['components'][name] for row in records]
        cosines = [v['cosine'] for v in values if v['cosine'] is not None]
        projections = [v['signed_projection_ratio'] for v in values
                       if v['signed_projection_ratio'] is not None]
        summary[name] = {
            'mean_cosine': float(np.mean(cosines)) if cosines else None,
            'fraction_positive_cosine': (float(np.mean(np.asarray(cosines) > 0))
                                         if cosines else None),
            'median_projection_ratio': (float(np.median(projections))
                                        if projections else None),
            'n_directional': len(cosines),
        }
    for molecule in ('ligand', 'ligand_b', 'poi'):
        improvements = [row['local_step'][molecule]['improvement_over_zero_A'] for row in records]
        summary[f'{molecule}_local_step'] = {
            'mean_improvement_over_zero_A': float(np.mean(improvements)),
            'fraction_improved': float(np.mean(np.asarray(improvements) > 0)),
        }
    payload = {'scope': 'fixed validation path tangents; not a benchmark',
               'args': vars(args), 'reproducibility': reproducibility,
               'checkpoint_epoch': trainer.resume_epoch,
               'checkpoint_global_step': trainer.global_step,
               'cases': [batch['case'] for batch in batches],
               'summary': summary, 'records': records}
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'summary.json').write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
