#!/usr/bin/env python
"""Overfit fixed prior-replay states to joint endpoint translations."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat
from factorial_arm import PairedTorsionFlow
from local_paired_pilot import load_batches, sha256
from train_entry import _load_score_model_args
from tricoopdiff.checkpoint_contract import validate_expert_base_identity
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import ExperimentalConsensus
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.joint_endpoint_experiment import joint_rigid_endpoint_targets
from tricoopdiff.reproducibility import configure_inference_reproducibility
from tricoopdiff.trajectory_experiment import (
    PriorReplayJointTranslationTrainer,
    joint_endpoint_translation_loss,
    prior_replay_schedule,
)


def cosine(prediction, target):
    """Return per-batch cosine with finite zero-vector behavior."""
    denominator = (torch.linalg.vector_norm(prediction, dim=-1)
                   * torch.linalg.vector_norm(target, dim=-1)).clamp_min(1e-12)
    return ((prediction * target).sum(-1) / denominator).mean().item()


def fixed_state(trainer, batch, coupling, time):
    """Replay a deterministic model trajectory from the training prior."""
    current = time.new_ones(1)
    ligand, poi, body, _ = trainer.diffusion.path(
        batch['ligand_data'], batch['poi_data']['coords'], coupling, current)
    count, dt = prior_replay_schedule(
        time, inference_steps=trainer.diffusion.inference_steps, max_steps=15)
    trainer.model.eval()
    with torch.no_grad():
        for _ in range(count):
            output = trainer._forward_state(
                batch['e3_data'], batch['poi_data'], batch['ligand_data'],
                ligand, poi, body, current, dt)
            normalized = trainer.diffusion.normalized_predictions(output, current)
            velocities = trainer.diffusion.velocities(
                normalized, batch['e3_data']['coords'], batch['poi_data']['coords'])
            ligand, poi, body = trainer.diffusion.step(
                ligand, poi, body, velocities, dt, batch['ligand_data'])
            current = current - dt
        targets, diagnostic = joint_rigid_endpoint_targets(
            ligand, poi, body, batch['ligand_data']['coords'],
            batch['poi_data']['coords'], time, reject_near_pi=False,
            exact_fit_tolerance=None, body_tolerance=None)
    return ligand.detach(), poi.detach(), body.detach(), targets, diagnostic


def measure(trainer, batch, state, time):
    """Measure direct translation fit on one immutable state."""
    ligand, poi, body, targets, _ = state
    trainer.model.eval()
    output = trainer._forward_state(
        batch['e3_data'], batch['poi_data'], batch['ligand_data'],
        ligand, poi, body, time, time.new_full((1,), .05))
    normalized = trainer.diffusion.normalized_predictions(output, time)
    velocities = trainer.diffusion.velocities(
        normalized, batch['e3_data']['coords'], batch['poi_data']['coords'])
    scales = trainer.diffusion.scales(
        batch['e3_data']['coords'], batch['poi_data']['coords'])
    loss = joint_endpoint_translation_loss(velocities, targets, scales)
    result = {'loss': loss.item()}
    for branch in ('e3_ligand', 'poi_ligand'):
        prediction = velocities[branch]['tr']
        target = targets[branch]['tr']
        result[branch] = {
            'cosine': cosine(prediction, target),
            'prediction_norm': torch.linalg.vector_norm(prediction).item(),
            'target_norm': torch.linalg.vector_norm(target).item(),
            'error_norm': torch.linalg.vector_norm(prediction - target).item(),
        }
    return loss, result


def build_trainer(base, expert, device):
    """Build a fresh optimizer/model at the common baseline EMA."""
    fuse = ConsensusFuse(**base['model_config']).to(device)
    fuse.load_state_dict(base['ema_trifuse_state_dict'])
    model = ExperimentalConsensus(expert, fuse, base['expert_lora_config'])
    common = dict(
        device=device, experiment_contract={'training': 'fixed_translation_overfit_v1'},
        lr=5e-5, lora_lr=1e-5, lora_early_lr=5e-6, lora_warmup_steps=0,
        activation_clamp=base['runtime_config']['activation_clamp'],
        activation_clamp_mode=base['runtime_config']['activation_clamp_mode'],
        residual_weight=base['runtime_config']['residual_weight'],
        ema_decay=base['runtime_config']['ema_decay'])
    return PriorReplayJointTranslationTrainer(
        model, PairedTorsionFlow(), trajectory_steps=3,
        trajectory_weight=1., trajectory_min_time=.3,
        replay_max_steps=15, **common)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', default=(
        '3CXH_C_E_6PH', '4I0P_D_H_PGE', '4IL6_C_I_LMT'))
    ap.add_argument('--times', nargs='+', type=float, default=(.8, .5, .3))
    ap.add_argument('--updates', type=int, default=500)
    ap.add_argument('--seed', type=int, default=27001)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.updates < 1 or any(not .3 <= value <= 1 for value in args.times):
        raise ValueError('updates must be positive and times must lie in [0.3, 1]')
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    policy = configure_inference_reproducibility(True)
    device = 'cuda'
    score_args = _load_score_model_args(str(args.artifacts / 'checkpoints/score_model'))
    available = load_batches(args.artifacts, 'val', 8, device, score_args)
    by_case = {batch['case']: batch for batch in available}
    missing = set(args.cases) - set(by_case)
    if missing:
        raise ValueError(f'cases not in validation index: {sorted(missing)}')
    base = torch.load(args.base, map_location='cpu', weights_only=False)
    expert = build_diffdock_expert(str(args.artifacts / 'checkpoints/score_model'), device)
    validate_expert_base_identity(base, expert)
    report = {
        'status': 'running', 'contract': 'fixed_translation_overfit_v1',
        'base': str(args.base), 'base_sha256': sha256(args.base),
        'seed': args.seed, 'updates': args.updates, 'times': args.times,
        'cases': args.cases, 'reproducibility': policy, 'experiments': [],
    }
    for case_index, case in enumerate(args.cases):
        batch = copy.deepcopy(by_case[case])
        for time_value in args.times:
            run_seed = args.seed + case_index * 100 + round(time_value * 10)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            trainer = build_trainer(base, expert, device)
            coupling = trainer.diffusion.sample_coupling(
                batch['e3_data']['coords'], batch['poi_data']['coords'],
                batch['ligand_data'])
            time = torch.tensor([time_value], device=device)
            state = fixed_state(trainer, batch, coupling, time)
            trace = []
            for update in range(args.updates + 1):
                trainer.optimizer.zero_grad(set_to_none=True)
                loss, metrics = measure(trainer, batch, state, time)
                if update in {0, 1, 2, 5, 10, 25, 50, 100, 200, args.updates}:
                    trace.append({'update': update, **metrics})
                if update == args.updates:
                    break
                loss.backward()
                parameters = [p for group in trainer.optimizer.param_groups
                              for p in group['params'] if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError('non-finite fixed-state gradient')
                trainer.optimizer.step()
            row = {
                'case': case, 'time': time_value, 'run_seed': run_seed,
                'endpoint_diagnostic': {
                    key: value.cpu().tolist()
                    for key, value in state[-1].items()
                    if torch.is_tensor(value)
                },
                'trace': trace,
            }
            report['experiments'].append(row)
            args.out.joinpath('report.json').write_text(
                json.dumps(report, indent=2) + '\n')
            print(json.dumps({
                'case': case, 'time': time_value,
                'initial': trace[0], 'final': trace[-1],
            }), flush=True)
    report['status'] = 'complete'
    args.out.joinpath('report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
