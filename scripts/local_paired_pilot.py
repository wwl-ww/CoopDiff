#!/usr/bin/env python
"""Deterministic short pilots from one baseline EMA on transferred v7 caches."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), '/home/wl/workspace/CoopDiff']
import tricoopdiff.diffdock_compat
from factorial_arm import PairedTorsionFlow, restrict_non_single
from featurize_common import _to_device_and_batch
from train_entry import _load_score_model_args
from tricoopdiff.checkpoint_contract import validate_expert_base_identity
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import (
    EndpointTranslationRolloutTrainer, ExperimentalConsensus,
    RolloutAugmentedTrainer)
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.reproducibility import configure_inference_reproducibility


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_batches(artifacts, split, limit, device, score_args, exclude_pdb=()):
    index = artifacts / f'results/overnight_root_cause_v3_plan/indexes/ternary_{split}_ccd.json'
    entries = [entry for entry in json.loads(index.read_text())['entries']
               if entry['case'].split('_', 1)[0] not in set(exclude_pdb)][:limit]
    if len(entries) != limit:
        raise ValueError(f'{split}: requested {limit} cases after PDB exclusion, found {len(entries)}')
    result = []
    for entry in entries:
        path = artifacts / f"results/overnight_root_cause_v3_plan/cache_ccd/{entry['case']}.pt"
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['cache_version'] != 7:
            raise ValueError(f"{entry['case']}: expected cache v7")
        digest = hashlib.sha256()
        for key in ('protein1', 'protein2', 'ligand'):
            digest.update(entry[key].encode())
            digest.update((artifacts / entry[key].removeprefix('/tmp/coopdiff/')).read_bytes())
        digest.update(json.dumps(vars(score_args), sort_keys=True, default=str).encode())
        if digest.hexdigest() != saved['source_signature']:
            raise ValueError(f"{entry['case']}: source signature mismatch")
        batch = {key: _to_device_and_batch(saved[key], device)
                 for key in ('e3_data', 'poi_data', 'ligand_data')}
        ligand = artifacts / entry['ligand'].removeprefix('/tmp/coopdiff/')
        result.append(restrict_non_single({'case': entry['case'], **batch}, str(ligand)))
    return result


def make_trainer(variant, model, flow, common, *, trajectory_steps=3,
                 trajectory_weight=.1):
    if variant == 'teacher':
        return RolloutAugmentedTrainer(model, flow, rollout_probability=0, rollout_steps=5, **common)
    if variant == 'coupling':
        return RolloutAugmentedTrainer(model, flow, rollout_probability=.5, rollout_steps=5, **common)
    if variant == 'e3_endpoint':
        return EndpointTranslationRolloutTrainer(
            model, flow, rollout_probability=.5, rollout_steps=5,
            rollout_min_time=.25, endpoint_blend=.25, endpoint_correction_ratio=1., **common)
    if variant == 'transported':
        from tricoopdiff.offpath_experiments import BodyFrameTransportRolloutTrainer
        return BodyFrameTransportRolloutTrainer(
            model, flow, rollout_probability=.5, rollout_steps=5,
            rollout_min_time=.25, **common)
    if variant == 'gauge_transported':
        from tricoopdiff.offpath_experiments import GaugeCorrectBodyFrameTransportRolloutTrainer
        return GaugeCorrectBodyFrameTransportRolloutTrainer(
            model, flow, rollout_probability=.5, rollout_steps=5,
            rollout_min_time=.25, **common)
    if variant == 'joint_endpoint':
        from tricoopdiff.offpath_experiments import BoundedJointEndpointRolloutTrainer
        return BoundedJointEndpointRolloutTrainer(
            model, flow, rollout_probability=.5, rollout_steps=5,
            rollout_min_time=.25, endpoint_blend=.25,
            endpoint_translation_cap=10., endpoint_rotation_cap=.5,
            **common)
    if variant == 'trajectory':
        from tricoopdiff.trajectory_experiment import TrajectoryConsistencyTrainer
        return TrajectoryConsistencyTrainer(
            model, flow, trajectory_steps=trajectory_steps,
            trajectory_weight=trajectory_weight,
            trajectory_min_time=.25, **common)
    if variant in ('poi_stage', 'poi_stage_distance'):
        from tricoopdiff.trajectory_experiment import (
            DISTANCE_STAGED_POI_CONTRACT_MODE,
            STAGED_POI_CONTRACT_MODE,
            StagedPOITrajectoryTrainer,
        )
        contract_mode = (DISTANCE_STAGED_POI_CONTRACT_MODE
                         if variant == 'poi_stage_distance'
                         else STAGED_POI_CONTRACT_MODE)
        return StagedPOITrajectoryTrainer(
            model, flow, trajectory_steps=trajectory_steps,
            trajectory_weight=trajectory_weight,
            trajectory_min_time=.25, trajectory_max_time=.65,
            distance_weight_cap=3., contract_mode=contract_mode, **common)
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', choices=('teacher', 'coupling', 'e3_endpoint',
                                           'transported', 'gauge_transported',
                                           'joint_endpoint', 'trajectory',
                                           'poi_stage', 'poi_stage_distance'), required=True)
    ap.add_argument('--train-cases', type=int, default=32)
    ap.add_argument('--val-cases', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=22001)
    ap.add_argument('--trajectory-steps', type=int, default=3)
    ap.add_argument('--trajectory-weight', type=float, default=.1)
    ap.add_argument('--save-every-epoch', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    policy = configure_inference_reproducibility(True)
    device = 'cuda'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    score_args = _load_score_model_args(str(args.artifacts / 'checkpoints/score_model'))
    val = load_batches(args.artifacts, 'val', args.val_cases, device, score_args)
    val_pdb = {batch['case'].split('_', 1)[0] for batch in val}
    train = load_batches(args.artifacts, 'train', args.train_cases, device, score_args,
                         exclude_pdb=val_pdb)
    base = torch.load(args.base, map_location='cpu', weights_only=False)
    expert = build_diffdock_expert(str(args.artifacts / 'checkpoints/score_model'), device)
    validate_expert_base_identity(base, expert)
    fuse = ConsensusFuse(**base['model_config']).to(device)
    fuse.load_state_dict(base['ema_trifuse_state_dict'])
    model = ExperimentalConsensus(expert, fuse, base['expert_lora_config'])
    contract = {'chemistry': 'ccd', 'torsion': 'single_bond', 'adaptation': 'frozen',
                'training': 'deterministic_paired_pilot_v2', 'variant': args.variant,
                'rng_pairing': 'per_example_reset_and_supervised_forward_restore_v1',
                'split_policy': 'train_excludes_validation_pdb_prefix_v1',
                'seed': args.seed, 'base_checkpoint_sha256': sha256(args.base),
                'train_cases': [b['case'] for b in train], 'val_cases': [b['case'] for b in val]}
    common = dict(device=device, experiment_contract=contract, lr=5e-5,
                  lora_lr=1e-5, lora_early_lr=5e-6, lora_warmup_steps=0,
                  activation_clamp=base['runtime_config']['activation_clamp'],
                  activation_clamp_mode=base['runtime_config']['activation_clamp_mode'],
                  residual_weight=base['runtime_config']['residual_weight'],
                  ema_decay=base['runtime_config']['ema_decay'])
    trainer = make_trainer(
        args.variant, model, PairedTorsionFlow(), common,
        trajectory_steps=args.trajectory_steps,
        trajectory_weight=args.trajectory_weight)
    original_training_state = trainer._training_state
    def paired_training_state(*state_args, **state_kwargs):
        # Rollout arms consume extra random numbers before the supervised
        # forward. Restore the entry state so that its dropout mask is paired
        # with the teacher control for the same example.
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        state = original_training_state(*state_args, **state_kwargs)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        return state
    trainer._training_state = paired_training_state
    # Fresh optimizer, but both raw and EMA start at the exact same baseline EMA.
    initial = {k: v.detach().clone() for k, v in model.trifuse.state_dict().items()}
    rng = np.random.default_rng(args.seed)
    fixed_validation = []
    numpy_state = np.random.get_state()
    for i, batch in enumerate(val):
        seed = args.seed + 10000 + i
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            np.random.seed(seed); torch.manual_seed(seed)
            coupling = trainer.diffusion.sample_coupling(
                batch['e3_data']['coords'], batch['poi_data']['coords'], batch['ligand_data'])
        fixed_validation.append((batch, coupling, torch.tensor([(.25, .5, .75)[i % 3]], device=device)))
    np.random.set_state(numpy_state)
    history, best, best_ema = [], float('inf'), float('inf')
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        losses = []
        trajectory_losses = []
        for position, index in enumerate(order):
            # Reset every mutable RNG at the example boundary. Variants may
            # consume different random counts internally, but the next example
            # still receives the same coupling, time, and initial dropout stream.
            step_seed = args.seed * 1_000_000 + epoch * len(train) + position
            np.random.seed(step_seed % (2 ** 32))
            torch.manual_seed(step_seed)
            torch.cuda.manual_seed_all(step_seed)
            step_result = trainer.train_step(**copy.deepcopy(train[int(index)]))
            losses.append(step_result['loss_total'])
            if 'loss_trajectory' in step_result:
                trajectory_losses.append(step_result['loss_trajectory'])
        raw_values, ema_values = [], []
        for use_ema, values in ((False, raw_values), (True, ema_values)):
            context = trainer.ema_parameters() if use_ema else torch.no_grad()
            with context, torch.no_grad():
                model.eval()
                old_rollout = getattr(trainer, 'rollout_probability', None)
                old_trajectory_weight = getattr(trainer, 'trajectory_weight', None)
                if old_rollout is not None:
                    trainer.rollout_probability = 0
                if old_trajectory_weight is not None:
                    trainer.trajectory_weight = 0
                try:
                    for batch, coupling, time in fixed_validation:
                        values.append(float(trainer.compute_loss(
                            **copy.deepcopy(batch), coupling_override=coupling,
                            t_override=time)['loss_total']))
                finally:
                    if old_rollout is not None:
                        trainer.rollout_probability = old_rollout
                    if old_trajectory_weight is not None:
                        trainer.trajectory_weight = old_trajectory_weight
        row = {'epoch': epoch, 'mean_train_loss': float(np.mean(losses)),
               'fixed_raw_val_loss': float(np.mean(raw_values)),
               'fixed_ema_val_loss': float(np.mean(ema_values)),
               'rollout_augmented_states': getattr(
                   trainer, 'rollout_augmented_states', 0),
               'trajectory_augmented_states': getattr(
                   trainer, 'trajectory_augmented_states', 0)}
        if trajectory_losses:
            row['mean_trajectory_loss'] = float(np.mean(trajectory_losses))
        history.append(row)
        if row['fixed_raw_val_loss'] < best:
            best = row['fixed_raw_val_loss']
            trainer.save_checkpoint(args.out / 'best_raw_selected.pt', epoch, best)
        if row['fixed_ema_val_loss'] < best_ema:
            best_ema = row['fixed_ema_val_loss']
            trainer.save_checkpoint(args.out / 'best_ema_selected.pt', epoch, best_ema)
        trainer.save_checkpoint(args.out / 'latest.pt', epoch, best)
        if args.save_every_epoch:
            trainer.save_checkpoint(args.out / f'epoch_{epoch:03d}.pt', epoch, best)
        (args.out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    changed = {k: float((v - initial[k]).norm()) for k, v in model.trifuse.state_dict().items()
               if v.is_floating_point()}
    report = {'status': 'complete', 'contract': contract, 'numerical_policy': policy,
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'history': history, 'best_raw_val_loss': best,
              'best_ema_val_loss': best_ema,
              'changed_tensors': sum(x > 0 for x in changed.values()),
              'parameter_deltas': changed}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
