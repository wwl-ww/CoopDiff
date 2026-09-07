"""Bounded single-state v2 fitting; no benchmark data or overfit-weight reuse."""

import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from tricoopdiff.consensus import ConsensusCoopDiff, ConsensusFuse
from tricoopdiff.consensus_train import ConsensusTrainer
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.reproducibility import configure_inference_reproducibility
from tricoopdiff.sdpi_flow import SDPIFlow
from train_entry import TernaryDataLoader, _load_score_model_args


def build(expert_dir, input_dim, dropout=0.1, lr=1e-4):
    torch.set_num_threads(4)
    configure_inference_reproducibility()
    torch.manual_seed(17)
    np.random.seed(17)
    expert = build_diffdock_expert(expert_dir, 'cuda')
    model = ConsensusCoopDiff(expert, ConsensusFuse(input_dim=input_dim, dropout=dropout))
    return ConsensusTrainer(model, SDPIFlow(), device='cuda', lr=lr)


def probe(trainer, batch):
    trainer.model.eval()
    with torch.no_grad():
        result = trainer.compute_loss(**copy.deepcopy(batch), diagnostics=True)
    metrics = {k: v.item() for k, v in result['components'].items()}
    metrics['loss_total'] = result['loss_total'].item()
    for prefix in ('pol_rot', 'e3l_rot'):
        p = result['rms'][prefix + '_prediction_rms']
        t = result['rms'][prefix + '_target_rms']
        metrics[prefix + '_cosine'] = float(np.clip(
            (p * p + t * t - metrics['loss_' + prefix]) / max(2 * p * t, 1e-12), -1, 1))
    if not all(np.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('non-finite validation probe')
    return metrics


def overfit(config, output, seconds):
    configure_inference_reproducibility()
    payload = torch.load(config['fixed_pair'], map_location='cuda', weights_only=False)
    batch = {**payload['batch'], 'coupling_override': payload['coupling'],
             't_override': torch.tensor([payload['time']], device='cuda')}
    trainer = build(config['expert_dir'], batch['e3_data']['features'].shape[-1],
                    dropout=0, lr=1e-3)
    initial = probe(trainer, batch)
    start = time.monotonic()
    history = []
    with (output / 'steps.jsonl').open('x') as handle:
        for step in range(200):
            if time.monotonic() - start > seconds - 30:
                raise TimeoutError('overfit did not finish within its budget')
            metrics = trainer.train_step(**copy.deepcopy(batch))
            handle.write(json.dumps({'step': step + 1, **metrics}, allow_nan=False) + '\n')
            handle.flush()
            if (step + 1) % 20 == 0:
                current = probe(trainer, batch)
                history.append({'step': step + 1, **current})
                print('overfit', step + 1, current, flush=True)
    final = probe(trainer, batch)
    # This checkpoint is diagnostic-only and never passed to formal training.
    trainer.save_checkpoint(output / 'overfit_only.pt')
    checks = {'loss_decreased': final['loss_total'] <= 0.25 * initial['loss_total'],
              'poi_rotation_learned': final['pol_rot_cosine'] >= 0.9,
              'finite_gradients': all(p.grad is None or torch.isfinite(p.grad).all().item()
                                      for p in trainer.model.parameters())}
    return {'checks': checks, 'initial': initial, 'final': final, 'history': history,
            'steps': trainer.global_step, 'elapsed_seconds': time.monotonic() - start,
            'scope': 'one training case, one time/path; not a generalization result'}


def train(config, output, seconds, pilot=False):
    configure_inference_reproducibility()
    args = _load_score_model_args(config['expert_dir'])
    torch.manual_seed(17)
    np.random.seed(17)
    train_loader = TernaryDataLoader(
        config['train_index'], args, shuffle=True, device='cuda',
        cache_dir=config['train_cache'], random_flip_proteins=True,
        max_clusters=32 if pilot else None)
    val_loader = TernaryDataLoader(
        config['val_index'], args, shuffle=False, device='cuda',
        cache_dir=config['train_cache'], max_clusters=4 if pilot else 8)
    first = train_loader._load_one(train_loader.entries[0])
    trainer = build(config['expert_dir'], first['e3_data']['features'].shape[-1])
    # Fixed held-out paths prevent checkpoint selection by favorable fresh noise.
    probes = []
    for batch in val_loader:
        coupling = trainer.diffusion.sample_coupling(
            batch['e3_data']['coords'], batch['poi_data']['coords'], batch['ligand_data'],
            no_torsion=bool(trainer.model.expert.model.no_torsion))
        for value in (.1, .5, .9):
            probes.append({**batch, 'coupling_override': coupling,
                           't_override': torch.tensor([value], device='cuda')})
    if not probes:
        raise ValueError('validation set is empty')
    torch.save(probes, output / 'validation_probes.pt')
    best = float('inf')
    validations = []

    def validate(epoch):
        nonlocal best
        with trainer.ema_parameters():
            records = [probe(trainer, sample) for sample in probes]
        mean = {key: float(np.mean([r[key] for r in records])) for key in records[0]}
        validations.append({'step': trainer.global_step, **mean})
        if mean['loss_total'] < best:
            best = mean['loss_total']
            if trainer.global_step:
                trainer.save_checkpoint(output / 'best.pt', epoch=epoch, best_val_loss=best)
        trainer.save_checkpoint(output / 'latest.pt', epoch=epoch, best_val_loss=best)
        (output / 'validation.json').write_text(json.dumps(validations, indent=2, allow_nan=False))
        print('validation', trainer.global_step, mean, flush=True)

    start = time.monotonic()
    validate(0)
    # Reserve time for a final validation/save; the parent enforces the hard cap.
    reserve = max(60, 2 * (time.monotonic() - start))
    deadline = start + seconds - reserve
    completed_epochs = 0
    with (output / 'steps.jsonl').open('x') as handle:
        while time.monotonic() < deadline:
            finished_epoch = True
            count = 0
            for batch in train_loader:
                if time.monotonic() >= deadline or (pilot and trainer.global_step >= 64):
                    finished_epoch = False
                    break
                metrics = trainer.train_step(**batch)
                count += 1
                handle.write(json.dumps({'step': trainer.global_step, 'case': batch['case'],
                                         **metrics}, allow_nan=False) + '\n')
                handle.flush()
                if trainer.global_step % 10 == 0:
                    print('train', trainer.global_step, metrics['loss_total'], flush=True)
                if trainer.global_step % 50 == 0:
                    trainer.save_checkpoint(output / 'latest.pt', epoch=completed_epochs,
                                            best_val_loss=best)
                if trainer.global_step % 100 == 0:
                    validate(completed_epochs)
            if finished_epoch:
                completed_epochs += 1
            if pilot and trainer.global_step >= 64:
                break
            if not count:
                break
    validate(completed_epochs)
    if trainer.global_step < (8 if pilot else 1):
        raise RuntimeError('too few steps completed; no successful training claim')
    candidate = output / 'best.pt'
    if not candidate.exists():
        candidate = output / 'latest.pt'
    checks = ({'heldout_validation': True, 'throughput_measured': True,
               'checkpoint_saved': candidate.is_file()} if pilot else
              {'finite_training': True, 'checkpoint_saved': candidate.is_file(),
               'validation_recorded': bool(validations)})
    return {'checks': checks, 'checkpoint': str(candidate.resolve()),
            'steps': trainer.global_step, 'completed_epochs': completed_epochs,
            'elapsed_seconds': time.monotonic() - start,
            'steps_per_second': trainer.global_step / (time.monotonic() - start),
            'validation': validations, 'validation_cases': sorted({p['case'] for p in probes}),
            'validation_scope': 'fixed subset of held-out clusters; not full validation',
            'initialization': 'fresh model; no pilot or overfit weights reused',
            'resume_scope': 'weights/optimizer saved; exact mid-epoch data replay not implemented'}
