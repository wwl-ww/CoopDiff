#!/usr/bin/env python
"""Real-case SDPI flow tangent gates and optional local GPU integration smoke."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validate_initialization import read_case
from tricoopdiff.sdpi_flow import SDPIFlow, FLOW_VERSION


def check_geometry(directory, seeds):
    a, b, data = read_case(directory)
    data['coords'] = torch.tensor(Chem.Mol(data['mol']).GetConformer().GetPositions(), dtype=a.dtype)[None]
    a, b = a[None], b[None]
    flow = SDPIFlow()
    records = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        coupling = flow.sample_coupling(a, b, data)
        maximum = 0.
        for value in (0., .1, .5, .9, 1.):
            t = torch.tensor([value], dtype=a.dtype)
            ligand, poi, body, velocity = flow.path(data, b, coupling, t)
            assert torch.isfinite(ligand).all() and torch.isfinite(poi).all()
            if value == 0:
                assert torch.allclose(ligand, data['coords'], atol=1e-9)
                assert torch.allclose(poi, b, atol=1e-9)
                continue
            errors = []
            for h in (1e-4, 5e-5):
                dt = t.new_tensor([h])
                actual = flow.step(ligand, poi, body, velocity, dt, data)
                expected = flow.path(data, b, coupling, t - dt)
                errors.append(max(float((actual[i] - expected[i]).abs().max()) for i in (0, 1)))
            assert errors[1] < errors[0] * .4 + 1e-8, (directory.name, seed, value, errors)
            assert errors[1] < 1e-3, (directory.name, seed, value, errors)
            maximum = max(maximum, errors[1])
        records.append({'seed': seed, 'max_half_step_cartesian_error_angstrom': maximum})
        print(f'[flow geometry] {directory.name} seed={seed} PASS error={maximum:.3g}', flush=True)
    return {'case': directory.name, 'records': records}


def gpu_smoke(directories, output):
    """Three real updates, LoRA warmup transition, checkpoint restore, sampling.

    Benchmark examples here are ONLY diagnostic. Never use this checkpoint
    for a benchmark accuracy claim or as a formal training initialization.
    """
    from train_entry import build_trainer, _load_score_model_args, seed_training
    from featurize_common import featurize_complex
    from tricoopdiff.initialization import TernaryInitializer
    from tricoopdiff.sampler import sample_single
    from evaluation_common import validate_checkpoint_payload

    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires a dedicated CUDA device; use SLURM on a cluster')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['NAN_CLAMP'] = '100'
    args = SimpleNamespace(
        expert_model_dir=str(ROOT / 'checkpoints/score_model'), lora_rank=4, lora_alpha=8.,
        lora_start_layer=0, lora_early_layers=4, lora_early_rank=2, lora_early_alpha=4.,
        node_dim=128, ns=32, nv=4, num_transformer_layers=3, lr=1e-4, weight_decay=1e-4,
        lora_lr=1e-5, lora_early_lr=5e-6, nan_clamp=100., torsion_expert_aux_weight=.25,
        lora_warmup_steps=1,
    )
    score_args = _load_score_model_args(args.expert_model_dir)
    batches = []
    for directory in directories:
        directory = directory.resolve()
        e3, poi, ligand, dim = featurize_complex(
            directory.name, str(directory / 'protein1.pdb'), str(directory / 'protein2.pdb'),
            str(directory / 'ligand.sdf'), score_args, 'cuda',
            cache_dir=str(ROOT / 'results/featurize_cache_dual_native_v5'),
        )
        batches.append(dict(e3_data=e3, poi_data=poi, ligand_data=ligand, case=directory.name))
    seed_training(0, torch.device('cuda'))
    model, flow, trainer = build_trainer(args, torch.device('cuda'), input_dim=dim)
    trainer.set_phase(1)
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
              if name.startswith('trifuse.') or 'lora_' in name}
    records = []
    for i, batch in enumerate(batches):
        result = trainer.train_step(**batch)
        assert np.isfinite(result['loss_total'])
        records.append({'case': batch['case'], **result})
        print(f'[flow train] step={i} {batch["case"]} loss={result["loss_total"]:.6g}', flush=True)
    changed = [name for name, p in model.named_parameters()
               if name in before and not torch.equal(before[name], p.detach().cpu())]
    for pattern in ('film', 'bridge', 'torsion_coordinator', 'lora_'):
        assert any(pattern in name for name in changed), f'no {pattern} parameter update'
    validation = trainer.validate_step(**batches[0])
    assert not model.training and np.isfinite(validation['loss_total'])
    checkpoint = output / 'diagnostic_only.pt'
    trainer.save_checkpoint(str(checkpoint), epoch=1, best_val_loss=validation['loss_total'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint_payload(payload)
    trainer.load_checkpoint(str(checkpoint))
    assert trainer.global_step == len(batches)
    sampled = []
    for batch in batches:
        result = sample_single(model, flow, TernaryInitializer(),
                               batch['e3_data'], batch['poi_data'], batch['ligand_data'],
                               seed=0, steps=20, return_trace=True)
        assert result['valid'], result['termination_reason']
        assert np.isfinite(result['ligand_coords']).all() and np.isfinite(result['poi_coords']).all()
        assert len([item for item in result['trace'] if item['stage'] == 'step']) == 20
        sampled.append({'case': batch['case'], 'valid': result['valid'], 'trace': result['trace']})
        print(f'[flow sample] {batch["case"]}: 20/20 finite', flush=True)
    return {'diagnostic_only': True, 'training': records, 'validation': validation,
            'changed_parameter_names': changed, 'checkpoint': str(checkpoint), 'sampling': sampled}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dirs', nargs='+', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 7, 42])
    parser.add_argument('--gpu-smoke-dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {'generative_process': FLOW_VERSION,
              'geometry': [check_geometry(d, args.seeds) for d in args.case_dirs]}
    if args.gpu_smoke_dir:
        report['gpu_smoke'] = gpu_smoke(args.case_dirs, args.gpu_smoke_dir)
    report['passed'] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'PASS: {args.out}', flush=True)


if __name__ == '__main__':
    main()
