"""Offline cached-case integration/overfit gate (N=1 or N=32/64).

Uses explicit local featurization caches as diagnostic fixtures, not benchmark
metrics. Run on an allocated GPU or a workstation; never a cluster login node.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import tricoopdiff.diffdock_compat  # noqa: E402,F401
import torch

from featurize_common import _to_device_and_batch
from train_entry import build_trainer, parse_args, seed_training
from tricoopdiff.dual_consensus import FLOW_VERSION
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.sampler import sample_single


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--cases', nargs='+', default=['5T35_D_A_759'])
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--sampling-steps', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    args = parser.parse_args()
    if args.steps < 1 or args.sampling_steps < 1:
        parser.error('steps must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    seed_training(42, device)
    batches = []
    for case in args.cases:
        cache = torch.load(Path(args.cache_dir) / f'{case}.pt', map_location='cpu', weights_only=False)
        if cache.get('cache_version') != 5:
            raise ValueError(f'{case}: require current v5 cache')
        batch = {k: _to_device_and_batch(cache[k], device) for k in ('e3_data', 'poi_data', 'ligand_data')}
        batch['case'] = case
        batches.append(batch)
    config = parse_args(['--train_data', 'unused', '--val_data', 'unused', '--output_dir', str(out),
                         '--lora-warmup-steps', '0', '--expert_model_dir', args.expert_model_dir,
                         '--device', args.device])
    model, flow, trainer = build_trainer(config, device, input_dim=batches[0]['e3_data']['features'].shape[-1])
    trainer.set_phase(1)
    trainer._set_lora_trainable(True)
    pairs = [(batch, flow.sample_coupling(batch['e3_data']['coords'], batch['poi_data']['coords'],
                                         batch['ligand_data'], model.expert.model.no_torsion)) for batch in batches]
    t = torch.tensor([.1], device=device)
    # At zero initialization, the full coordinated forward must preserve both
    # native binary fields for their respective (different) geometries.
    with torch.no_grad():
        batch, coupling = pairs[0]
        state, _ = flow.path(batch['ligand_data'], batch['poi_data']['coords'], coupling, t)
        def binary_forward(stage):
            model.dual_stage = stage
            model.eval()
            return model(batch['e3_data'], {**batch['poi_data'], 'coords': state['poi']},
                         {**batch['ligand_data'], 'coords': state['a']}, t_e3l=t,
                         ligand_b_coords=state['b'], r_e3_to_poi=state['body'])['interface_scores']
        native = binary_forward('independent')
        coordinated = binary_forward('annealed')
        parity = max(float((native[n][k] - coordinated[n][k]).abs().max())
                     for n in native for k in native[n] if native[n][k].numel())
        for name in native:
            for key in native[name]:
                assert torch.allclose(native[name][key], coordinated[name][key], atol=2e-4, rtol=2e-4), (name, key, parity)
    def measure(batch, coupling):
        model.eval()
        return trainer.compute_loss(**batch, coupling_override=coupling, t_override=t, diagnostics=True)
    def record(result):
        return {'loss': float(result['loss_total'].detach()),
                'components': {k: float(v.detach()) for k, v in result['components'].items()},
                'metrics': result['rms']}
    report = {'process': FLOW_VERSION, 'cases': args.cases, 'initial': [], 'steps': [],
              'diagnostic_only': True, 'time': .1, 'binary_preservation_max_error': parity}
    with torch.no_grad():
        for batch, coupling in pairs:
            report['initial'].append(record(measure(batch, coupling)))
    for i in range(args.steps):
        batch, coupling = pairs[i % len(pairs)]
        trainer.optimizer.zero_grad(set_to_none=True)
        result = measure(batch, coupling)
        result['loss_total'].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError('missing or nonfinite gradients')
        if any(p.grad is not None for p in model.expert.parameters() if not p.requires_grad):
            raise AssertionError('frozen expert base received gradients')
        groups = {}
        for prefix in ('consensus_controller', 'film_heads', 'layerwise_bridge'):
            groups[prefix] = sum(float(p.grad.square().sum()) for n, p in model.trifuse.named_parameters()
                                 if n.startswith(prefix) and p.grad is not None) ** .5
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        trainer.optimizer.step()
        trainer.global_step += 1
        row = {'step': i, 'case': batch['case'], **record(result), 'gradient_norms': groups}
        report['steps'].append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with torch.no_grad():
        report['final'] = [record(measure(*pair)) for pair in pairs]
    trainer.save_checkpoint(str(out / 'smoke_model.pt'), epoch=0)
    trainer.load_checkpoint(str(out / 'smoke_model.pt'))
    report['checkpoint_roundtrip'] = True
    batch = batches[0]
    result = sample_single(model, flow, TernaryInitializer(no_torsion=model.expert.model.no_torsion),
                           batch['e3_data'], batch['poi_data'], batch['ligand_data'],
                           device=args.device, steps=args.sampling_steps, seed=0, return_trace=True)
    report['sampling'] = {k: result[k] for k in ('valid', 'termination_reason', 'consensus_rmsd', 'trace', 'collapse')}
    if not result['valid']:
        raise AssertionError('sampling rejected a nonfinite field')
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Completed: {out / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
