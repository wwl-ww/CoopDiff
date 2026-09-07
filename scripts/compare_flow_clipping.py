"""Compare two clipping thresholds on identical random paths and dropout seeds."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from diagnose_checkpoint import dump, measure
from diagnose_flow_training import grouped_norm
from train_entry import build_trainer, parse_args, seed_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-probes', required=True)
    parser.add_argument('--val-probes', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['NAN_CLAMP'] = '100'
    device = torch.device('cuda')
    train = torch.load(args.train_probes, map_location=device, weights_only=False)
    val = torch.load(args.val_probes, map_location=device, weights_only=False)
    batches = list({batch['case']: batch for batch, _, _ in train}.values())
    if {b['case'] for b in batches} & {b['case'] for b, _, _ in val}:
        raise ValueError('training and validation diagnostic structures overlap')
    config = parse_args([
        '--train_data', 'unused', '--val_data', 'unused',
        '--output_dir', str(out), '--lr', '1e-4', '--device', 'cuda'])
    model, flow, trainer = build_trainer(
        config, device, input_dim=batches[0]['e3_data']['features'].shape[-1])
    trainer.set_phase(1)
    seed_training(51000, device)
    pairs = []
    for step in range(args.steps):
        batch = batches[step % len(batches)]
        coupling = flow.sample_coupling(
            batch['e3_data']['coords'], batch['poi_data']['coords'],
            batch['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
        pairs.append((batch, coupling, float(torch.rand(()).item())))
    torch.save(pairs, out / 'training_paths.pt')

    def evaluate():
        model.eval()
        with torch.no_grad():
            rows = [measure(trainer, pair)[1] for pair in val]
        return {'mean': float(np.mean([r['loss'] for r in rows])), 'probes': rows}

    report = {'args': vars(args), 'path_seed': 51000, 'dropout_seed_start': 52000,
              'scope': 'controlled short training-subset comparison; full-data optimum not established',
              'treatments': {}}
    for clip in (1., 10.):
        trainer.load_checkpoint(args.checkpoint)
        before_val = evaluate()
        model.train()
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        losses = []
        norms = []
        with (out / f'clip_{clip:g}.jsonl').open('w') as log:
            for step, pair in enumerate(pairs):
                seed_training(52000 + step, device)
                trainer.optimizer.zero_grad(set_to_none=True)
                result, row = measure(trainer, pair)
                result['loss_total'].backward()
                row.update(step=step, global_step=trainer.global_step)
                if step % 10 == 0:
                    row['gradients'] = grouped_norm((n, p.grad) for n, p in named)
                    before = {n: p.detach().clone() for n, p in named}
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError('non-finite clipping-control gradients')
                row['preclip_norm'] = norm.item()
                row['clip_factor'] = min(1., clip / max(norm.item(), 1e-12))
                trainer.optimizer.step()
                if step % 10 == 0:
                    row['updates'] = grouped_norm((n, p - before[n]) for n, p in named)
                    print(f'clip={clip:g} step={step} loss={row["loss"]:.6g}', flush=True)
                trainer.global_step += 1
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                losses.append(row['loss'])
                norms.append(row['preclip_norm'])
        after_val = evaluate()
        trainer.save_checkpoint(str(out / f'diagnostic_clip_{clip:g}.pt'), epoch=trainer.resume_epoch)
        report['treatments'][str(clip)] = {
            'validation_before': before_val, 'validation_after': after_val,
            'training_mean': float(np.mean(losses)),
            'last_20_training_mean': float(np.mean(losses[-20:])),
            'median_preclip_norm': float(np.median(norms)),
            'clipped_fraction': float(np.mean(np.array(norms) > clip)),
        }
        dump(out / 'summary.json', report)
        print(f'clip={clip:g} validation {before_val["mean"]:.6g} -> {after_val["mean"]:.6g}', flush=True)


if __name__ == '__main__':
    main()
