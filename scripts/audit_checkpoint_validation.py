"""Evaluate both TriFuse conditioning directions on held-out cluster members."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from diagnose_checkpoint import dump, measure
from flow_condition_audit import audit_conditioning
from train_entry import (TernaryDataLoader, _load_score_model_args,
                         build_trainer, parse_args, seed_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--paths', type=int, default=3)
    parser.add_argument('--seed', type=int, default=30100)
    args = parser.parse_args()
    if args.paths < 1:
        parser.error('paths must be positive')
    index = json.loads(Path(args.index).read_text())
    if index.get('metadata', {}).get('split') != 'val':
        raise ValueError('requires an explicitly labeled official validation subset')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['NAN_CLAMP'] = '100'
    device = torch.device('cuda')
    config = parse_args([
        '--train_data', args.index, '--val_data', args.index,
        '--output_dir', str(out), '--lr', '1e-4', '--device', 'cuda'])
    loader = TernaryDataLoader(
        args.index, _load_score_model_args(config.expert_model_dir),
        shuffle=False, device='cuda', cache_dir='results/featurize_cache')
    batches = [loader._load_one(entry) for entry in loader.entries]
    model, flow, trainer = build_trainer(
        config, device, input_dim=batches[0]['e3_data']['features'].shape[-1])
    trainer.set_phase(1)
    trainer.load_checkpoint(args.checkpoint)
    model.eval()
    seed_training(args.seed, device)
    probes = []
    for path_id in range(args.paths):
        for batch in batches:
            coupling = flow.sample_coupling(
                batch['e3_data']['coords'], batch['poi_data']['coords'],
                batch['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
            probes.extend((batch, coupling, t) for t in (.1, .5, .9))
    torch.save(probes, out / 'probes.pt')
    report = {'args': vars(args), 'epoch': trainer.resume_epoch,
              'global_step': trainer.global_step,
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'scope': 'official validation clusters; not a benchmark or a hyperparameter search',
              'cases': [b['case'] for b in batches]}
    with torch.no_grad():
        report['validation'] = [measure(trainer, pair)[1] for pair in probes]
    report['mean_loss'] = float(np.mean([r['loss'] for r in report['validation']]))
    dump(out / 'summary.json', report)
    print('Validation mean loss:', report['mean_loss'], flush=True)
    for branch in ('e3_ligand', 'poi_ligand'):
        seed_training(args.seed + 10000, device)
        result = audit_conditioning(trainer, probes, branch=branch)
        report[branch] = result
        dump(out / 'summary.json', report)
        print(branch, 'correct:', result['correct_mean'],
              'shuffled:', result['shuffled_mean'], flush=True)


if __name__ == '__main__':
    main()
