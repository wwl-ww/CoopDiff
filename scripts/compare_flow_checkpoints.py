"""Compare saved epochs on identical validation paths without parameter updates."""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from diagnose_checkpoint import dump, measure
from train_entry import build_trainer, parse_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probes', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    os.environ['NAN_CLAMP'] = '100'
    device = torch.device('cuda')
    probes = torch.load(args.probes, map_location=device, weights_only=False)
    config = parse_args([
        '--train_data', 'unused', '--val_data', 'unused',
        '--output_dir', str(out.parent), '--lr', '1e-4', '--device', 'cuda'])
    model, _, trainer = build_trainer(
        config, device, input_dim=probes[0][0]['e3_data']['features'].shape[-1])
    trainer.set_phase(1)
    report = {'probes': args.probes, 'scope': 'same independent validation paths; no updates',
              'checkpoints': []}
    for checkpoint in args.checkpoints:
        trainer.load_checkpoint(checkpoint)
        model.eval()
        with torch.no_grad():
            rows = [measure(trainer, pair)[1] for pair in probes]
        entry = {'path': checkpoint, 'epoch': trainer.resume_epoch,
                 'global_step': trainer.global_step,
                 'sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                 'saved_best_val_loss': trainer.resume_best_val_loss,
                 'probe_mean_loss': float(np.mean([r['loss'] for r in rows])),
                 'components': {k: float(np.mean([r['components'][k] for r in rows]))
                                for k in rows[0]['components']}, 'probes': rows}
        report['checkpoints'].append(entry)
        dump(out, report)
        print('epoch', entry['epoch'], 'same-path loss', entry['probe_mean_loss'], flush=True)


if __name__ == '__main__':
    main()
