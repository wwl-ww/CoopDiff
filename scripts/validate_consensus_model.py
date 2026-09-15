"""Bounded real-data gate: frozen experts, optimization, EMA and one trajectory.

Use a previously recorded TRAINING fixed_pair.pt, never benchmark inputs for
optimization. This writes smoke-only weights, not a quality-selected model.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tricoopdiff.diffdock_compat  # noqa: E402,F401

import numpy as np
import torch

from tricoopdiff.consensus import ConsensusCoopDiff, ConsensusFuse
from tricoopdiff.consensus_train import ConsensusTrainer
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.sampler import sample_single
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.reproducibility import configure_inference_reproducibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expert-model-dir', required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    if args.device == 'cuda' and (
            not os.environ.get('CUDA_VISIBLE_DEVICES') or
            os.environ['CUDA_VISIBLE_DEVICES'] == '-1'):
        parser.error('set CUDA_VISIBLE_DEVICES to explicitly authorized GPU IDs')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    configure_inference_reproducibility()
    torch.manual_seed(17)
    np.random.seed(17)
    payload = torch.load(args.pair, map_location=args.device, weights_only=False)
    batch = payload['batch']
    native_snapshot = {k: batch[k]['coords'].clone() for k in ('e3_data', 'poi_data', 'ligand_data')}
    expert = build_diffdock_expert(args.expert_model_dir, args.device)
    expert_snapshot = {n: p.detach().cpu().clone() for n, p in expert.named_parameters()}
    fuse = ConsensusFuse(input_dim=batch['e3_data']['features'].shape[-1], dropout=0)
    model = ConsensusCoopDiff(expert, fuse).to(args.device)
    flow = SDPIFlow()
    trainer = ConsensusTrainer(model, flow, device=args.device, lr=1e-4, ema_decay=0.9)
    fixed = {**batch, 'coupling_override': payload['coupling'],
             't_override': torch.tensor([payload['time']], device=args.device)}
    history = []
    for step in range(4):
        result = trainer.train_step(**copy.deepcopy(fixed))
        history.append(result)
        print(f'step={step} loss={result["loss_total"]:.6f}', flush=True)
    assert all(p.grad is None and not p.requires_grad for p in expert.parameters())
    assert all(torch.equal(p.detach().cpu(), expert_snapshot[n]) for n, p in expert.named_parameters())
    missing_gradients = [n for n, p in fuse.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing_gradients, missing_gradients
    assert all(torch.equal(batch[k]['coords'], v) for k, v in native_snapshot.items())
    validation = trainer.validate_step(**copy.deepcopy(fixed))
    trainer.save_checkpoint(output / 'smoke_only.pt', epoch=1,
                            best_val_loss=validation['loss_total'])
    before = trainer.validate_step(**copy.deepcopy(fixed))
    restored = ConsensusTrainer(model, flow, device=args.device, lr=1e-4, ema_decay=0.9)
    restored.load_checkpoint(output / 'smoke_only.pt')
    after = restored.validate_step(**copy.deepcopy(fixed))
    assert before == after, 'EMA validation differs after resume'
    with restored.ema_parameters():
        prediction = sample_single(model, flow, TernaryInitializer(),
                                   batch['e3_data'], batch['poi_data'], batch['ligand_data'],
                                   steps=20, seed=17, return_trace=True)
    assert np.isfinite(prediction['ligand_coords']).all()
    assert np.isfinite(prediction['poi_coords']).all()
    assert len([s for s in prediction['trace'] if s['stage'] == 'step']) == 20
    summary = {'case': batch.get('case'), 'training': history, 'ema_validation': validation,
               'expert_unchanged': True, 'all_trainable_gradients_present': True,
               'checkpoint_ema_roundtrip_exact': True, 'trajectory_steps': 20,
               'quality_claim': False, 'pair': args.pair,
               'expert': expert.provenance}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'trace.json').write_text(json.dumps(prediction['trace'], indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
