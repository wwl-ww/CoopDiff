"""Bounded DeepTernary TRAIN-case smoke for model-4.5; no quality claims.

Uses cached features only. Runs real frozen experts, endpoint optimization,
checkpoint resume, and the same 20-step sampler used in production.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tricoopdiff.diffdock_compat  # noqa: E402,F401

import numpy as np
import torch

from featurize_common import featurize_complex
from train_entry import _load_index, _load_score_model_args, fullatom_targets
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.models.consensus import (
    ConsensusCoopDiff, ConsensusFuse, PRODUCTION_MODEL_PROFILE, production_model_config,
)
from tricoopdiff.models.expert import build_diffdock_expert
from tricoopdiff.runtime import atomic_json, configure_inference_reproducibility, evaluation_context
from tricoopdiff.sampler import sample_single
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.train import ConsensusTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-index', default='data/ternary_train.json')
    parser.add_argument('--case', default='4X28_B_C_FDA')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--node-dim', type=int, default=32)
    parser.add_argument('--expert-coordinate-grad', action='store_true')
    args = parser.parse_args()
    if args.updates < 1:
        parser.error('--updates must be positive')
    if args.device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('select a physical GPU explicitly with CUDA_VISIBLE_DEVICES')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    configure_inference_reproducibility()
    torch.set_num_threads(2)
    torch.manual_seed(73)
    np.random.seed(73)
    entries, _, by_case = _load_index(args.train_index)
    if args.case not in by_case:
        raise ValueError('smoke case must belong to the supplied training index')
    entry = by_case[args.case]
    score_args = _load_score_model_args(args.expert_model_dir)
    ed, pd, ld, input_dim = featurize_complex(args.case, entry['protein1'], entry['protein2'],
        entry['ligand'], score_args, args.device, cache_dir=args.cache_dir,
        cache_read_only=True, cache_only=True)
    batch = dict(case=args.case, e3_data=ed, poi_data=pd, ligand_data=ld,
                 fullatom_targets=fullatom_targets(entry, ed, pd))
    expert = build_diffdock_expert(args.expert_model_dir, args.device)
    expert_snapshot = {name: value.detach().cpu().clone() for name, value in expert.state_dict().items()}
    fuse = ConsensusFuse(**production_model_config(args.node_dim, input_dim, 1,
        model_profile=PRODUCTION_MODEL_PROFILE))
    model = ConsensusCoopDiff(expert, fuse, checkpoint_ternary=True,
                             expert_coordinate_grad=args.expert_coordinate_grad).to(args.device)
    flow = SDPIFlow(inference_steps=20)
    trainer = ConsensusTrainer(model, flow, device=args.device, lr=1e-4, activation_clamp=100.,
                               structural_supervision=True, steric_loss_weight=.05)
    initializer = TernaryInitializer(no_torsion=expert.model.no_torsion)
    ligand, poi, body, _ = initializer.initialize(ed['coords'][0], pd['coords'][0], ld, seed=73)
    fixed = {**batch, 'coupling_override': {'initial_state':
        dict(ligand=ligand, poi=poi, body_frame=body)}}
    report = {'case': args.case, 'model_profile': PRODUCTION_MODEL_PROFILE, 'seed': 73,
              'expert_coordinate_grad': args.expert_coordinate_grad, 'quality_claim': False,
              'config': vars(args), 'training': []}
    for step in range(args.updates):
        result = trainer.train_step(**fixed)
        gradients = {name: p.grad for name, p in fuse.named_parameters() if p.requires_grad}
        assert all(g is not None and torch.isfinite(g).all() for g in gradients.values()), \
            [name for name, g in gradients.items() if g is None or not torch.isfinite(g).all()]
        report['training'].append(result)
        atomic_json(report, output / 'progress.json')
        print(f'update={step + 1} endpoint={result["loss_endpoint_main"]:.6f} total={result["loss_total"]:.6f}', flush=True)
    assert all(not p.requires_grad and p.grad is None for p in expert.parameters())
    assert all(torch.equal(value.cpu(), expert_snapshot[name]) for name, value in expert.state_dict().items())
    print('checking EMA endpoint / sampler parity', flush=True)
    with trainer.ema_parameters(), evaluation_context(model):
        expected = trainer.compute_loss(**fixed, diagnostics=True)
        prediction = sample_single(model, flow, initializer, ed, pd, ld, seed=73,
                                   device=args.device, return_trace=True)
    for name in ('ligand', 'poi'):
        assert np.allclose(prediction[name + '_coords'], expected['predicted_' + name][0].cpu().numpy(), atol=2e-4), \
            'training and sampling must execute the same local compatibility velocities'
    poi_final = torch.as_tensor(prediction['poi_coords'], device=pd['coords'].device)
    # Direct pair vectors avoid float32 cdist cancellation on translated bodies.
    before = torch.linalg.vector_norm(pd['coords'][0, :, None] - pd['coords'][0, None], dim=-1)
    after = torch.linalg.vector_norm(poi_final[:, None] - poi_final[None], dim=-1)
    assert (before - after).abs().max() < 1e-3
    assert prediction['numerically_valid']
    assert len([row for row in prediction['trace'] if row['stage'] == 'step']) == 20
    print('checking checkpoint resume', flush=True)
    before_validation = trainer.validate_step(**fixed)
    trainer.save_checkpoint(output / 'smoke_only.pt', epoch=1, best_val_loss=before_validation['loss_total'])
    with torch.no_grad():
        next(fuse.parameters()).add_(.1)
    trainer.load_checkpoint(output / 'smoke_only.pt')
    after_validation = trainer.validate_step(**fixed)
    assert before_validation == after_validation, 'EMA checkpoint roundtrip must be exact'
    report.update(expert_unchanged=True, trainable_gradients_finite=True, trajectory_steps=20,
                  endpoint_sampler_parity=True, checkpoint_ema_roundtrip_exact=True,
                  maximum_protein_distance_error=float((before - after).abs().max()),
                  validation=after_validation,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
    atomic_json(report, output / 'summary.json')
    atomic_json(prediction['trace'], output / 'trace.json')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
