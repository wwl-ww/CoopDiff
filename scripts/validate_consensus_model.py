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

from tricoopdiff.models.consensus import ConsensusCoopDiff, ConsensusFuse, production_model_config
from tricoopdiff.train import ConsensusTrainer
from tricoopdiff.models.expert import build_diffdock_expert
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.initialization.endpoints import reachable_ligand_target
from tricoopdiff.sampler import sample_single
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.runtime import configure_inference_reproducibility


def audit_training_contracts(args):
    """Single-seed, same-weight diagnostics; never update the running trainer."""
    from argparse import Namespace
    from train_entry import (build_trainer, TernaryDataLoader, _load_score_model_args,
                             validation_metric_context)
    from tricoopdiff.metrics.dockq import compute_fullatom_dockq
    from tricoopdiff.runtime import atomic_json, capture_rng_state, restore_rng_state
    from tricoopdiff.inputs.graphs import build_binary_graph
    from tricoopdiff.models.expert import _set_time_embeddings
    from tricoopdiff.train import clip_grad_norm_finite

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    config = Namespace(**json.loads(Path(args.audit_config).read_text()))
    config.resume = args.audit_checkpoint
    config.expert_coordinate_grad = config.structural_supervision = None
    model, flow, trainer = build_trainer(config, torch.device(args.device), input_dim=1281)
    loader = TernaryDataLoader(config.train_data, _load_score_model_args(config.expert_model_dir),
                              device=args.device, cache_dir=config.cache_dir, cache_only=True,
                              fullatom_supervision=True)
    entry = loader.entries_by_case[args.audit_case]
    batch = loader._load_one(entry)
    trainer.load_checkpoint(args.audit_checkpoint)
    saved_coordinate_grad = model.expert_coordinate_grad
    saved_structural = trainer.structural_supervision
    ed, pd, ld = (batch[key] for key in ('e3_data', 'poi_data', 'ligand_data'))
    ligand, poi, body, _ = TernaryInitializer(no_torsion=model.expert.model.no_torsion).initialize(
        ed['coords'][0], pd['coords'][0], ld, seed=73)
    fixed = {**batch, 'source': 0, 'coupling_override': {'initial_state':
             {'ligand': ligand, 'poi': poi, 'body_frame': body}}}
    metric_context = validation_metric_context(entry, config.native_root)
    def metric(ligand, poi):
        return compute_fullatom_dockq(ligand[0].detach().cpu().numpy(), poi[0].detach().cpu().numpy(),
                                      ed['coords'][0].cpu().numpy(), **metric_context)['dockq']
    report = {'case': args.audit_case, 'seed': 73, 'checkpoint': args.audit_checkpoint,
              'section': args.audit_section, 'quality_claim': False}
    if args.audit_section == 'gradients':
        vectors, rows = [], []
        rng = capture_rng_state()
        for enabled in (True, False):
            trainer.load_checkpoint(args.audit_checkpoint)
            model.expert_coordinate_grad = enabled
            trainer.structural_supervision = True
            restore_rng_state(rng)
            model.train()
            model.zero_grad(set_to_none=True)
            result = trainer.compute_loss(**fixed, diagnostics=True)
            result['loss_total'].backward()
            parameters = [p for p in model.trifuse.parameters() if p.requires_grad]
            vector = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().double().cpu()
                                for p in parameters])
            vectors.append(vector)
            row = {'expert_coordinate_grad': enabled, 'loss': float(result['loss_total'].detach()),
                   'grad_norm': float(vector.norm()), 'finite': bool(torch.isfinite(vector).all()),
                   'dockq_before': metric(result['predicted_ligand'], result['predicted_poi']),
                   'components': {key: float(value.detach()) for key, value in result['components'].items()}}
            if row['finite']:
                clip_grad_norm_finite(parameters, 1.)
                trainer.optimizer.step()
                with torch.no_grad():
                    after = trainer.compute_loss(**fixed, diagnostics=True)
                row.update(loss_after_one_update=float(after['loss_total']),
                           dockq_after_one_update=metric(after['predicted_ligand'], after['predicted_poi']))
            rows.append(row)
            report['comparisons'] = rows
            atomic_json(report, output / 'gradients.json')
            # Restore the checkpoint's contracts before loading the next arm.
            model.expert_coordinate_grad = saved_coordinate_grad
            trainer.structural_supervision = saved_structural
        report['gradient_cosine'] = float(torch.nn.functional.cosine_similarity(vectors[0][None], vectors[1][None]))
    elif args.audit_section == 'boundary':
        model.eval()
        expert = model.expert.model
        t = ligand.new_tensor([.5])
        cutoff = float(flow.t_to_sigma(t, t, t)[0] * 3 + 20) if expert.dynamic_max_cross else float(expert.cross_max_distance)
        direction = ligand.new_tensor([1., 0., 0.])
        rows = []
        for epsilon in (.1, .01, .001):
            outputs, counts = [], []
            for sign in (-1, 1):
                coords = ld['coords'][0] - ld['coords'][0, 0] + ed['coords'][0, 0] + direction * (cutoff + sign * epsilon)
                graph = build_binary_graph(ed['coords'][0], ed['features'][0], coords, ld['features'][0],
                    rec_edge_index=ed['edge_index'], lig_edge_index=ld['edge_index'], lig_edge_attr=ld['edge_attr'],
                    lig_edge_mask=ld['edge_mask'], lig_mask_rotate=ld['mask_rotate'])
                _set_time_embeddings(graph, t, expert)
                with torch.no_grad():
                    values = model.expert.forward_features(graph, return_vectors=True)
                distance = torch.cdist(coords, ed['coords'][0], compute_mode='donot_use_mm_for_euclid_dist')
                counts.append(int((distance < cutoff).sum()))
                outputs.append(torch.cat([value.flatten() for value in values[:4]]))
            rows.append({'epsilon_angstrom': epsilon, 'cross_pair_counts': counts,
                         'output_delta_norm': float((outputs[1] - outputs[0]).norm()),
                         'delta_over_2epsilon': float((outputs[1] - outputs[0]).norm() / (2 * epsilon))})
        report.update(smooth_edges=bool(expert.smooth_edges), cutoff_angstrom=cutoff, perturbations=rows)
    else:
        model.eval()
        rows = []
        reference = None
        for steps in (20, 40, 80):
            prediction = sample_single(model, flow, TernaryInitializer(no_torsion=model.expert.model.no_torsion),
                ed, pd, ld, steps=steps, seed=73, device=args.device, allow_step_change=True)
            coords = torch.as_tensor(prediction['ligand_coords'], device=args.device)[None]
            predicted_poi = torch.as_tensor(prediction['poi_coords'], device=args.device)[None]
            if reference is None:
                reference = coords
            rows.append({'steps': steps, 'dockq': metric(coords, predicted_poi),
                         'ligand_rmsd_from_20step': float((coords - reference).square().sum(-1).mean().sqrt()),
                         'numerically_valid': prediction['numerically_valid']})
            report['same_seed_step_comparison'] = rows
            atomic_json(report, output / 'steps.json')
        report['interpretation'] = 'Both dt conditioning and integration change; not an integration-accuracy claim.'
    atomic_json(report, output / (args.audit_section + '.json'))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair')
    parser.add_argument('--audit-config')
    parser.add_argument('--audit-checkpoint')
    parser.add_argument('--audit-case', default='4X28_B_C_FDA')
    parser.add_argument('--audit-section', choices=('gradients', 'boundary', 'steps'), default='gradients')
    parser.add_argument('--out', required=True)
    parser.add_argument('--expert-model-dir', required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    if args.device == 'cuda' and (
            not os.environ.get('CUDA_VISIBLE_DEVICES') or
            os.environ['CUDA_VISIBLE_DEVICES'] == '-1'):
        parser.error('set CUDA_VISIBLE_DEVICES to explicitly authorized GPU IDs')
    if args.audit_config:
        if not args.audit_checkpoint:
            parser.error('--audit-config requires --audit-checkpoint')
        torch.set_num_threads(2)
        return audit_training_contracts(args)
    if not args.pair:
        parser.error('--pair is required outside audit mode')
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
    fuse = ConsensusFuse(**production_model_config(128, batch['e3_data']['features'].shape[-1], 3))
    model = ConsensusCoopDiff(expert, fuse).to(args.device)
    flow = SDPIFlow()
    trainer = ConsensusTrainer(model, flow, device=args.device, lr=1e-4, ema_decay=0.9)
    ligand, poi, body, _ = TernaryInitializer(no_torsion=expert.model.no_torsion).initialize(
        batch['e3_data']['coords'][0], batch['poi_data']['coords'][0], batch['ligand_data'], seed=17)
    target, _ = reachable_ligand_target(batch['ligand_data'], ligand, poi,
                                      batch['poi_data']['coords'], no_torsion=expert.model.no_torsion)
    fixed = {**batch, 'coupling_override': {
        'initial_state': {'ligand': ligand, 'poi': poi, 'body_frame': body},
        'target_ligand': target}}

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
