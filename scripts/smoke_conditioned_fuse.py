"""CPU-only migration of real EMA fusion weights using explicitly synthetic inputs."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TRICOOPDIFF_CACHE_DIR', str(ROOT / 'results/runtime_cache/diffdock'))
import torch
from rdkit import Chem

from evaluation_common import sha256_file
from tricoopdiff.conditioning.condition_features import SideCondition, build_condition_features
from tricoopdiff.conditioning.conditioned_consensus_train import ConditionedConsensusFuse, load_prototype_checkpoint
from tricoopdiff.models.consensus import BRANCHES, ConsensusFuse
from tricoopdiff.train import validate_consensus_checkpoint


def velocity_vector(result):
    return torch.cat([result['normalized_velocity'][b][c].flatten() for b in BRANCHES for c in ('tr', 'rot')]
                     + [result['normalized_velocity']['torsion'].flatten()])


def synthetic_case(input_dim):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1715)
        graph = {'node_features': torch.zeros(15, input_dim), 'node_coords': torch.randn(15, 3),
                 'node_types': torch.tensor([0] * 5 + [1] * 6 + [2] * 4),
                 'lig_edge_index': torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
                 'lig_edge_mask': torch.tensor([False, False, True, False, False, False]),
                 'lig_mask_rotate': torch.tensor([[False, False, True, True]])}
        proposals = {b: {c: torch.randn(1, 1 if c == 'tor' else 3) for c in ('tr', 'rot', 'tor')} for b in BRANCHES}
    return graph, proposals, torch.zeros(1, 9), Chem.MolFromSmiles('CCON')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.out.exists():
        raise FileExistsError('use a new migration smoke output')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_consensus_checkpoint(checkpoint)
    base = ConsensusFuse(**checkpoint['model_config']).eval()
    base.load_state_dict(checkpoint['ema_trifuse_state_dict'], strict=True)
    model = ConditionedConsensusFuse.from_base(base).eval()
    graph, proposals, state, molecule = synthetic_case(base.config['input_dim'])
    empty = build_condition_features(molecule, (5, 6))
    side = SideCondition(provenance='template', evidence_id='synthetic-smoke-only',
                         fragment=Chem.MolFromSmiles('CO'), atom_map=(1, 2),
                         pocket_probabilities=[.8, .3, .1, 0., 0.])
    conditions = build_condition_features(molecule, (5, 6), (side, None))
    checks = []
    for source in (0, 1):
        expected = base(graph, proposals, state, source=source)
        plain = model(graph, proposals, state, source=source)
        absent = model(graph, proposals, state, source=source, conditions=empty)
        conditional = model(graph, proposals, state, source=source, conditions=conditions)
        matches = all(torch.equal(velocity_vector(expected), velocity_vector(r))
                      and torch.equal(expected['rank_features'], r['rank_features'])
                      for r in (plain, absent, conditional))
        delta = float((velocity_vector(conditional) - velocity_vector(expected)).detach().norm())
        if not matches or not torch.isfinite(velocity_vector(conditional)).all() or delta != 0:
            raise AssertionError('EMA migration/condition effect check failed')
        model.zero_grad(set_to_none=True)
        (velocity_vector(conditional) - .5).square().mean().backward()
        encoder_gradient = sum(float(p.grad.detach().norm()) for p in model.condition_encoder.parameters()
                               if p.grad is not None)
        scale_gradient = float(model.condition_scale.grad.detach().abs())
        if not scale_gradient > 0 or encoder_gradient != 0:
            raise AssertionError('zero-migration gate did not receive the first condition gradient')
        checks.append({'source': source, 'all_conditions_exact_parent_match': matches,
                       'conditional_velocity_delta_l2': delta,
                       'condition_scale_gradient_norm': scale_gradient,
                       'condition_encoder_gradient_norm_sum': encoder_gradient})
    parent_hash = sha256_file(args.checkpoint)
    prototype = model.prototype_checkpoint(parent_hash)
    restored = load_prototype_checkpoint(prototype)
    if not all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()):
        raise AssertionError('prototype checkpoint roundtrip changed weights')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(prototype, args.out / 'prototype_untrained.pt')
    report = {'scope': 'real parent EMA weights, synthetic graph/proposals, no DockQ accuracy claim',
              'parent_checkpoint_sha256': parent_hash, 'parent_version': checkpoint['checkpoint_version'],
              'weight_field': 'ema_trifuse_state_dict', 'model_config': base.config,
              'device': 'cpu', 'optimizer_steps': 0, 'checkpoint_roundtrip_exact': True, 'checks': checks,
              'code_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                  Path(__file__), ROOT / 'tricoopdiff/conditioning/conditioned_consensus_train.py', ROOT / 'tricoopdiff/models/consensus.py',
                  ROOT / 'tricoopdiff/conditioning/condition_features.py', ROOT / 'tricoopdiff/models/consensus.py')}}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
