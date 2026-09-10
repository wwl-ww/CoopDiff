"""Diagnostic: fit one cached teacher state; never a docking-quality benchmark."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tricoopdiff.diffdock_compat
from factorial_arm import PairedTorsionFlow, restrict_non_single
from featurize_common import _to_device_and_batch
from train_entry import _load_score_model_args
from tricoopdiff.checkpoint_contract import validate_expert_base_identity
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import ExperimentalConsensus
from tricoopdiff.expert_factory import build_diffdock_expert


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--case', default='3CXH_C_E_6PH')
    ap.add_argument('--steps', type=int, default=100)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    np.random.seed(1729)
    root = args.artifacts
    ckpt = root / 'results/overnight_root_cause_v3_run04/ccd_single_bond_frozen_seed1729/best.pt'
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    runtime = saved['runtime_config']
    os.environ['NAN_CLAMP'] = str(runtime['activation_clamp'])
    os.environ['NAN_CLAMP_MODE'] = runtime['activation_clamp_mode']
    expert = build_diffdock_expert(str(root / 'checkpoints/score_model'), 'cuda')
    validate_expert_base_identity(saved, expert)
    fuse = ConsensusFuse(**saved['model_config']).cuda().eval()
    fuse.load_state_dict(saved['ema_trifuse_state_dict'])
    model = ExperimentalConsensus(expert, fuse, saved['expert_lora_config']).cuda().eval()
    cache = root / f'results/overnight_root_cause_v3_plan/cache_ccd/{args.case}.pt'
    batch = torch.load(cache, map_location='cuda', weights_only=False)
    assert batch['cache_version'] == 7
    for key in ('e3_data', 'poi_data', 'ligand_data'):
        batch[key] = _to_device_and_batch(batch[key], 'cuda')
    entries = json.loads((root / 'results/overnight_root_cause_v3_plan/indexes/ternary_val_ccd.json').read_text())['entries']
    entry = next(e for e in entries if e['case'] == args.case)
    digest = hashlib.sha256()
    for key in ('protein1', 'protein2', 'ligand'):
        digest.update(entry[key].encode())
        digest.update((root / entry[key].removeprefix('/tmp/coopdiff/')).read_bytes())
    score_args = _load_score_model_args(str(root / 'checkpoints/score_model'))
    digest.update(json.dumps(vars(score_args), sort_keys=True, default=str).encode())
    assert digest.hexdigest() == batch['source_signature'], 'historical cache source identity mismatch'
    batch = restrict_non_single(batch, str(root / entry['ligand'].removeprefix('/tmp/coopdiff/')))
    e3, poi, lig = (batch[k] for k in ('e3_data', 'poi_data', 'ligand_data'))
    flow = PairedTorsionFlow()
    t = torch.tensor([0.5], device='cuda')
    dt = torch.tensor([0.05], device='cuda')
    coupling = flow.sample_coupling(e3['coords'], poi['coords'], lig)
    ligand, protein, body, target = flow.path(lig, poi['coords'], coupling, t)
    captured = []
    hook = fuse.register_forward_pre_hook(lambda module, inputs: captured.append(inputs))
    with torch.no_grad():
        sigma = flow.t_to_sigma(t, t, t)[0]
        model(e3, {**poi, 'coords': protein}, {**lig, 'coords': ligand}, t,
              r_e3_to_poi=body, receptor_crop_cutoff=3 * sigma + 20,
              diffusion_state=flow.conditioning(t, dt, e3['coords'], poi['coords']))
    hook.remove()
    graph, proposals, conditioning = captured[0]
    scales = flow.scales(e3['coords'], poi['coords'])
    targets = {}
    for i, b in enumerate(('e3_ligand', 'poi_ligand')):
        targets[b] = {'tr': target[b]['tr'] / scales[i], 'rot': target[b]['rot'] / np.pi}
    original = copy.deepcopy(fuse.state_dict())
    optimizer = torch.optim.AdamW([p for p in fuse.parameters() if p.requires_grad], lr=1e-4)
    history = []
    for step in range(args.steps + 1):
        output = fuse(graph, proposals, conditioning)
        pred = output['normalized_velocity']
        losses = {f'{b}_{c}': (pred[b][c] - targets[b][c]).square().mean()
                  for b in targets for c in ('tr', 'rot')}
        tor = pred['torsion'] - target['torsion'] / np.pi
        losses['torsion'] = tor.square().mean() if tor.numel() else t.new_zeros(())
        loss = sum(losses.values()) + runtime['residual_weight'] * output['residual_penalty']
        assert torch.isfinite(loss)
        row = {'step': step, 'loss': loss.item(), **{k: v.item() for k, v in losses.items()}}
        history.append(row)
        if step % 10 == 0:
            print(json.dumps(row), flush=True)
        if step == args.steps:
            break
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(fuse.parameters(), 1.0)
        assert torch.isfinite(grad)
        row['gradient_norm'] = grad.item()
        optimizer.step()
    delta = {k: float((v - original[k]).norm()) for k, v in fuse.state_dict().items() if v.is_floating_point()}
    report = {'scope': 'one fixed teacher state, EMA start, eval-mode optimization, cached frozen proposals; not free rollout or training validation',
              'case': args.case, 'time': 0.5, 'seed': 1729, 'history': history,
              'changed_tensors': sum(v > 0 for v in delta.values()), 'parameter_deltas': delta}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
