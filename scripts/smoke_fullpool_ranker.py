"""One real training-case backward pass; no deployment checkpoint or efficacy claim."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat  # noqa: E402,F401
import torch

from train_structural_ranker import initialize_encoder
from tricoopdiff.structural_ranker import (StructuralQualityRanker, decision_ranking_loss,
                                          structural_ranking_loss, budget_retention_loss, component_quality_loss)
from tricoopdiff.structural_ranking_data import StructuralArchiveDataset, future_utility_case, file_digest
from tricoopdiff.ranking_backward import backward_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--encoder-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--component-weight', type=float, default=0.)
    parser.add_argument('--interface-layers', type=int, default=0)
    parser.add_argument('--interface-readout', choices=('pooled', 'contacts', 'contacts-independent'), default='pooled')
    parser.add_argument('--wait-label-seconds', type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.component_weight) or args.component_weight < 0:
        parser.error('component weight must be finite and nonnegative')
    if args.out.exists():
        parser.error('use a new report path')
    if args.wait_label_seconds < 0 or args.chunk_size < 1:
        parser.error('invalid wait or chunk size')
    deadline = time.monotonic() + args.wait_label_seconds
    label_path = args.labels / f'{args.case}.json'
    if not label_path.exists():
        print(f'Waiting for sealed labels: {label_path}', flush=True)
    while not label_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('full-pool labels were not produced before deadline')
        time.sleep(min(5., max(.01, deadline - time.monotonic())))
    dataset = StructuralArchiveDataset(args.archives, args.labels, 'train')
    case = dataset[dataset.cases.index(args.case)]
    future = future_utility_case(case, target='union')
    if future is None or len(future['graphs']) != case['pool_size'] or case['pool_size'] <= 64:
        raise ValueError('complete full-pool outcomes required')
    torch.manual_seed(7315)
    model = StructuralQualityRanker(case['graphs'][0]['node_features'].shape[1], 128,
                                    component_aux=args.component_weight > 0,
                                    interface_layers=args.interface_layers,
                                    interface_readout=args.interface_readout).to(args.device)
    if args.component_weight and not bool(case['component_mask'].all()):
        raise ValueError('complete component labels required')
    initialization = initialize_encoder(model, args.encoder_checkpoint, 'ema')
    model.quality_head.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    if str(args.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    quality, _ = backward_list(model, case, args.device, lambda out: decision_ranking_loss(
        out, case['labels'].to(args.device), case['stages'], top_weight=1.) + (
            args.component_weight * component_quality_loss(out, case['component_targets'].to(args.device))
            if args.component_weight else 0.), chunk_size=args.chunk_size)
    utility, _ = backward_list(model, future, args.device, lambda out: (
        structural_ranking_loss(out, future['labels'].to(args.device), top_weight=1.)
        + budget_retention_loss(out['score'], future['labels'].to(args.device))), chunk_size=args.chunk_size)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    for module in (model.rank_head, model.future_head, model.encoder.protein_feature_proj):
        if module.weight.grad is None or not bool(module.weight.grad.abs().sum() > 0):
            raise ValueError('missing trained-module gradient')
    if args.component_weight and not bool(model.component_head.weight.grad.abs().sum() > 0):
        raise ValueError('missing component-head gradient')
    optimizer.step()
    gpu = str(args.device).startswith('cuda')
    if gpu:
        torch.cuda.synchronize()
    report = {'scope': 'one training case resource/gradient smoke, not validation efficacy',
              'case': args.case, 'candidates': case['pool_size'], 'quality_states': len(case['graphs']),
              'chunk_size': args.chunk_size, 'quality_loss': float(quality), 'future_loss': float(utility),
              'component_weight': args.component_weight,
              'interface_layers': args.interface_layers, 'interface_readout': args.interface_readout,
              'gradient_norm_before_clip': float(norm), 'seconds': time.monotonic() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated() if gpu else None,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved() if gpu else None,
              'initialization': initialization, 'archive_sha256': dataset.seals[args.case]['archive_sha256'],
              'entrypoint_sha256': file_digest(Path(__file__))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
