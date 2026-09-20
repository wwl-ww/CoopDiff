"""Train-only, one-case capacity diagnostic; never exports a deployment model."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

import tricoopdiff.diffdock_compat  # noqa: F401
import torch

from train_structural_ranker import geometry_shuffle_diagnostic, initialize_encoder, ordering_metrics
from tricoopdiff.structural_ranker import StructuralQualityRanker, structural_ranking_loss
from tricoopdiff.structural_ranking_data import StructuralArchiveDataset, file_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--source', choices=('sdpi', 'bitseed'), required=True)
    parser.add_argument('--stage', choices=('initial', 'final'), default='initial')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--interface-layers', type=int, default=0)
    parser.add_argument('--encoder-checkpoint', type=Path)
    parser.add_argument('--encoder-weights', choices=('model', 'ema'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.steps < 1 or args.interface_layers < 0:
        parser.error('positive steps and a new report path required')
    if (args.encoder_checkpoint is None) != (args.encoder_weights is None):
        parser.error('checkpoint and explicit encoder weight choice must be provided together')
    dataset = StructuralArchiveDataset(args.archives, args.labels, 'train')
    case = dataset[dataset.cases.index(args.case)]
    ids = [i for i, (stage, source) in enumerate(zip(case['stages'], case['sources']))
           if stage == args.stage and source == args.source]
    if len(ids) < 2:
        raise ValueError('at least two same-flag training states required')
    labels = case['labels'][ids].to(args.device)
    if float(labels.max() - labels.min()) <= 1e-4:
        raise ValueError('no nontrivial quality ordering in diagnostic group')
    graphs = [{k: v.to(args.device) for k, v in case['graphs'][i].items()} for i in ids]
    torch.manual_seed(args.seed)
    model = StructuralQualityRanker(graphs[0]['node_features'].shape[1], 128,
                                    interface_layers=args.interface_layers).to(args.device)
    initialization = (initialize_encoder(model, args.encoder_checkpoint, args.encoder_weights)
                      if args.encoder_checkpoint is not None else 'random')
    for module in (model.stage_embedding, model.source_embedding, model.future_head, model.quality_head):
        module.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    stages, sources = [args.stage] * len(ids), [args.source] * len(ids)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    @torch.no_grad()
    def measure():
        model.eval()
        out = model.forward_list(graphs, stages=stages, sources=sources)
        scores = out['score'].cpu()
        truth = labels.cpu()
        return {'loss': float(structural_ranking_loss(out, labels)),
                **ordering_metrics(scores, truth),
                'top1_dockq': float(truth[scores.argmax()]), 'oracle': float(truth.max()),
                'shuffle': geometry_shuffle_diagnostic(scores, truth, stages, sources,
                                                        repeats=64, seed=args.seed)}

    before = measure()
    history = []
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_list(graphs, stages=stages, sources=sources)
        loss = structural_ranking_loss(out, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite diagnostic loss')
        loss.backward()
        gradient = model.encoder.pair_net[1].weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or not bool(gradient.abs().sum() > 0):
            raise FloatingPointError('missing/nonfinite/zero structural ranking gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        history.append(float(loss.detach()))
        if (step + 1) % 10 == 0:
            print(f'diagnostic step={step + 1} loss={history[-1]:.6f}', flush=True)
    after = measure()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    report = {'scope': 'single training-case capacity diagnostic; NOT generalization',
              'case': args.case, 'stage': args.stage, 'source': args.source, 'states': len(ids),
              'seed': args.seed, 'steps': args.steps, 'hidden_dim': 128, 'lr': 3e-4,
              'interface_layers': args.interface_layers, 'encoder_initialization': initialization,
              'flags_frozen': True, 'checkpoint_exported': False,
              'archive_sha256': file_digest(args.archives / f'{args.case}.pt'),
              'labels_sha256': file_digest(args.labels / f'{args.case}.json'),
              'entrypoint_sha256': file_digest(Path(__file__)),
              'seconds': time.perf_counter() - started,
              'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
              'before': before, 'after': after, 'loss_history': history}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: report[k] for k in ('case', 'states', 'seconds', 'before', 'after')}, indent=2))


if __name__ == '__main__':
    main()
