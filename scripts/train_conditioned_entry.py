"""Train the isolated PROTAC-conditioned model from a frozen C checkpoint.

The first protocol preserves C's 1:1 SDPI/BiTSeed endpoint schedule. Native
ternary coordinates supervise only discrete 6 A pocket/ligand contact masks;
they are never model inputs or checkpoint contents. Checkpoint selection uses
all six validation cells: endpoint source x full/pocket-only/absent conditions.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from train_entry import (TernaryDataLoader, _guarded_train_batch, _load_entries,
                         _load_score_model_args, append_metric, prepare_source_batch,
                         seed_training)
from featurize_common import featurize_complex
from tricoopdiff.conditioning.condition_augmentation import ConditionFeatureDropout, dropout_condition_features
from tricoopdiff.conditioning.conditioned_consensus_train import (
    ConditionedConsensusTrainer, build_conditioned_model, checkpoint_sha256,
    supervised_condition_batch,
)
from tricoopdiff.train import validate_assembly_checkpoint
from tricoopdiff.models.expert import build_diffdock_expert
from tricoopdiff.initialization.endpoints import CachedEndpointStore, ONLINE_TRAINING_ENDPOINT_PROTOCOL
from tricoopdiff.sdpi_flow import SDPIFlow


FULL = ConditionFeatureDropout(0., 0., 0., 0., 0., 0.)
POCKET_ONLY = ConditionFeatureDropout(0., 0., 0., 1., 1., 0.)
ABSENT = ConditionFeatureDropout(1., 0., 0., 0., 0., 0.)
CHECKPOINT_SELECTION = (
    'mean of six SDPI/BiTSeed x full/pocket-only/absent validation cells')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, help='C checkpoint used for EMA initialization')
    parser.add_argument('--train-data', default='data/ternary_train.json')
    parser.add_argument('--val-data', default='data/ternary_val.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--endpoint-cache', required=True,
                        help='Historical projected endpoint archive; no online legacy generation')
    parser.add_argument('--paired-cache-follower', action='store_true',
                        help='Never generate shared features/endpoints; wait for the writer arm')
    parser.add_argument('--paired-cache-wait-seconds', type=float, default=3600.)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--max-train-clusters', type=int)
    parser.add_argument('--max-val-clusters', type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--condition-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--residual-weight', type=float, default=1e-3)
    parser.add_argument('--ema-decay', type=float, default=.999)
    parser.add_argument('--nan-clamp', type=float, default=100.)
    parser.add_argument('--condition-drop-all', type=float, default=.1)
    parser.add_argument('--condition-drop-side', type=float, default=.2)
    parser.add_argument('--condition-drop-pocket', type=float, default=.1)
    parser.add_argument('--condition-drop-ligand-role', type=float, default=.1)
    parser.add_argument('--condition-drop-binary', type=float, default=.1)
    parser.add_argument('--condition-pocket-only', type=float, default=.2,
                        help='Probability of retaining only both pocket masks')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--resume')
    parser.add_argument('--allow-resume-dropout-change', action='store_true',
                        help='Explicitly replace only the augmentation policy on resume')
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.save_interval < 1 or args.log_interval < 1
            or args.nan_clamp <= 0 or args.paired_cache_wait_seconds < 0):
        parser.error('epochs/intervals and --nan-clamp must be positive')
    try:
        condition_dropout(args).validate()
    except ValueError as error:
        parser.error(str(error))
    return args


def condition_dropout(args):
    return ConditionFeatureDropout(
        args.condition_drop_all, args.condition_drop_side,
        args.condition_drop_pocket, args.condition_drop_ligand_role,
        args.condition_drop_binary, args.condition_pocket_only)


def build_trainer(args, device):
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    validate_assembly_checkpoint(parent)
    expert = build_diffdock_expert(args.expert_model_dir, device=str(device))
    model = build_conditioned_model(parent, expert)
    trainer = ConditionedConsensusTrainer(
        model, SDPIFlow(inference_steps=20), device=device, lr=args.lr,
        weight_decay=args.weight_decay, activation_clamp=args.nan_clamp,
        activation_clamp_mode=parent['runtime_config']['activation_clamp_mode'],
        residual_weight=args.residual_weight, ema_decay=args.ema_decay,
        training_source_mix='sdpi_bitseed_1_1',
        training_endpoint_protocol=ONLINE_TRAINING_ENDPOINT_PROTOCOL,
        parent_checkpoint_sha256=checkpoint_sha256(args.parent),
        condition_dropout=condition_dropout(args),
        condition_lr=args.condition_lr,
    )
    return model, trainer, parent


def condition_seed(case, base):
    """Stable per-case seed; Python's randomized hash is deliberately avoided."""
    import hashlib
    digest = hashlib.sha256(f'{base}:{case}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2**63)


@torch.no_grad()
def validate_grid(trainer, loader, endpoint_store, device, seed):
    """Evaluate the predeclared source x condition-availability grid."""
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == 'cuda' else [])
    numpy_state = np.random.get_state()
    cells = {f'{source}/{mode}': [] for source in ('sdpi', 'bitseed_endpoint')
             for mode in ('full', 'pocket_only', 'absent')}
    fallback = 0
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            for batch in loader:
                full, _ = supervised_condition_batch(
                    batch, seed=condition_seed(batch['case'], seed), dropout=FULL)
                pocket_only, _ = dropout_condition_features(
                    full, seed=0, config=POCKET_ONLY)
                absent, _ = dropout_condition_features(full, seed=0, config=ABSENT)
                coupling, record = endpoint_store.coupling(
                    batch['case'], **({'batch': batch} if endpoint_store.lazy else {}))
                endpoint_source = record['actual_source']
                fallback += int(endpoint_source == 'sdpi')
                t = torch.rand(1, device=device)
                sdpi_coupling = trainer.diffusion.sample_coupling(
                    batch['e3_data']['coords'], batch['poi_data']['coords'],
                    batch['ligand_data'],
                    no_torsion=bool(getattr(trainer.model.expert.model, 'no_torsion', False)))
                source_rows = [
                    ('sdpi', {**batch, 'source': 0, 'coupling_override': sdpi_coupling,
                              't_override': t}),
                    ('bitseed_endpoint', {
                        **batch, 'source': int(endpoint_source == 'bitseed'),
                        'coupling_override': coupling, 't_override': t,
                    }),
                ]
                for source, row in source_rows:
                    for mode, conditions in (('full', full),
                                             ('pocket_only', pocket_only),
                                             ('absent', absent)):
                        result = trainer.validate_step(**row, conditions=conditions)
                        loss = result['loss_total']
                        if not np.isfinite(loss):
                            raise RuntimeError(f'non-finite validation: {batch["case"]}/{source}/{mode}')
                        cells[f'{source}/{mode}'].append(float(loss))
    finally:
        np.random.set_state(numpy_state)
        trainer.model.train()
    means = {name: float(np.mean(values)) for name, values in cells.items() if values}
    required = {
        'sdpi/full', 'sdpi/pocket_only', 'sdpi/absent',
        'bitseed_endpoint/full', 'bitseed_endpoint/pocket_only',
        'bitseed_endpoint/absent',
    }
    if set(means) != required:
        raise RuntimeError(f'validation grid incomplete: present={sorted(means)}')
    return {'selection_loss': float(np.mean(list(means.values()))), 'by_cell': means,
            'counts': {name: len(values) for name, values in cells.items()},
            'bitseed_fallback_count': fallback}


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    seed_training(args.seed, device)
    os.chdir(ROOT)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    endpoint_cache = Path(args.endpoint_cache).resolve()
    config = {**vars(args), 'parent': str(Path(args.parent).resolve()),
              'parent_checkpoint_sha256': checkpoint_sha256(args.parent),
              'training_source_mix': 'sdpi_bitseed_1_1',
              'training_endpoint_protocol': ONLINE_TRAINING_ENDPOINT_PROTOCOL,
              'condition_supervision': 'native discrete contact masks at 6 A',
              'condition_dropout': vars(condition_dropout(args)),
              'checkpoint_selection': CHECKPOINT_SELECTION}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    # Warm/validate featurization before the expensive expert is built.
    entries = _load_entries(args.train_data)
    if not entries:
        raise RuntimeError('empty training index')
    score_args = _load_score_model_args(args.expert_model_dir)
    first = entries[0]
    featurize_complex(first['case'], first['protein1'], first['protein2'], first['ligand'],
                      score_args, str(device), cache_dir=args.cache_dir,
                      cache_read_only=args.paired_cache_follower,
                      cache_wait_seconds=args.paired_cache_wait_seconds)
    seed_training(args.seed, device)
    model, trainer, _ = build_trainer(args, device)
    endpoint_store = CachedEndpointStore(endpoint_cache)
    train_loader = TernaryDataLoader(
        args.train_data, score_args, shuffle=True, device=str(device), cache_dir=args.cache_dir,
        max_clusters=args.max_train_clusters, random_flip_proteins=False,
        cache_read_only=args.paired_cache_follower,
        cache_wait_seconds=args.paired_cache_wait_seconds)
    val_loader = TernaryDataLoader(
        args.val_data, score_args, shuffle=False, device=str(device), cache_dir=args.cache_dir,
        max_clusters=args.max_val_clusters, random_flip_proteins=False,
        cache_read_only=args.paired_cache_follower,
        cache_wait_seconds=args.paired_cache_wait_seconds)
    if args.resume:
        trainer.load_checkpoint(
            args.resume, allow_dropout_change=args.allow_resume_dropout_change)
    start_epoch, best = trainer.resume_epoch + 1, trainer.resume_best_val_loss
    if start_epoch > args.epochs:
        raise RuntimeError('resume epoch meets or exceeds target epochs')
    if args.resume:
        # A new formal output directory must remain self-contained even when
        # the resumed epoch stays best for the rest of training.
        trainer.save_checkpoint(output / 'best_model.pt', epoch=trainer.resume_epoch,
                                best_val_loss=best)

    for epoch in range(start_epoch, args.epochs + 1):
        started, losses = time.time(), []
        for batch_index, raw_batch in enumerate(train_loader):
            conditions, audit = trainer.condition_batch(raw_batch, seed=condition_seed(
                raw_batch['case'], args.seed + trainer.global_step))
            batch = prepare_source_batch(raw_batch, trainer.global_step, 'C', endpoint_store)
            result = _guarded_train_batch(trainer, [{**batch, 'conditions': conditions}])[0]
            losses.append(result['loss_total'])
            append_metric(output, {'event': 'train_step', 'epoch': epoch,
                                   'global_step': trainer.global_step, 'case': batch['case'],
                                   'source': 'bitseed' if batch['source'] else 'sdpi',
                                   'condition_events': audit['events'], **result})
            if batch_index % args.log_interval == 0:
                print(f'epoch={epoch} batch={batch_index} loss={result["loss_total"]:.4f}', flush=True)
        if not losses:
            raise RuntimeError('training loader produced no batches')
        validation = validate_grid(trainer, val_loader, endpoint_store, device, args.seed)
        validation['raw_condition_scale'] = float(
            trainer.model.trifuse.condition_scale.detach().cpu())
        validation['ema_condition_scale'] = float(
            trainer.ema['condition_scale'].detach().cpu())
        trainer.scheduler_step(validation['selection_loss'])
        append_metric(output, {'event': 'validation', 'epoch': epoch,
                               'global_step': trainer.global_step, **validation})
        print(f'epoch={epoch} train={np.mean(losses):.4f} val={validation["selection_loss"]:.4f} '
              f'condition_scale={validation["raw_condition_scale"]:.6g}/ema={validation["ema_condition_scale"]:.6g} '
              f'cells={validation["by_cell"]} seconds={time.time()-started:.1f}', flush=True)
        if validation['selection_loss'] < best:
            best = validation['selection_loss']
            trainer.save_checkpoint(output / 'best_model.pt', epoch=epoch, best_val_loss=best)
        if epoch % args.save_interval == 0:
            trainer.save_checkpoint(output / f'checkpoint_epoch{epoch}.pt', epoch=epoch,
                                    best_val_loss=best)
    trainer.save_checkpoint(output / 'final_model.pt', epoch=args.epochs, best_val_loss=best)


if __name__ == '__main__':
    main()
