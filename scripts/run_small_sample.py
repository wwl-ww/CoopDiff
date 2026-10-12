#!/usr/bin/env python
"""Fresh small-sample endpoint experiment using the production trainer unchanged."""
import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

import tricoopdiff.diffdock_compat
from train_entry import (parse_args, build_trainer, TernaryDataLoader,
                         _load_score_model_args, _load_entries, seed_training,
                         iter_training_batches, _guarded_train_batch,
                         validation_metric_context, append_metric, case_seed)
from tricoopdiff.runtime import (evaluation_context, preserve_rng, atomic_torch_save,
                                sha256_file)
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.initialization.endpoints import reachable_ligand_target
from tricoopdiff.metrics.dockq import compute_fullatom_dockq
from native_contact_evaluation import native_contact_preflight, contact_strata


def options():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Additional production train_entry options are accepted, e.g. --lr, '
        '--native-root, --cache_dir and --device. No --resume or weight transfer.'))
    base = 'results/small_sample_seed73'
    parser.add_argument('--train-data', default=f'{base}/train.json')
    parser.add_argument('--same-complex-val', default=f'{base}/same_complex_val.json')
    parser.add_argument('--unseen-complex-val', default=f'{base}/unseen_complex_val.json')
    parser.add_argument('--output-dir', default=f'{base}/run')
    parser.add_argument('--max-updates', type=int, default=400)
    parser.add_argument('--eval-updates', default='0,50,100,200,400')
    parser.add_argument('--model-seed', type=int, default=73)
    parser.add_argument('--memory-fraction', type=float, default=.25)
    parser.add_argument('--deepternary-root', default=str(ROOT / 'DeepTernary'),
                        help='DeepTernary source directory with executable DockQ/fnat')
    opt, extra = parser.parse_known_args()
    if opt.max_updates < 0 or not 0 < opt.memory_fraction <= 1:
        parser.error('max-updates must be nonnegative and memory-fraction in (0,1]')
    marks = {int(value) for value in opt.eval_updates.split(',')}
    if any(value < 0 for value in marks):
        parser.error('eval-updates must be nonnegative')
    opt.eval_updates = sorted(({0, opt.max_updates} | marks) & set(range(opt.max_updates + 1)))
    args = parse_args(['--train_data', opt.train_data, '--val_data', opt.same_complex_val,
        '--output_dir', opt.output_dir, '--expert-backend', 'matcha', '--training-arm', 'A1',
        '--seed', str(opt.model_seed), '--batch_size', '4', '--lr', '.0001',
        '--no-random-flip-proteins', *extra])
    if args.resume or args.expert_backend != 'matcha' or args.training_arm != 'A1':
        parser.error('this experiment requires fresh Matcha A1; no resume/weight transfer')
    if args.seed != opt.model_seed or args.random_flip_proteins:
        parser.error('model seed must match --model-seed; protein flipping is disabled')
    if args.max_train_clusters or args.max_val_clusters:
        parser.error('use explicit experiment indexes, not silent cluster truncation')
    args.training_endpoint_protocol = 'none'
    args.generative_process = 'sdpi_flow_v1'
    args.deepternary_root = opt.deepternary_root
    return opt, args


def scoring_context(entry, args):
    context = validation_metric_context(entry, args.native_root)
    context['deepternary_root'] = str(Path(args.deepternary_root).resolve())
    return context


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    return value


@preserve_rng
def make_bank(batch, seed, context):
    """Persist actual starts and reachable targets; evaluation never samples anew."""
    ed, pd, ld = (batch[key] for key in ('e3_data', 'poi_data', 'ligand_data'))
    ligand, poi, body, provenance = TernaryInitializer(no_torsion=False).initialize(
        ed['coords'][0], pd['coords'][0], ld, seed=seed)
    target, projection = reachable_ligand_target(ld, ligand, poi, pd['coords'], no_torsion=False)
    coupling = cpu_tree({'initial_state': {'ligand': ligand, 'poi': poi, 'body_frame': body},
                         'target_ligand': torch.as_tensor(target)})
    initial = compute_fullatom_dockq(ligand.reshape(-1, 3).cpu().numpy(),
        poi.reshape(-1, 3).cpu().numpy(), ed['coords'][0].cpu().numpy(), **context)
    return {'seed': seed, 'coupling': coupling, 'projection': projection,
            'initialization_provenance': provenance, 'initial_dockq': initial}


def evaluate(trainer, loaders, banks, args, output):
    """Paired raw/EMA use the exact same persisted initial states and atom mapping."""
    contact_audit = {}
    for loader in loaders.values():
        for case, entry in loader.entries_by_case.items():
            if case not in contact_audit:
                contact_audit[case] = native_contact_preflight(scoring_context(entry, args))
    for group, loader in loaders.items():
        for weights in ('raw', 'ema'):
            records = []
            with evaluation_context(trainer.model), (trainer.ema_parameters() if weights == 'ema' else nullcontext()):
                for batch in loader:
                    bank = banks[group][batch['case']]
                    context = scoring_context(loader.entries_by_case[batch['case']], args)
                    result = trainer.compute_loss(**batch, source=0,
                        coupling_override=bank['coupling'], diagnostics=True)
                    ligand = result['predicted_ligand']
                    native = batch['ligand_data']['coords']
                    target = result['target_ligand']
                    metric = compute_fullatom_dockq(ligand[0].cpu().numpy(),
                        result['predicted_poi'][0].cpu().numpy(),
                        batch['e3_data']['coords'][0].cpu().numpy(), **context)
                    record = {'case': batch['case'], **contact_audit[batch['case']],
                        'initial_seed': bank['seed'],
                        'initial_dockq': bank['initial_dockq']['dockq'], **metric,
                        'loss_total': result['loss_total'].item(),
                        'loss_endpoint_main': result['loss_endpoint_main'].item(),
                        'ligand_target_rmsd_angstrom': result['ligand_target_rmsd_angstrom'].item(),
                        'ligand_crystal_rmsd_angstrom': (ligand-native).square().sum(-1).mean().sqrt().item(),
                        'projection_residual_rmsd_angstrom': (target-native).square().sum(-1).mean().sqrt().item(),
                        'loss_components': {key:value.item() for key,value in result['components'].items()}}
                    records.append(record)
            values = torch.tensor([record['dockq'] for record in records], dtype=torch.float64)
            payload = {'event': 'evaluation', 'global_step': trainer.global_step,
                'group': group, 'weights': weights, 'count': len(records),
                'dockq_mean': values.mean().item(),
                'dockq_median': values.quantile(.5).item(),
                'dockq_above_023': int((values > .23).sum()),
                'loss_mean': sum(row['loss_total'] for row in records) / len(records),
                'ligand_crystal_rmsd_mean': sum(row['ligand_crystal_rmsd_angstrom'] for row in records) / len(records),
                'ligand_reachable_rmsd_mean': sum(row['ligand_target_rmsd_angstrom'] for row in records) / len(records),
                **contact_strata(records),
                'per_case': records}
            append_metric(output, payload)
            (Path(output)/f"eval_update{trainer.global_step}_{group}_{weights}.json").write_text(json.dumps(payload, indent=2))
            print(json.dumps({key: value for key, value in payload.items() if key != 'per_case'}), flush=True)


def main():
    opt, args = options()
    os.chdir(ROOT)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # An OS lock automatically releases on crash; never steal another writer's lock.
    with (output/'writer.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'output already has an active writer: {output}') from error
        if (output/'manifest.json').exists():
            raise RuntimeError('fresh experiment requires a new output directory; manifest already exists')
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        index_paths = {'train': args.train_data, 'same_complex': opt.same_complex_val,
                       'unseen_complex': opt.unseen_complex_val}
        fnat = Path(args.deepternary_root) / 'DockQ/fnat'
        if not fnat.is_file() or not os.access(fnat, os.X_OK):
            raise ValueError(f'DockQ requires an executable fnat; check --deepternary-root: {fnat}')
        indexed = {name: _load_entries(path) for name, path in index_paths.items()}
        for name, entries in indexed.items():
            if not entries or any(not entry.get('cluster_id') for entry in entries):
                raise ValueError(f'{name}: every indexed case must have a cluster_id')
        if args.cached_only:
            missing = sorted({entry['case'] for entries in indexed.values() for entry in entries
                              if not (Path(args.cache_dir) / (entry['case'] + '.pt')).is_file()})
            if missing:
                raise FileNotFoundError(f'cached-only requires every indexed case; missing caches: {missing}')
        training_membership = {entry['case']: entry['cluster_id'] for entry in indexed['train']}
        same_membership = {entry['case']: entry['cluster_id'] for entry in indexed['same_complex']}
        if training_membership != same_membership:
            raise ValueError('same-complex validation case/cluster membership differs from training')
        train_clusters = {entry['cluster_id'] for entry in indexed['train']}
        unseen_clusters = {entry['cluster_id'] for entry in indexed['unseen_complex']}
        if train_clusters & unseen_clusters:
            raise ValueError('unseen-complex clusters overlap training')
        device = torch.device(args.device)
        if device.type == 'cuda':
            if device.index is None:
                device = torch.device('cuda', torch.cuda.current_device())
            torch.cuda.set_per_process_memory_fraction(opt.memory_fraction, device)
        seed_training(opt.model_seed, device)
        score_args = _load_score_model_args(args.expert_model_dir)
        # Load cached feature shape before resetting the fresh model RNG, as production does.
        probe = TernaryDataLoader(args.train_data, score_args, shuffle=False,
            device=str(device), cache_dir=args.cache_dir)
        first = next(iter(probe))
        input_dim = int(first['e3_data']['features'].shape[-1])
        del first, probe
        seed_training(opt.model_seed, device)
        model, _, trainer = build_trainer(args, device, input_dim=input_dim)
        model.to(device)
        trainer.set_phase(args.phase)
        train_loader = TernaryDataLoader(args.train_data, score_args, shuffle=True,
            device=str(device), cache_dir=args.cache_dir, expert=model.expert,
            random_flip_proteins=False, cached_only=args.cached_only)
        loaders = {name: TernaryDataLoader(path, score_args, shuffle=False,
            device=str(device), cache_dir=args.cache_dir, expert=model.expert,
            cached_only=args.cached_only) for name, path in (
            ('same_complex', opt.same_complex_val), ('unseen_complex', opt.unseen_complex_val))}
        actual_loaders = {'train': train_loader, **loaders}
        for name, loader in actual_loaders.items():
            if len(loader.entries) != len(indexed[name]):
                raise ValueError(f'{name}: cache filtering changed indexed population; prepare all caches first')
        coverage = {name: {'case_count': len(loader.entries),
            'cases': sorted(entry['case'] for entry in loader.entries),
            'cluster_count': len({entry['cluster_id'] for entry in loader.entries}),
            'clusters': sorted({entry['cluster_id'] for entry in loader.entries}),
            'case_to_cluster': {entry['case']: entry['cluster_id'] for entry in loader.entries}}
            for name, loader in actual_loaders.items()}
        train_cases = {entry['case'] for entry in train_loader.entries}
        if train_cases != set(loaders['same_complex'].entries_by_case):
            raise ValueError('same-complex validation must contain exactly the training cases')
        if train_cases & set(loaders['unseen_complex'].entries_by_case):
            raise ValueError('unseen-complex cases overlap training')
        if any(len(loader) == 0 for loader in [train_loader, *loaders.values()]):
            raise ValueError('empty experiment group')
        banks = {}
        with evaluation_context(model):
            for group, loader in loaders.items():
                banks[group] = {}
                for batch in loader:
                    seed = case_seed(batch['case'], 75 if group == 'same_complex' else 76)
                    context = scoring_context(loader.entries_by_case[batch['case']], args)
                    banks[group][batch['case']] = make_bank(batch, seed, context)
        atomic_torch_save(banks, output/'validation_starts.pt')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        physical_gpu = None
        if device.type == 'cuda':
            physical_gpu = visible.split(',')[device.index or 0] if visible else device.index or 0
        manifest = {'fresh': True, 'loaded_ternary_checkpoint': None, 'model_seed': opt.model_seed,
            'train_initialization': f'online SDPI, persistent seed{opt.model_seed} RNG; validation banks fixed independently',
            'coverage': coverage, 'device': str(device),
            'cuda_visible_devices': visible, 'physical_gpu': physical_gpu,
            'cuda_device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'protein_flip': False, 'evaluation_seed_bases': {'same_complex':75, 'unseen_complex':76},
            'validation_bank_sha256': sha256_file(output/'validation_starts.pt'),
            'indexes': {name: {'path':str(Path(path).resolve()),'sha256':sha256_file(path)} for name,path in (
                ('train',args.train_data),('same_complex',opt.same_complex_val),('unseen_complex',opt.unseen_complex_val))},
            'expert_provenance': model.expert.provenance, 'model_config':args.resolved_model_config,
            'scheduler_policy':'fixed learning rate; ReduceLROnPlateau is not advanced by this update-budget experiment',
            'script_sha256':sha256_file(__file__), 'options':vars(opt)}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (output/'config.json').write_text(json.dumps(vars(args),indent=2))
        started = time.monotonic()
        trainer.save_checkpoint(output/'checkpoint_update0.pt')
        evaluate(trainer,loaders,banks,args,output)
        run_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        epoch = 0
        while trainer.global_step < opt.max_updates:
            epoch += 1
            for batches in iter_training_batches(train_loader,args.batch_size):
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                update_started = time.monotonic()
                metrics = _guarded_train_batch(trainer,[{**batch,'source':0} for batch in batches],
                    failure_path=output/'projection_failure.pt')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                update_seconds = time.monotonic() - update_started
                peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
                if peak is not None:
                    run_peak = max(run_peak, peak)
                append_metric(output,{'event':'train_step','epoch':epoch,'global_step':trainer.global_step,
                    'cases':[batch['case'] for batch in batches], 'microbatch_metrics':metrics,
                    'elapsed_seconds':update_seconds, 'peak_cuda_allocated_bytes':peak,
                    'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None})
                trainer.save_checkpoint(output/'progress_model.pt',epoch=epoch-1)
                if trainer.global_step in opt.eval_updates:
                    trainer.save_checkpoint(output/f'checkpoint_update{trainer.global_step}.pt',epoch=epoch-1)
                    evaluate(trainer,loaders,banks,args,output)
                    if device.type == 'cuda':
                        run_peak = max(run_peak, torch.cuda.max_memory_allocated(device))
                if trainer.global_step >= opt.max_updates:
                    break
        complete = {'event':'complete','global_step':trainer.global_step,
            'elapsed_seconds':time.monotonic()-started, 'peak_cuda_allocated_bytes':run_peak}
        append_metric(output,complete)
        print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
