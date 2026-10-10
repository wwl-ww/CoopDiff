#!/usr/bin/env python
"""Coupled predicted-center experiment using the unchanged production trainer."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import random
import numpy as np
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
from tricoopdiff.models.expert import (build_diffdock_expert, configure_expert_policy,
                                        expert_runtime_identity)
from tricoopdiff.initialization.candidates import (DualExpertAnchorInitializer,
    coupled_anchor_perturbation)
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.initialization.endpoints import reachable_ligand_target
from tricoopdiff.metrics.dockq import compute_fullatom_dockq


def options():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Additional production train_entry options are accepted, e.g. --lr, '
        '--native-root, --cache_dir and --device. No --resume or weight transfer.'))
    base = 'results/small_sample_seed73'
    parser.add_argument('--train-data', default=f'{base}/train.json')
    parser.add_argument('--same-complex-val', default=f'{base}/same_complex_val.json')
    parser.add_argument('--unseen-complex-val', default=f'{base}/unseen_complex_val.json')
    parser.add_argument('--output-dir', default=f'{base}/run')
    parser.add_argument('--binary-steps', type=int, default=20)
    parser.add_argument('--torsion-std', type=float, default=.15)
    parser.add_argument('--translation-std', type=float, default=.5)
    parser.add_argument('--rotation-std', type=float, default=.05)
    parser.add_argument('--max-updates', type=int, default=1)
    parser.add_argument('--eval-updates', default='0,1')
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


def center_hash(center):
    digest = hashlib.sha256()
    for key in ('ligand', 'poi', 'rotation'):
        digest.update(center[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@preserve_rng
def generate_center(batch, expert, seed, binary_steps):
    """Generate one actual BiTSeed center; failures abort rather than masquerade as E."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)
    ed, pd, ld = (batch[key] for key in ('e3_data', 'poi_data', 'ligand_data'))
    graph_only = {key: value for key, value in ld.items() if key != 'coords'}
    initializer = DualExpertAnchorInitializer(expert, ed, pd,
        binary_steps=binary_steps, proposal_attempts=1)
    candidates, generation, _ = initializer.generate_candidates(
        ed['coords'][0], pd['coords'][0], graph_only, seed=seed,
        conformer_count=1, candidate_budget=1, binary_retry_limit=8,
        binary_consecutive_failure_limit=8)
    if len(candidates) != 1:
        raise RuntimeError(f"{batch['case']}: no genuine BiTSeed center")
    center = cpu_tree(candidates[0])
    print(json.dumps({'event':'center_generated', 'case':batch['case'],
                      'seed':seed, 'sha256':center_hash(center)}), flush=True)
    return {'center': center, 'seed': seed, 'sha256': center_hash(center),
            'generation': generation, 'backend': 'frozen_diffdock_v1.1',
            'native_information_used': False}


def perturb(batch, center_record, seed, opt):
    ed, pd, ld = (batch[key] for key in ('e3_data', 'poi_data', 'ligand_data'))
    center = {key: value.to(ed['coords']) if torch.is_tensor(value) else value
              for key, value in center_record['center'].items()}
    state, provenance = coupled_anchor_perturbation(center, ed['coords'][0],
        {key: value for key, value in ld.items() if key != 'coords'}, seed=seed,
        torsion_std=opt.torsion_std, translation_std=opt.translation_std,
        rotation_std=opt.rotation_std)
    coupling = {'initial_state': {'ligand':state['ligand'], 'poi':state['poi'],
                                  'body_frame':state['rotation']}}
    return coupling, provenance


@preserve_rng
def make_bank(batch, seed, context, center_record, opt):
    ed, pd, ld = (batch[key] for key in ('e3_data', 'poi_data', 'ligand_data'))
    coupling, provenance = perturb(batch, center_record, seed, opt)
    state = coupling['initial_state']
    ligand, poi = state['ligand'], state['poi']
    target, projection = reachable_ligand_target(ld, ligand, poi, pd['coords'], no_torsion=False)
    coupling['target_ligand'] = torch.as_tensor(target)
    initial = compute_fullatom_dockq(ligand.reshape(-1,3).cpu().numpy(),
        poi.reshape(-1,3).cpu().numpy(), ed['coords'][0].cpu().numpy(), **context)
    center = center_record['center']
    center_metric = compute_fullatom_dockq(center['ligand'].reshape(-1,3).numpy(),
        center['poi'].reshape(-1,3).numpy(), ed['coords'][0].cpu().numpy(), **context)
    return {'seed':seed, 'coupling':cpu_tree(coupling), 'projection':projection,
            'initialization_provenance':provenance, 'initial_dockq':initial,
            'center_dockq':center_metric, 'center_sha256':center_record['sha256']}


def evaluate(trainer, loaders, banks, args, output):
    """Paired raw/EMA use the exact same persisted initial states and atom mapping."""
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
                    record = {'case': batch['case'], 'initial_seed': bank['seed'],
                        'initial_dockq': bank['initial_dockq']['dockq'], **metric,
                        'center_dockq': bank['center_dockq']['dockq'],
                        'center_sha256': bank['center_sha256'],
                        'initialization_mode': bank['initialization_provenance']['mode'],
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
                'mode_summaries': {mode:{'count':sum(row['initialization_mode']==mode for row in records),
                    'dockq_mean':float(np.mean([row['dockq'] for row in records if row['initialization_mode']==mode])),
                    'initial_dockq_mean':float(np.mean([row['initial_dockq'] for row in records if row['initialization_mode']==mode]))}
                    for mode in sorted({row['initialization_mode'] for row in records})},
                'initial_dockq_mean':float(np.mean([row['initial_dockq'] for row in records])),
                'center_dockq_mean':float(np.mean([row['center_dockq'] for row in records])),
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
                       'unseen_complex': opt.unseen_complex_val,
                       'independent_center': opt.same_complex_val}
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
            ('same_complex', opt.same_complex_val), ('unseen_complex', opt.unseen_complex_val),
            ('independent_center', opt.same_complex_val))}
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
        # Two train centers per case reproduce E's center-count scope. New-center
        # validation has separate deterministic generation seeds and no quality selection.
        center_expert = build_diffdock_expert(args.expert_model_dir, device=str(device))
        configure_expert_policy(center_expert, args.nan_clamp, args.activation_clamp_mode)
        center_expert_runtime = expert_runtime_identity(center_expert)
        centers = {'train':{}, 'independent_center':{}, 'unseen_complex':{}}
        with evaluation_context(center_expert):
            for batch in loaders['same_complex']:
                case = batch['case']
                centers['train'][case] = [generate_center(batch, center_expert,
                    case_seed(f'{case}:train_center:{index}', 7300), opt.binary_steps)
                    for index in range(2)]
                independent = generate_center(batch, center_expert,
                    case_seed(f'{case}:independent_center', 7400), opt.binary_steps)
                if independent['sha256'] in {record['sha256'] for record in centers['train'][case]}:
                    raise ValueError(f'{case}: independent center exactly duplicates training')
                distances = []
                for train_center in centers['train'][case]:
                    left, right = independent['center'], train_center['center']
                    distances.append({key:float((left[key]-right[key]).square().sum(-1).mean().sqrt())
                                      for key in ('ligand','poi')})
                independent['distance_to_training_centers_angstrom'] = distances
                centers['independent_center'][case] = independent
            for batch in loaders['unseen_complex']:
                centers['unseen_complex'][batch['case']] = generate_center(batch, center_expert,
                    case_seed(f"{batch['case']}:unseen_center", 7600), opt.binary_steps)
        atomic_torch_save(centers, output/'predicted_centers.pt')
        del center_expert
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        banks = {}
        with evaluation_context(model):
            for group, loader in loaders.items():
                banks[group] = {}
                for batch in loader:
                    case = batch['case']
                    center_record = (centers['train'][case][0] if group == 'same_complex'
                                     else centers[group][case])
                    seed = case_seed(f'{case}:{group}:heldout_perturbation', 7500)
                    context = scoring_context(loader.entries_by_case[case], args)
                    banks[group][case] = make_bank(batch, seed, context, center_record, opt)
        atomic_torch_save(banks, output/'validation_starts.pt')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        physical_gpu = None
        if device.type == 'cuda':
            physical_gpu = visible.split(',')[device.index or 0] if visible else device.index or 0
        manifest = {'fresh': True, 'loaded_ternary_checkpoint': None, 'model_seed': opt.model_seed,
            'train_initialization': 'online coupled perturbations around two frozen DiffDock predicted centers per case',
            'coverage': coverage, 'device': str(device),
            'cuda_visible_devices': visible, 'physical_gpu': physical_gpu,
            'cuda_device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'protein_flip': False, 'evaluation_seed_bases': {'perturbations':7500, 'train_centers':7300, 'independent_centers':7400, 'unseen_centers':7600},
            'center_bank_sha256': sha256_file(output/'predicted_centers.pt'),
            'center_backend': 'frozen DiffDock v1.1 with production clamp; ternary expert frozen Matcha',
            'center_expert_runtime': center_expert_runtime,
            'same_center_validation': 'train center 0, independent perturbation seed',
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
        occurrences = {}
        while trainer.global_step < opt.max_updates:
            epoch += 1
            for batches in iter_training_batches(train_loader,args.batch_size):
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                update_started = time.monotonic()
                prepared = []
                initialization = []
                for batch in batches:
                    case = batch['case']
                    occurrence = occurrences.get(case, 0)
                    occurrences[case] = occurrence + 1
                    center_record = centers['train'][case][occurrence % 2]
                    seed = case_seed(f'{case}:training_perturbation:{occurrence}', 7301)
                    coupling, provenance = perturb(batch, center_record, seed, opt)
                    prepared.append({**batch, 'source':0, 'coupling_override':coupling})
                    initialization.append({'case':case, 'occurrence':occurrence,
                        'center_sha256':center_record['sha256'], **provenance})
                metrics = _guarded_train_batch(trainer, prepared,
                    failure_path=output/'projection_failure.pt')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                update_seconds = time.monotonic() - update_started
                peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
                if peak is not None:
                    run_peak = max(run_peak, peak)
                append_metric(output,{'event':'train_step','epoch':epoch,'global_step':trainer.global_step,
                    'cases':[batch['case'] for batch in batches], 'microbatch_metrics':metrics, 'initialization':initialization,
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
