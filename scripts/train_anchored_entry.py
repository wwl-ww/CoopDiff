"""Bounded foreground training of the complete two-sided conditional chart.

Uses an explicit update budget and a chemistry-restored, benchmark-excluded
training index. EMA selection uses fixed internal-development validation loss,
never benchmark structures or oracle scores.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import time as wall_clock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import tricoopdiff.diffdock_compat  # noqa: F401
from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from train_entry import _load_index
from tricoopdiff.anchored_trainer import (AnchoredTrainer, validate_anchored_checkpoint,
                                        warm_start_anchored_weights)
from tricoopdiff.anchored_training import prepare_augmented_anchored_sample
from tricoopdiff.anchored_tasks import prepare_task_sample
from tricoopdiff.chemical_prior_contract import ChemicalPriorLibrary, chemical_prior_config, apply_training_prior
from tricoopdiff.anchored_data_schedule import epoch_entries, INDEX_ORDER, CLUSTER_ORDER
from tricoopdiff.anchored_training_metric import PARAMETER_METRIC, PHYSICAL_METRIC
from tricoopdiff.anchored_action_adapter import install_action_adapter
from tricoopdiff.anchored_validation import (VALIDATION_POLICY, VALIDATION_TIMES,
                                           evaluate_panel, validate_index_pair)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'parent', 'index', 'cache-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-updates', type=int, required=True,
                        help='Number of new updates; zero means no update limit')
    parser.add_argument('--max-seconds', type=float,
                        help='Graceful wall-clock budget; finish the current update and save before exiting')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--warm-start', type=Path,
                        help='Explicit trained EMA weight transfer; new optimizer, validation and task clock')
    parser.add_argument('--reset-action-adapter', action='store_true',
                        help='Explicitly replace old stacked action heads; preserve the shared backbone')
    parser.add_argument('--resume-best', type=Path)
    parser.add_argument('--val-index', type=Path)
    parser.add_argument('--val-cache-dir', type=Path)
    parser.add_argument('--scope', choices=('training', 'implementation_smoke'), required=True)
    parser.add_argument('--binary-translation-bound', type=float)
    parser.add_argument('--binary-rotation-bound', type=float, help='Radians, total deviation from each template')
    parser.add_argument('--conditioning-task', choices=('dual_binary', 'dual_core', 'dual_contact_core', 'dual_assembly', 'single_anchor'), default='dual_binary')
    parser.add_argument('--prior-joint-action', action='store_true',
                        help='Replace the action module with binary-evidence joint body/torsion attention')
    parser.add_argument('--chemical-prior-manifest', type=Path)
    parser.add_argument('--val-chemical-prior-manifest', type=Path)
    parser.add_argument('--chemical-prior-count', type=int, default=16)
    parser.add_argument('--uniform-torsion-probability', type=float, default=0.)
    parser.add_argument('--binary-metric', choices=(PARAMETER_METRIC, PHYSICAL_METRIC), default=PARAMETER_METRIC)
    parser.add_argument('--data-schedule', choices=(INDEX_ORDER, CLUSTER_ORDER), default=INDEX_ORDER)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--body-aware-action', action='store_true',
                        help='Install the trainable kinematics-aware torsion residual with explicit checkpoint migration')
    parser.add_argument('--coupled-linker-action', action='store_true',
                        help='Install periodic and cross-bond linker action attention (checkpoint v7)')
    args = parser.parse_args()
    started = wall_clock.monotonic()
    if args.max_seconds is not None and not 0 < args.max_seconds < float('inf'):
        parser.error('max-seconds must be positive and finite')
    if args.reset_action_adapter and (not args.warm_start or args.resume
                                     or not (args.coupled_linker_action or args.prior_joint_action)):
        parser.error('action replacement requires a coupled warm-start without resume')
    if sum((args.body_aware_action, args.coupled_linker_action, args.prior_joint_action)) > 1:
        parser.error('select one action adapter; coupled linker includes body-aware actions')
    if (args.binary_translation_bound is None) != (args.binary_rotation_bound is None):
        parser.error('provide both binary translation and rotation bounds')
    bounds = (None if args.binary_translation_bound is None else
              {'translation': [args.binary_translation_bound] * 2,
               'rotation': [args.binary_rotation_bound] * 2})
    if args.max_updates < 0 or min(args.save_every, args.log_every) < 1 or args.seed < 0:
        parser.error('update budget and seed must be nonnegative; intervals must be positive')
    if args.out.exists():
        raise FileExistsError('training/resume must publish into a new run directory')
    if bool(args.val_index) != bool(args.val_cache_dir):
        parser.error('validation index and cache must be provided together')
    if args.scope == 'training' and not args.val_index:
        parser.error('training scope requires explicit development validation')
    if bool(args.val_chemical_prior_manifest) != bool(args.chemical_prior_manifest and args.val_index):
        parser.error('chemical-prior validation needs its matching manifest')
    if args.uniform_torsion_probability and not args.chemical_prior_manifest:
        parser.error('the mixed prior requires its chemical conformer manifest')
    if bool(args.resume_best) != bool(args.resume and args.val_index):
        parser.error('resuming with validation requires the matching best checkpoint')
    payload = json.loads(args.index.read_text())
    excluded = payload.get('metadata', {}).get('excluded_benchmark_pdb_ids')
    entries, clusters, _ = _load_index(args.index)
    if (not entries or not excluded
            or any(row['case'].split('_')[0].upper() in {x.upper() for x in excluded} for row in entries)):
        raise ValueError('require a nonempty explicitly benchmark-excluded training index')
    if any(Path(row['ligand']).suffix.lower() != '.sdf' for row in entries):
        raise ValueError('anchored training requires restored SDF chemistry, not inferred PDB bonds')
    validation_contract, validation_entries, split_audit = None, None, None
    if args.val_index:
        validation_payload = json.loads(args.val_index.read_text())
        split_audit = validate_index_pair(payload, validation_payload)
        validation_entries, _, _ = _load_index(args.val_index)
        validation_contract = dict(policy=VALIDATION_POLICY, times=list(VALIDATION_TIMES), seed=args.seed,
                                   index_sha256=hashlib.sha256(args.val_index.read_bytes()).hexdigest())
    runner = EvaluationRunner(args.project_root, args.parent.resolve(), device=args.device,
                              allow_conditioned_checkpoint=True)

    def load(entry, cache_dir=None):
        cache_dir = args.cache_dir if cache_dir is None else cache_dir
        cached = torch.load(cache_dir / f'{entry["case"]}.pt', map_location='cpu', weights_only=False)
        migration = cached.get('chemistry_migration', {})
        ccd_hash = migration.get('ccd_sha256', '')
        if (migration.get('coordinates_changed') is not False
                or migration.get('ligand_graph_rebuilt') is not True
                or len(ccd_hash) != 64 or any(c not in '0123456789abcdef' for c in ccd_hash)):
            raise ValueError('training cache lacks verified CCD chemistry-migration provenance')
        del cached
        return featurize_complex(
            entry['case'], entry['protein1'], entry['protein2'], entry['ligand'], runner.score_model_args,
            args.device, cache_dir=str(cache_dir), cache_read_only=True, cache_wait_seconds=0)

    first = load(entries[0])
    runner._ensure_model(first[3])
    del first
    if args.body_aware_action or args.coupled_linker_action or args.prior_joint_action:
        from tricoopdiff.anchored_action_adapter import ADAPTER_POLICY
        from tricoopdiff.coupled_linker_action import COUPLED_ADAPTER_POLICY
        from tricoopdiff.prior_joint_action import PRIOR_ACTION_POLICY
        install_action_adapter(runner.model, policy=PRIOR_ACTION_POLICY if args.prior_joint_action else (
            COUPLED_ADAPTER_POLICY if args.coupled_linker_action else ADAPTER_POLICY))
    parent_sha = runner.checkpoint_sha256
    if args.warm_start:
        if not args.resume:
            warm_start_anchored_weights(runner.model, torch.load(
                args.warm_start, map_location='cpu', weights_only=False),
                reset_action_adapter=args.reset_action_adapter)
        parent_sha = hashlib.sha256(args.warm_start.read_bytes()).hexdigest()
    trainer = AnchoredTrainer(runner.model, parent_checkpoint_sha256=parent_sha,
                              training_index_sha256=hashlib.sha256(args.index.read_bytes()).hexdigest(),
                              lr=args.lr, validation_contract=validation_contract, binary_bounds=bounds,
                              conditioning_task=args.conditioning_task,
                              binary_metric=args.binary_metric, data_schedule=args.data_schedule,
                              prior_config=(chemical_prior_config(seed=args.seed, count=args.chemical_prior_count,
                                                                 uniform_probability=args.uniform_torsion_probability,
                                                                 assembly=args.conditioning_task == 'dual_assembly')
                                            if args.chemical_prior_manifest else None),
                              prior_library_identity=(
                                  {'train': hashlib.sha256(args.chemical_prior_manifest.read_bytes()).hexdigest(),
                                   'validation': (hashlib.sha256(args.val_chemical_prior_manifest.read_bytes()).hexdigest()
                                                  if args.val_chemical_prior_manifest else None)}
                                  if args.chemical_prior_manifest else None))
    prior_library = (ChemicalPriorLibrary(args.chemical_prior_manifest, trainer.prior_config,
        index_sha256=trainer.index_sha) if args.chemical_prior_manifest else None)
    val_prior_library = (ChemicalPriorLibrary(args.val_chemical_prior_manifest, trainer.prior_config,
        index_sha256=validation_contract['index_sha256']) if args.val_chemical_prior_manifest else None)
    epoch, cursor = 0, 0
    torch.manual_seed(args.seed)
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('scope') != args.scope or saved.get('progress', {}).get('seed') != args.seed:
            raise ValueError('resume cannot change scope or seed')
        progress = trainer.restore(saved)
        epoch, cursor = progress['epoch'], progress['next_case']
        if cursor >= len(epoch_entries(entries, clusters, seed=args.seed, epoch=epoch, policy=args.data_schedule)):
            raise ValueError('resume cursor outside training index')
        if args.resume_best:
            best = torch.load(args.resume_best, map_location='cpu', weights_only=False)
            validate_anchored_checkpoint(best)
            for key in ('scope', 'parent_checkpoint_sha256', 'training_index_sha256',
                        'validation_contract', 'runtime_config', 'model_config',
                        'expert_base_identity', 'expert_environment', 'binary_bounds', 'action_policy',
                        'conditioning_task', 'prior_config', 'prior_library_identity', 'binary_metric', 'data_schedule',
                        'action_adapter'):
                default = {'conditioning_task': 'dual_binary', 'binary_metric': PARAMETER_METRIC,
                           'data_schedule': INDEX_ORDER}.get(key)
                if best.get(key, default) != saved.get(key, default):
                    raise ValueError(f'resume best checkpoint mismatch: {key}')
            if (best['global_step'] != saved['validation_state']['best_step']
                    or best['validation_state'] != saved['validation_state']):
                raise ValueError('resume best checkpoint is not the recorded selected checkpoint')
            del best
    args.out.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), ROOT / 'scripts/featurize_common.py', ROOT / 'scripts/evaluation_runner.py',
                    *sorted((ROOT / 'tricoopdiff').rglob('*.py'))]
    config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'training_index_sha256': trainer.index_sha, 'parent_checkpoint_sha256': trainer.parent_sha,
              'code_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in source_files}, 'validation_contract': validation_contract,
              'split_audit': split_audit, 'validation_scope': 'internal development; parent used historical validation'}
    config['prior_library_sha256'] = prior_library.sha256 if prior_library else None
    config['val_prior_library_sha256'] = val_prior_library.sha256 if val_prior_library else None
    with (args.out / 'config.json').open('x') as handle:
        json.dump(config, handle, indent=2)
    if args.resume_best:
        with args.resume_best.open('rb') as source, (
                args.out / f'step_{trainer.validation_state["best_step"]}.pt').open('xb') as destination:
            shutil.copyfileobj(source, destination)

    def validate():
        report = evaluate_panel(trainer, validation_entries,
                                lambda entry: load(entry, args.val_cache_dir),
                                index_sha256=validation_contract['index_sha256'], seed=args.seed,
                                prior_library=val_prior_library)
        improved = trainer.record_validation(report)
        with (args.out / f'validation_step_{trainer.global_step}.json').open('x') as handle:
            json.dump({**report, 'improved': improved}, handle, indent=2, allow_nan=False)
        print(json.dumps({'validation': report['mean_loss'], 'evaluated': report['evaluated'],
                          'rejected': report['rejected'], 'step': trainer.global_step,
                          'improved': improved}), flush=True)

    if validation_contract is not None and not args.resume:
        validate()
        trainer.save(args.out / 'step_0.pt', epoch=epoch, next_case=cursor, seed=args.seed, scope=args.scope)
    initial_step = trainer.global_step
    deadline_reached = False
    stop_requested = False

    def request_save_and_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_save_and_stop)
    since_success = 0
    visits = epoch_entries(entries, clusters, seed=args.seed, epoch=epoch, policy=args.data_schedule)
    with (args.out / 'metrics.jsonl').open('x', buffering=1) as log:
        while ((args.max_updates == 0 or trainer.global_step - initial_step < args.max_updates)
               and not deadline_reached and not stop_requested):
            entry = visits[cursor]
            case_seed = int.from_bytes(hashlib.sha256(
                f'{args.seed}:{epoch}:{entry["case"]}'.encode()).digest()[:8], 'big')
            generator = torch.Generator(device=args.device).manual_seed(case_seed)
            e3, poi, ligand, _ = load(entry)
            record = {'case': entry['case'], 'epoch': epoch, 'case_index': cursor, 'case_seed': case_seed}
            try:
                sample = prepare_task_sample(args.conditioning_task,
                    e3, poi, ligand, generator=generator, evidence_id=f'training:{entry["case"]}',
                    binary_bounds=bounds)
                if sample is not None and prior_library is not None:
                    sample = apply_training_prior(sample, ligand, prior_library.get(entry['case'], ligand['mol']),
                        generator=generator, config=trainer.prior_config)
            except ValueError as error:
                sample = None
                record['rejection'] = str(error)
            # Model/backward exceptions are fatal, not relabeled as data skips.
            if sample is None:
                since_success += 1
                record.update(status='skipped', rejection=record.get('rejection', 'no_chemical_cut_pair'))
            else:
                time = float(torch.rand((), device=args.device, generator=generator).clamp_min(1e-4))
                metrics = trainer.step(sample, e3, poi, ligand, time=time, dt=.05)
                record.update(status='updated', time=time, audit=sample.audit, **metrics)
                since_success = 0
            cursor += 1
            if cursor == len(visits):
                cursor, epoch = 0, epoch + 1
                visits = epoch_entries(entries, clusters, seed=args.seed, epoch=epoch, policy=args.data_schedule)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            deadline_reached = (args.max_seconds is not None
                                and wall_clock.monotonic() - started >= args.max_seconds
                                and trainer.global_step > initial_step)
            if (record['status'] == 'updated' and trainer.global_step % args.log_every == 0):
                print(json.dumps({key: value for key, value in record.items() if key != 'audit'}), flush=True)
            if since_success >= len(visits):
                raise RuntimeError('one complete traversal had no eligible training samples')
            if record['status'] == 'updated' and (
                    trainer.global_step % args.save_every == 0
                    or trainer.global_step - initial_step == args.max_updates):
                if validation_contract is not None:
                    validate()
                trainer.save(args.out / f'step_{trainer.global_step}.pt', epoch=epoch,
                             next_case=cursor, seed=args.seed, scope=args.scope)
    final_path = args.out / f'step_{trainer.global_step}.pt'
    if not final_path.exists():
        if validation_contract is not None:
            validate()
        trainer.save(final_path, epoch=epoch, next_case=cursor, seed=args.seed, scope=args.scope)
    with (args.out / 'completion.json').open('x') as handle:
        json.dump({'global_step': trainer.global_step, 'initial_step': initial_step,
                   'stop_requested': stop_requested,
                   'deadline_reached': deadline_reached, 'elapsed_seconds': wall_clock.monotonic() - started,
                   'final_checkpoint': str(final_path.resolve())}, handle, indent=2)
    if validation_contract is not None:
        best_path = args.out / f'step_{trainer.validation_state["best_step"]}.pt'
        with (args.out / 'best.json').open('x') as handle:
            json.dump({**trainer.validation_state, 'criterion': VALIDATION_POLICY,
                       'binary_metric': trainer.binary_metric, 'conditioning_task': trainer.conditioning_task,
                       'checkpoint': str(best_path.resolve()),
                       'checkpoint_sha256': hashlib.sha256(best_path.read_bytes()).hexdigest(),
                       'scope': args.scope, 'accuracy_claim': False}, handle, indent=2)


if __name__ == '__main__':
    main()
