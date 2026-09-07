"""Measured stage adapters for the single-state model-v2 overnight workflow."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
from run_overnight import V2_STAGES, gpu3_visibility, write_json


def read_entries(path):
    return json.loads(Path(path).read_text())['entries']


def prepare_training_index(config, output):
    """Preserve training; filter contaminated validation clusters in run-local copies."""
    source_path = Path(config['val_index'])
    source = json.loads(source_path.read_text())
    excluded_pdbs = {e['case'].split('_')[0].upper()
                     for key in ('train_index', 'protac_index', 'mgd_index')
                     for e in read_entries(config[key])}
    if source.get('clusters'):
        clusters = [c for c in source['clusters'] if not any(
            case.split('_')[0].upper() in excluded_pdbs for case in c['items'])]
        retained = {case for c in clusters for case in c['items']}
        prepared = {**source, 'clusters': clusters,
                    'entries': [e for e in source['entries'] if e['case'] in retained]}
    else:
        prepared = {**source, 'entries': [e for e in source['entries']
                     if e['case'].split('_')[0].upper() not in excluded_pdbs]}
    if not prepared['entries']:
        raise ValueError('no validation samples remain after split decontamination')
    prepared['overnight_provenance'] = {
        'source': str(source_path), 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'policy': 'retain full training; drop validation clusters sharing training/benchmark PDB',
        'original_entries': len(source['entries']), 'retained_entries': len(prepared['entries']),
        'removed_cases': sorted({e['case'] for e in source['entries']} -
                                {e['case'] for e in prepared['entries']})}
    path = output / 'val_index.json'
    write_json(path, prepared)
    train_path = output / 'train_index.json'
    write_json(train_path, json.loads(Path(config['train_index']).read_text()))
    return {**config, 'train_index': str(train_path), 'val_index': str(path)}, prepared['overnight_provenance']


def inspect_inputs(config):
    indexes = [read_entries(config[key]) for key in
               ('train_index', 'val_index', 'protac_index', 'mgd_index')]
    train, val, protac, mgd = [{e['case'].split('_')[0].upper() for e in entries}
                              for entries in indexes]
    if not train or not val or train & val or (train | val) & (protac | mgd):
        raise ValueError('empty split or same-PDB leakage across training/validation/benchmark')
    missing = [e[key] for entries in indexes for e in entries
               for key in ('protein1', 'protein2', 'ligand') if not Path(e[key]).is_file()]
    for path in (Path(config['expert_dir']) / 'best_ema_inference_epoch_model.pt',
                 Path(config['fixed_pair']), Path(config['deepternary_root']) / 'DockQ/fnat'):
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing inputs, first: {missing[:5]}')
    return [len(entries) for entries in indexes]


def regression(config, output):
    import torch
    torch.set_num_threads(2)
    config, split = prepare_training_index(config, output)
    counts = inspect_inputs(config)
    pair = torch.load(config['fixed_pair'], map_location='cpu', weights_only=False)
    if pair['batch']['case'] not in {e['case'] for e in read_entries(config['train_index'])}:
        raise ValueError('overfit pair is not a member of the training set')
    passed = []
    for file in ('test_model_build.py', 'test_consensus.py', 'test_sdpi_flow.py',
                 'test_training_contract.py', 'test_module_boundaries.py',
                 'test_evaluation_common.py', 'test_overnight.py', 'test_v2_experiment.py'):
        namespace = runpy.run_path(str(ROOT / 'tests' / file))
        for name, function in namespace.items():
            if name.startswith('test_') and callable(function):
                with tempfile.TemporaryDirectory(prefix='v2-regression-') as directory:
                    kwargs = {'tmp_path': Path(directory)} if 'tmp_path' in inspect.signature(function).parameters else {}
                    function(**kwargs)
                passed.append(file + '::' + name)
        print('PASS', file, flush=True)
    return {'checks': dict.fromkeys(V2_STAGES['regression'], True), 'tests': passed,
            'index_counts': counts, 'fixed_case': pair['batch']['case'], 'split': split}


def smoke(config, output):
    target = output / 'real_smoke'
    subprocess.run([sys.executable, str(ROOT / 'scripts/validate_consensus_model.py'),
                    '--pair', config['fixed_pair'], '--out', str(target),
                    '--expert-model-dir', config['expert_dir']], check=True)
    evidence = json.loads((target / 'summary.json').read_text())
    checks = {name: evidence.get(name) is True for name in V2_STAGES['smoke']
              if name != 'finite_trajectory'}
    checks['finite_trajectory'] = evidence['trajectory_steps'] == 20
    return {'checks': checks, 'summary': evidence}


def evaluate(config, output, checkpoint, kind):
    from evaluation_runner import EvaluationRunner
    entry = read_entries(config[kind + '_index'])[0]
    paths = {key: Path(entry[key]) for key in ('protein1', 'protein2', 'ligand')}
    if kind == 'protac':
        # Bound templates are metric-only; the predictors still receive unbound receptors.
        for key in ('protein1', 'protein2'):
            paths['metric_' + key] = paths[key].parent / (key + '.pdb')
    native = Path(entry.get('complex', paths['ligand'].parent / 'gt_complex.pdb'))
    runner = EvaluationRunner(ROOT, checkpoint, model_dir=config['expert_dir'],
                              deepternary_root=config['deepternary_root'])
    report = runner.evaluate_case(
        entry['case'], paths, cache_dir=config[kind + '_cache'],
        native_complex_pdb=native if native.is_file() else None,
        n_walkers=config['walkers'], steps=20, seed=42)
    write_json(output / 'metrics.json', report)
    ensemble, structure = report['ensemble'], report['structure']
    checks = {'fresh_checkpoint_load': runner.model is not None,
              'local_metrics': structure['protac_rmsd_mean'] is not None,
              'global_metrics': ensemble['n_exact_dockq_walkers'] > 0,
              'failures_counted': ensemble['n_walkers'] == config['walkers']
              and ensemble['n_valid_walkers'] + ensemble['n_nan_walkers'] == config['walkers']}
    return {'checks': checks, 'checkpoint': str(Path(checkpoint).resolve()),
            'case': entry['case'], 'structure': structure, 'ensemble': ensemble,
            'scope': 'one preselected case; oracle best and mean, not confidence top-1'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=list(V2_STAGES), required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'scripts/overnight_v2.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu3_visibility(direct=True):
        parser.error('runner must expose only the UUID of physical GPU 3')
    if os.environ.get('EXPERIMENT_STAGE') != args.stage or os.environ.get('EXPERIMENT_ARCHITECTURE') != 'residual_consensus_v2':
        parser.error('stage/architecture must match the overnight runner')
    config = json.loads(args.config.read_text())['experiment']
    output = args.out.resolve()
    if args.stage in ('pilot', 'train', 'overfit'):
        config['train_index'] = str(output.parent / 'regression/train_index.json')
        config['val_index'] = str(output.parent / 'regression/val_index.json')
        if not Path(config['train_index']).is_file():
            raise FileNotFoundError('prepared training index from regression is required')
    started = time.monotonic()
    if args.stage == 'regression':
        evidence = regression(config, output)
    elif args.stage == 'smoke':
        evidence = smoke(config, output)
    elif args.stage in ('overfit', 'pilot', 'train'):
        from v2_training import overfit, train
        evidence = (overfit(config, output, args.seconds) if args.stage == 'overfit'
                    else train(config, output, args.seconds, pilot=args.stage == 'pilot'))
    else:
        evidence = evaluate(config, output, args.checkpoint, args.stage.removeprefix('evaluate_'))
    evidence['elapsed_seconds_total'] = time.monotonic() - started
    evidence['known_limitation'] = (
        'Q-012: native training scaffold versus ETKDG inference scaffold remains unresolved. '
        'This is a bounded v2 baseline experiment, not a claim of corrected endpoint support.')
    write_json(output / 'evidence.json', evidence)
    report = {'run_id': os.environ['EXPERIMENT_RUN_ID'], 'architecture': 'residual_consensus_v2',
              'stage': args.stage, 'checks': evidence['checks'],
              'passed': all(evidence['checks'].get(key) is True for key in V2_STAGES[args.stage]),
              'artifacts': [str(output / 'evidence.json')],
              'checkpoint': evidence.get('checkpoint')}
    write_json(Path(os.environ['EXPERIMENT_REPORT']), report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
