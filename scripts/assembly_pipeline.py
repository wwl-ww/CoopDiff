#!/usr/bin/env python
"""Candidate-driven ternary assembly: generate, train/resume, and predict."""

import argparse
from argparse import Namespace
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import timedelta
import fcntl
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from rdkit import Chem
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tricoopdiff.diffdock_compat  # noqa: E402,F401
from featurize_common import featurize_complex  # noqa: E402
from train_entry import _load_index, _load_score_model_args, TernaryDataLoader  # noqa: E402
from tricoopdiff.initialization.candidates import (  # noqa: E402
    load_assembly_candidate, generate_joint_matcha_ligand,
)
from tricoopdiff.models.assembly import (  # noqa: E402
    CANDIDATE_FRAMEWORK_VERSION, TernaryAssembly, TernaryAssemblyModel,
    FLOW_MOTION_CONTRACT, BOUNDED_REPAIR_MOTION_CONTRACT,
    QUALITY_GATED_REPAIR_MOTION_CONTRACT, QUALITY_GATE_ERROR_SCALE,
    production_model_config, CANDIDATE_GEOMETRY_CONTRACT, CANDIDATE_QUALITY_CONTRACT,
    CANDIDATE_CONDITION_CONTRACT,
    CANDIDATE_REFERENCE_CONTRACT,
    CANDIDATE_TASK_CONTRACT,
)
from tricoopdiff.models.expert import FrozenMatcha  # noqa: E402
from tricoopdiff.runtime import (  # noqa: E402
    atomic_json, atomic_torch_save, capture_rng_state, restore_rng_state,
    sha256_file, preserve_rng,
)
from tricoopdiff.sdpi_flow import (  # noqa: E402
    BOUNDED_CORRECTION_LIMITS, CORRECTION_STATE_NOISE, SDPIFlow,
)
from tricoopdiff.train import AssemblyTrainer  # noqa: E402
from tricoopdiff.distributed import GradientSynchronizer  # noqa: E402
from tricoopdiff.initialization.endpoints import EndpointProjectionError  # noqa: E402

RUNNING_DOCKING = set()
STOPPING = False
ACTIVE_PREFETCH = None


class ConformerGenerationFailure(ValueError):
    """A known ETKDG data failure, not an arbitrary model/runtime error."""


class FeatureCacheUnavailable(FileNotFoundError):
    """No compatible reusable cache; feature reconstruction is forbidden."""


def mark_conformer_failure(opt, entry, expert, reason):
    protocol = generation_protocol(opt, entry, expert)
    path = opt.bank/entry['case']/f'initialization_failed.{protocol["generator_sha256"][:12]}.json'
    atomic_json({'protocol': protocol,
                 'case': entry['case'], 'reason': reason, 'retry': False}, path)
    return path


def cached_admission(opt, entries):
    """Pin availability without changing the original rank/cursor ordering.

    File existence is only availability, not validity. The original featurizer
    verifies chemistry/source signatures with cache_only=True at each read.
    """
    path = opt.output/'data_admission.json'
    base = {'contract': 'cached_only_features_online_initializations_v1',
            'training_index_sha256': sha256_file(opt.index),
            'ordered_cases': [e['case'] for e in entries]}
    if path.exists():
        result = json.loads(path.read_text())
        if any(result.get(k) != v for k, v in base.items()):
            raise ValueError('pinned cached-only admission differs from training index')
        return result
    available, missing = [], []
    for entry in entries:
        name = entry['case']+'.pt'
        exists = (Path(opt.cache_dir)/name).is_file() or (
            opt.feature_cache_source is not None and (opt.feature_cache_source/name).is_file())
        (available if exists else missing).append(entry['case'])
    if not available:
        raise ValueError('cached-only training has no available feature caches')
    result = {**base, 'available_cases': available, 'missing_cache_cases': missing,
              'admission_note': 'Availability only; compatibility is checked on read, no rebuild permitted.'}
    atomic_json(result, path)
    return result


@preserve_rng
def cluster_epoch_plan(entries, clusters, available, seed, epoch, world):
    """Reuse the original loader's 80/20 selection and shuffle verbatim."""
    eligible = [e for e in entries if available is None or e['case'] in available]
    by_case = {e['case']: e for e in eligible}
    clusters = [c for c in clusters if any(case in by_case for case in c['items'])]
    owner = {}
    for index, cluster in enumerate(clusters):
        for case in cluster['items']:
            if case in owner:
                raise ValueError('a case belongs to multiple training clusters')
            owner[case] = index % world
    loader = Namespace(entries=eligible, entries_by_case=by_case, clusters=clusters, shuffle=True)
    np.random.seed(seed+epoch)
    selected = TernaryDataLoader._epoch_entries(loader)
    indices = {e['case']: i for i, e in enumerate(entries)}
    shards = [[indices[e['case']] for e in selected if owner[e['case']] == rank]
              for rank in range(world)]
    return selected, shards, owner


def stop_cpu_jobs(signum, frame):
    """Torchrun termination must not leave CPU docking workers orphaned."""
    global STOPPING
    STOPPING = True
    for process in RUNNING_DOCKING.copy():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128+signum)


def finish_prefetch(pool):
    """Release bounded CPU work after a saved training budget boundary."""
    global STOPPING
    STOPPING = True
    for process in RUNNING_DOCKING.copy():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pool:
        pool.shutdown(wait=True, cancel_futures=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('generate', 'train', 'predict', 'baseline', 'migrate'))
    parser.add_argument('--index', required=True)
    parser.add_argument('--bank', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--feature-cache-source', type=Path,
                        help='Copy existing features into this run private cache on first use; never hard-link shared files')
    parser.add_argument('--cached-only', action='store_true',
                        help='Reuse compatible feature caches only; never reconstruct PDB chemistry or ESM')
    parser.add_argument('--adopt-cached-only-resume', action='store_true',
                        help='Explicitly adopt cached-only admission for a pre-admission checkpoint; preserve its data ordering')
    parser.add_argument('--adopt-cluster-sampling', action='store_true',
                        help='Explicitly reset a former flat data cursor to cluster epoch 0, retaining weights/optimizer')
    parser.add_argument('--score-model-dir', default='checkpoints/score_model')
    parser.add_argument('--matcha-source-dir', default='results/matcha_qualification_source')
    parser.add_argument('--matcha-weights-dir', default='results/matcha_qualification_weights')
    parser.add_argument('--hdock-dir', default='results/hdocklite_install', type=Path)
    parser.add_argument('--fftw-lib-dir', default='/home/wl/micromamba/envs/proteinprep/lib')
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--hdock-workers', type=int, default=16,
                        help='Parallel CPU docking processes; Matcha generation stays on one GPU')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--motion-contract', choices=(FLOW_MOTION_CONTRACT,
                        BOUNDED_REPAIR_MOTION_CONTRACT,
                        QUALITY_GATED_REPAIR_MOTION_CONTRACT),
                        help='Training motion contract; weights-from can deliberately change it')
    parser.add_argument('--generator-steps-per-stage', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--online-cache-size', type=int, default=0,
                        help='Generate and use one new pose per visit until full, then uniformly sample; 0 uses static pools')
    parser.add_argument('--reuse-generator-sha256', action='append', default=[],
                        help='Explicitly retain immutable successful poses from a recorded older generator digest')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--cpu-budget', type=int, help='Total CPU affinity budget shared across training ranks')
    parser.add_argument('--max-updates', type=int, help='Qualification limit; full checkpoint remains resumable')
    parser.add_argument('--max-hours', type=float,
                        help='Stop after a saved update at this wall-clock budget; resume can continue later')
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--quality-lr', type=float, default=1e-3)
    parser.add_argument('--sdpi-replay-weight', type=float, default=.2,
                        help='Fraction of motion objective on a fresh SDPI start for each pool update')
    parser.add_argument('--node-dim', type=int, default=128)
    parser.add_argument('--near-target-frequency', type=int, default=4,
                        help='Every N updates add a zero/perturbed near-target task; 0 disables')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cuda-memory-fraction', type=float,
                        help='Optional CUDA allocator limit for sharing a physical GPU')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--weights-from', type=Path,
                        help='Explicit weights-only start: reset optimizer/RNG/counters for a new objective')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--baseline', type=Path, help='SDPI checkpoint; migrate always reads EMA, never raw')
    parser.add_argument('--frame-policy', choices=('legacy_argmax_v1', 'backbone_rigid_transport_v1'),
                        default='backbone_rigid_transport_v1')
    parser.add_argument('--selection', choices=('hdock', 'learned'), default='hdock',
                        help='Keep HDOCK rank by default; learned scoring is an explicit uncalibrated experiment')
    parser.add_argument('--case', action='append', dest='cases',
                        help='Restrict evaluation to this case; may be repeated')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--native-root', type=Path,
                        help='Optional post-selection metrics only; never used by the predictor')
    opt = parser.parse_args(argv)
    if min(opt.candidates, opt.steps, opt.generator_steps_per_stage, opt.epochs, opt.hdock_workers) < 1:
        parser.error('candidate counts, steps and epochs must be positive')
    if opt.limit is not None and opt.limit < 1:
        parser.error('limit must be positive')
    if opt.near_target_frequency < 0:
        parser.error('near-target frequency must be nonnegative')
    if opt.online_cache_size < 0 or opt.cpu_threads < 1 or (opt.max_updates is not None and opt.max_updates < 1):
        parser.error('invalid cache size, CPU thread count or update limit')
    if opt.max_hours is not None and (not np.isfinite(opt.max_hours) or opt.max_hours <= 0):
        parser.error('max-hours must be finite positive')
    if opt.cpu_budget is not None and opt.cpu_budget < 1:
        parser.error('CPU budget must be positive')
    if opt.cuda_memory_fraction is not None and not 0 < opt.cuda_memory_fraction <= 1:
        parser.error('CUDA memory fraction must be in (0, 1]')
    if opt.online_cache_size and opt.mode != 'train':
        parser.error('online caches are training-only; validation uses a separate fixed bank')
    if not all(np.isfinite(value) and value > 0 for value in (opt.lr, opt.quality_lr)):
        parser.error('motion and quality learning rates must be finite and positive')
    if not np.isfinite(opt.sdpi_replay_weight) or not 0 <= opt.sdpi_replay_weight < 1:
        parser.error('SDPI replay weight must be finite in [0,1)')
    if opt.mode == 'predict' and not opt.checkpoint:
        parser.error('predict requires --checkpoint')
    if opt.resume and opt.mode != 'train':
        parser.error('resume is a training-only operation')
    if opt.weights_from and (opt.mode != 'train' or opt.resume):
        parser.error('weights-from is training-only and mutually exclusive with resume')
    if opt.checkpoint and opt.mode != 'predict':
        parser.error('checkpoint is a prediction-only operation')
    if (opt.mode == 'migrate') != bool(opt.baseline):
        parser.error('migrate requires --baseline; other modes do not accept it')
    return opt


def input_identity(entry):
    return {key: sha256_file(entry[key]) for key in ('protein1', 'protein2', 'ligand')}


def read_payload(path, device='cpu'):
    """Pin the loaded bytes and their digest, even if a producer publishes a new file."""
    contents = Path(path).read_bytes()
    return (torch.load(io.BytesIO(contents), map_location=device, weights_only=False),
            hashlib.sha256(contents).hexdigest())


def matcha_reference_paths(entry, cache_dir, e3, poi):
    """Give Matcha the same receptor residues and CA altlocs as DiffDock."""
    import prody as pr

    directory = Path(cache_dir)/'matcha_references'/entry['case']
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, data in (('protein1', e3), ('protein2', poi)):
        receptor_ca = pr.parsePDB(str(entry[key])).ca
        selected = {
            (chain, f'{int(number):4d}{insertion or " "}', name): xyz
            for chain, number, insertion, name, xyz in zip(
                receptor_ca.getChids(), receptor_ca.getResnums(),
                receptor_ca.getIcodes(), receptor_ca.getResnames(),
                receptor_ca.getCoords())}
        if len(selected) != data['coords'].shape[1]:
            raise ValueError(f'{entry["case"]}: Matcha receptor selection differs from DiffDock')
        output = []
        found_ca = set()
        for line in Path(entry[key]).read_text().splitlines():
            if not line.startswith('ATOM'):
                continue
            identity = (line[21], line[22:27], line[17:20])
            if identity not in selected:
                continue
            if line[12:16].strip() == 'CA':
                xyz = selected[identity]
                line = line[:30]+''.join(f'{float(value):8.3f}' for value in xyz)+line[54:]
                found_ca.add(identity)
            output.append(line)
        if found_ca != set(selected):
            raise ValueError(f'{entry["case"]}: selected receptor CA records are missing')
        destination = directory/f'{key}.pdb'
        contents = '\n'.join(output)+'\nEND\n'
        if not destination.exists() or destination.read_text() != contents:
            destination.write_text(contents)
        paths[key] = str(destination)
    return paths


def load_case(opt, entry, expert, score_args):
    # Offline mode prevents a cache miss from silently downloading models.
    destination = Path(opt.cache_dir)/(entry['case']+'.pt')
    if opt.feature_cache_source and not destination.exists():
        source = opt.feature_cache_source/destination.name
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f'.copy.{os.getpid()}')
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
    try:
        data = featurize_complex(entry['case'], entry['protein1'], entry['protein2'],
            entry['ligand'], score_args, opt.device, cache_dir=opt.cache_dir,
            cache_only=opt.cached_only)
    except FileNotFoundError as error:
        if opt.cached_only and 'cache-only mode requires a compatible cache directory' in str(error):
            raise FeatureCacheUnavailable(str(error)) from error
        raise
    loader = Namespace(expert=expert, cache_dir=opt.cache_dir, device=torch.device(opt.device))
    matcha_entry = entry
    if opt.mode in ('generate', 'baseline'):
        matcha_entry = {**entry, **matcha_reference_paths(entry, opt.cache_dir, data[0], data[1])}
    e3, poi, ligand = preserve_rng(TernaryDataLoader._prepare_matcha)(loader, matcha_entry, data[:3])
    if expert.frame_policy == 'legacy_argmax_v1':
        parts = []
        for part in (e3, poi):
            centered = part['coords'][0]-part['coords'][0].mean(0)
            first, second = part['matcha']['anchors']
            e1 = centered[first]/centered[first].norm()
            e2 = centered[second]-(centered[second] @ e1)*e1
            e2 = e2/e2.norm()
            parts.append({**part, 'matcha': {**part['matcha'],
                'reference_basis': torch.stack((e1, e2, torch.linalg.cross(e1, e2)), -1)}})
        e3, poi = parts
    from tricoopdiff.metrics.dockq import _records, _featurized_ca_coords, _fit_rigid
    table = Chem.GetPeriodicTable()
    prepared = []
    for part, key in ((e3, 'protein1'), (poi, 'protein2')):
        lines, indices, atoms = _records(entry[key], 'ATOM')
        ca = _featurized_ca_coords(entry[key])
        rotation, translation = _fit_rigid(ca, part['coords'][0].cpu().numpy())
        elements = []
        for index in indices:
            line = lines[index]
            # Standard protein ATOM names: CA is carbon, not calcium.
            elements.append((line[76:78].strip() if len(line) >= 78 else '') or line[12:16].strip()[0])
        selected = [i for i, element in enumerate(elements) if element.upper() not in ('H', 'D')]
        prepared.append({**part,
            'steric_atoms': part['coords'].new_tensor(atoms[selected] @ rotation+translation)[None],
            'steric_ca_center': part['coords'][0].mean(0).detach(),
            'steric_radii': part['coords'].new_tensor([table.GetRvdw(elements[i].title()) for i in selected])})
    e3, poi = prepared
    return e3, poi, ligand, data[3]


def write_hdock_input(path, destination, chain, rotation):
    records = []
    identities = set()
    for line in Path(path).read_text().splitlines():
        key = (line[22:27], line[12:16])
        if (not line.startswith('ATOM') or line[16] not in (' ', 'A')
                or key in identities):
            continue
        identities.add(key)
        records.append(line)
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in records])
    if not len(records) or not np.isfinite(xyz).all():
        raise ValueError('HDOCK input must contain finite protein atoms')
    xyz = (xyz-xyz.mean(0)) @ rotation
    destination.write_text('\n'.join(l[:16]+' '+l[17:21]+chain+l[22:30]
        + ''.join(f'{v:8.3f}' for v in point)+l[54:]
        for l, point in zip(records, xyz)) + '\nEND\n')


def docking_job(opt, entry):
    """Independent CPU search, with exact input/binary resume identity."""
    directory = opt.bank/entry['case']
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'hdock.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _docking_job_locked(opt, entry, directory)


def _docking_job_locked(opt, entry, directory):
    if STOPPING:
        raise InterruptedError('CPU docking stopped with training')
    seed = (opt.seed+int(hashlib.sha256(entry['case'].encode()).hexdigest()[:8], 16)) % (2**31-1)
    identity = {'inputs': input_identity(entry), 'seed': opt.seed,
                'hdock_sha256': sha256_file(opt.hdock_dir/'hdock')}
    marker = directory/'docking_complete.json'
    if marker.exists() and (directory/'Hdock.out').is_file() and json.loads(marker.read_text()) == identity:
        return entry
    write_hdock_input(entry['protein1'], directory/'r.pdb', 'A', np.eye(3))
    write_hdock_input(entry['protein2'], directory/'l.pdb', 'B',
                      Rotation.random(random_state=seed).as_matrix())
    env = {**os.environ, 'LD_LIBRARY_PATH': opt.fftw_lib_dir,
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (directory/'hdock.log').open('w') as log:
        process = subprocess.Popen([str((opt.hdock_dir/'hdock').resolve()), 'r.pdb', 'l.pdb', '-out', 'Hdock.out'],
            cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        RUNNING_DOCKING.add(process)
        try:
            status = process.wait(timeout=14400)
            if status:
                raise subprocess.CalledProcessError(status, process.args)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
            RUNNING_DOCKING.discard(process)
    atomic_json(identity, marker)
    return entry


def generate(opt, entries, expert, score_args):
    # Start bounded CPU searches ahead of the resident GPU generator. Do not
    # fork a process containing initialized CUDA models.
    pending = []
    for entry in entries:
        bank_path = opt.bank/entry['case']/'candidates.pt'
        if bank_path.exists():
            existing = torch.load(bank_path, map_location='cpu', weights_only=False)
            if not same_generation_protocol(existing['protocol'], generation_protocol(opt, entry, expert), opt):
                raise ValueError('candidate bank provenance changed; use a new bank directory')
            print(f"SKIP generated {entry['case']}", flush=True)
        else:
            pending.append(entry)
    failures = []
    with ThreadPoolExecutor(max_workers=opt.hdock_workers) as pool:
        futures = {pool.submit(docking_job, opt, entry): entry for entry in pending}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                generate_case(opt, future.result(), expert, score_args)
                (opt.bank/entry['case']/'generation_failed.json').unlink(missing_ok=True)
            except Exception as error:
                row = {'case': entry['case'], 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc()}
                failures.append(row)
                atomic_json(row, opt.bank/entry['case']/'generation_failed.json')
                print(f'GENERATION FAILED {entry["case"]}: {row["error"]}', flush=True)
    atomic_json({'requested': len(entries), 'failed': failures}, opt.output/'generation_status.json')


def generation_protocol(opt, entry, expert):
    protocol = {'format': CANDIDATE_FRAMEWORK_VERSION, 'inputs': input_identity(entry),
        'expert': expert.provenance, 'seed': opt.seed, 'count': opt.candidates,
        'steps_per_stage': opt.generator_steps_per_stage,
        'generator_sha256': sha256_file(ROOT/'tricoopdiff/initialization/candidates.py'),
        'hdock_sha256': sha256_file(opt.hdock_dir/'hdock')}
    if opt.mode == 'generate':
        protocol['candidate_import_ca_policy'] = 'prody_receptor_residue_selection_v1'
    return protocol


def same_generation_protocol(recorded, expected, opt):
    """Only explicitly named old generators may retain their original poses.

    Do not rewrite provenance or content hashes. All other identities, seeds,
    frame contracts and sampling counts must still match exactly.
    """
    if recorded == expected:
        return True
    previous = recorded.get('generator_sha256')
    if previous not in opt.reuse_generator_sha256:
        return False
    return {**recorded, 'generator_sha256': expected['generator_sha256']} == expected


def candidate_import_paths(entry, directory, model_path, e3, poi):
    """Remove nonprotein residues excluded from DiffDock receptor nodes."""
    import prody as pr

    selected = []
    for key, data in (('protein1', e3), ('protein2', poi)):
        ca = pr.parsePDB(str(entry[key])).ca
        keys = {(f'{int(number):4d}{insertion or " "}', name)
                for number, insertion, name in zip(
                    ca.getResnums(), ca.getIcodes(), ca.getResnames())}
        if len(keys) != data['coords'].shape[1]:
            raise ValueError(f'{entry["case"]}: receptor C-alpha keys differ from features')
        selected.append(keys)

    def filter_pdb(source, destination, keys, chain=None):
        lines = [line for line in Path(source).read_text().splitlines()
                 if line.startswith('ATOM') and (chain is None or line[21] == chain)
                 and (line[22:27], line[17:20]) in keys]
        destination.write_text('\n'.join(lines)+'\nEND\n')
        return destination

    reference_paths = []
    for key, keys in zip(('protein1', 'protein2'), selected):
        source = entry[key]
        atoms = [line for line in Path(source).read_text().splitlines() if line.startswith('ATOM')]
        if all((line[22:27], line[17:20]) in keys for line in atoms):
            reference_paths.append(source)
        else:
            reference_paths.append(filter_pdb(source, directory/f'{key}_receptor_only.pdb', keys))
    if all(reference_paths[index] == entry[key]
           for index, key in enumerate(('protein1', 'protein2'))):
        return model_path, *reference_paths
    lines = [line for line in Path(model_path).read_text().splitlines()
             if line.startswith('ATOM') and line[21] in ('A', 'B')
             and (line[22:27], line[17:20]) in selected[0 if line[21] == 'A' else 1]]
    filtered_pose = directory/f'{Path(model_path).stem}_receptor_only.pdb'
    filtered_pose.write_text('\n'.join(lines)+'\nEND\n')
    return filtered_pose, *reference_paths


def generate_case(opt, entry, expert, score_args):
    """Main-thread CUDA generation, after the CPU search completes."""
    directory = opt.bank/entry['case']
    protocol = generation_protocol(opt, entry, expert)
    e3, poi, ligand, _ = load_case(opt, entry, expert, score_args)
    seed = (opt.seed+int(hashlib.sha256(entry['case'].encode()).hexdigest()[:8], 16)) % (2**31-1)
    with (directory/'export.log').open('w') as log:
        subprocess.run([str((opt.hdock_dir/'createpl').resolve()), 'Hdock.out', 'top.pdb',
            '-nmax', str(opt.candidates), '-complex', '-models'], cwd=directory,
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    candidates = []
    for rank in range(1, opt.candidates+1):
        model_path, protein1_path, protein2_path = candidate_import_paths(
            entry, directory, directory/f'model_{rank}.pdb', e3, poi)
        candidate = load_assembly_candidate(model_path, None,
            e3['coords'], poi['coords'], ligand, candidate_id=entry['case'], rank=rank,
            protein1_path=protein1_path, protein2_path=protein2_path)
        candidate = generate_joint_matcha_ligand(expert, e3, poi, ligand, candidate,
            seed=(seed+rank*1009) % (2**31-1), steps_per_stage=opt.generator_steps_per_stage)
        candidates.append({key: value.detach().cpu() if torch.is_tensor(value) else value
                           for key, value in candidate.items()})
    atomic_torch_save({'protocol': protocol, 'candidates': candidates}, directory/'candidates.pt')
    print(f"GENERATED {entry['case']} {len(candidates)} candidates", flush=True)


def load_bank(opt, entry, expert):
    path = opt.bank/entry['case']/'candidates.pt'
    bank, digest = read_payload(path)
    protocol = bank['protocol']
    if (protocol['format'] != CANDIDATE_FRAMEWORK_VERSION
            or protocol['inputs'] != input_identity(entry)
            or protocol['expert'] != expert.provenance):
        raise ValueError('candidate bank input, generator or format identity differs')
    ranks = [c['provenance']['protein_candidate_rank'] for c in bank['candidates']]
    if len(ranks) != protocol['count'] or sorted(ranks) != list(range(1, protocol['count']+1)):
        raise ValueError('candidate bank must contain exactly one pose per declared HDOCK rank')
    return bank['candidates'], digest


def online_candidate(opt, entry, expert, prepared, visit):
    """Immutable slots: visits 0..K-1 create/use that slot; later visits draw.

    A slot published before an interrupted optimizer update is reused on retry,
    not counted as an extra visit. Deterministic slot seeds and RNG preservation
    keep generation from perturbing the training stream.
    """
    size = opt.online_cache_size
    case_seed = (opt.seed+int(hashlib.sha256(entry['case'].encode()).hexdigest()[:8], 16)) % (2**31-1)
    slot = visit if visit < size else int(np.random.RandomState(case_seed+visit).randint(size))
    directory = opt.bank/entry['case']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'online_slot_{slot}.pt'
    protocol = {**generation_protocol(opt, entry, expert),
        'cache_contract': 'fill_new_then_uniform_fixed_slots_v1', 'capacity': size,
        'slot': slot, 'slot_seed': (case_seed+1009*(slot+1)) % (2**31-1)}
    with (directory/'online.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        failed = directory/f'initialization_failed.{protocol["generator_sha256"][:12]}.json'
        if failed.exists():
            record = json.loads(failed.read_text())
            if record['protocol'] != generation_protocol(opt, entry, expert):
                raise ValueError('failed initialization provenance differs')
            raise ConformerGenerationFailure(record['reason'])
        if path.exists():
            record, digest = read_payload(path)
            if not same_generation_protocol(record['protocol'], protocol, opt):
                raise ValueError(f'online candidate provenance differs: {path}')
            return record['candidate'], path, digest
        if visit >= size:
            raise FileNotFoundError(f'full online cache has a missing slot: {path}')
        docking_job(opt, entry)
        with (directory/'export.log').open('w') as log:
            subprocess.run([str((opt.hdock_dir/'createpl').resolve()), 'Hdock.out', 'top.pdb',
                '-nmax', str(opt.candidates), '-complex', '-models'], cwd=directory,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        e3, poi, ligand, _ = prepared
        rank = slot % opt.candidates+1
        candidate = load_assembly_candidate(directory/f'model_{rank}.pdb', None,
            e3['coords'], poi['coords'], ligand, candidate_id=entry['case'], rank=rank,
            protein1_path=entry['protein1'], protein2_path=entry['protein2'])
        candidate = preserve_rng(generate_joint_matcha_ligand)(expert, e3, poi, ligand, candidate,
            seed=protocol['slot_seed'], steps_per_stage=opt.generator_steps_per_stage)
        candidate = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in candidate.items()}
        atomic_torch_save({'protocol': protocol, 'candidate': candidate}, path)
        return candidate, path, sha256_file(path)


def model_for(opt, expert, input_dim, payload=None):
    config = (dict(payload['model_config']) if payload else production_model_config(
        opt.node_dim, input_dim, True, expert_backend='matcha',
        expert_scalar_dim=expert.node_scalar_dim, expert_vector_dim=expert.node_vector_dim,
        candidate_driven=True, candidate_expert_frame_policy=expert.frame_policy,
        motion_contract=getattr(opt, 'motion_contract', None) or FLOW_MOTION_CONTRACT))
    if payload and getattr(opt, 'weights_from', None):
        contract = getattr(opt, 'motion_contract', None) or FLOW_MOTION_CONTRACT
        if contract == FLOW_MOTION_CONTRACT:
            config.pop('motion_contract', None)
        else:
            config['motion_contract'] = contract
    if (input_dim != config['input_dim'] or not config.get('candidate_driven')
            or config.get('candidate_geometry_contract') != CANDIDATE_GEOMETRY_CONTRACT
            or config.get('candidate_quality_contract') != CANDIDATE_QUALITY_CONTRACT
            or config.get('candidate_condition_contract') != CANDIDATE_CONDITION_CONTRACT
            or config.get('candidate_reference_contract') != CANDIDATE_REFERENCE_CONTRACT
            or config.get('candidate_task_contract') != CANDIDATE_TASK_CONTRACT):
        raise ValueError('checkpoint is not a compatible candidate-driven model')
    if payload and (payload['format'] != CANDIDATE_FRAMEWORK_VERSION
                    or payload['expert_identity'] != expert.provenance):
        raise ValueError('checkpoint format or frozen expert identity differs')
    model = TernaryAssemblyModel(expert, TernaryAssembly(**config), expert_input_grad=True,
                                 checkpoint_ternary=True).to(opt.device)
    if payload:
        model.trifuse.load_state_dict(payload['model_state'], strict=True)
    return model


def train(opt, entries, expert, score_args):
    global ACTIVE_PREFETCH
    training_started = time.monotonic()
    if not opt.resume and (opt.output/'progress.pt').exists():
        raise ValueError('training output already contains a checkpoint; use --resume or a new directory')
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    clusters = _load_index(opt.index)[1]
    # --limit also limits materialized cases in qualification; remove empty
    # clusters through exactly the same cached-only membership filtering.
    total_cases = len(entries)
    if not clusters:
        entries = entries[rank::world]
    if not entries:
        raise ValueError('each training rank requires at least one case')
    payload = read_payload(opt.resume, opt.device)[0] if opt.resume else None
    admission = getattr(opt, 'data_admission', None)
    available = set(admission['available_cases']) if admission else None
    sampling_contract = 'deepternary_cluster_80_20_v1' if clusters else 'flat_entries_v1'
    changing_sampling = bool(payload and payload.get('sampling_contract', 'flat_entries_v1') != sampling_contract)
    if changing_sampling and not opt.adopt_cluster_sampling:
        raise ValueError('flat checkpoint needs explicit --adopt-cluster-sampling; cursor cannot be reused')
    owner = None
    if clusters:
        _, initial_shards, owner = cluster_epoch_plan(entries, clusters, available, opt.seed, 0, world)
        if any(not shard for shard in initial_shards):
            raise ValueError('each rank needs a materialized training cluster')
        owned_indices = [i for i, e in enumerate(entries) if owner.get(e['case']) == rank]
        total_cases = sum(map(len, initial_shards))
    else:
        owned_indices = list(range(len(entries)))
    if payload and payload.get('data_admission') is not None and admission is None:
        raise ValueError('resume cannot disable pinned cached-only admission')
    if payload and admission:
        previous = payload.get('data_admission')
        if previous is None and not opt.adopt_cached_only_resume:
            raise ValueError('old checkpoint needs explicit --adopt-cached-only-resume')
        if previous is not None and previous != admission:
            raise ValueError('resume cached-only data admission changed')
    if payload and payload.get('world_size', 1) != world:
        raise ValueError('resume requires the same distributed world size')
    if payload and payload.get('retained_generator_sha256') is not None and not \
            set(payload['retained_generator_sha256']).issubset(
                set(opt.reuse_generator_sha256)):
        raise ValueError('resume retained generator identities changed')
    rank_state = payload['rank_states'][rank] if payload and 'rank_states' in payload else payload
    identity, tasks = {}, []
    for i in owned_indices:
        entry = entries[i]
        if not opt.online_cache_size:
            candidates, digest = load_bank(opt, entry, expert)
            identity[entry['case']] = digest
        tasks.append(i)
    if opt.online_cache_size and rank_state:
        identity = rank_state['bank_identity'].copy()
        for path, digest in identity.items():
            if sha256_file(path) != digest:
                raise ValueError(f'resume online cache changed: {path}')
    # Never preload the full dataset onto CUDA. Retain at most two cases.
    case_cache = OrderedDict()

    def current_case(index):
        if index not in case_cache:
            if len(case_cache) >= 2:
                case_cache.popitem(last=False)
            entry = entries[index]
            case_cache[index] = (load_case(opt, entry, expert, score_args),
                                 None if opt.online_cache_size else load_bank(opt, entry, expert)[0])
        case_cache.move_to_end(index)
        return case_cache[index]
    transfer, transfer_digest = read_payload(opt.weights_from, opt.device) if opt.weights_from else (None, None)
    if payload and (rank_state['bank_identity'] != identity or payload['steps'] != opt.steps or payload['seed'] != opt.seed):
        raise ValueError('resume bank, rollout steps or seed differs')
    if payload and payload['training_index_sha256'] != sha256_file(opt.index):
        raise ValueError('resume training index differs')
    training_options = {'quality_weight': .1, 'steric_weight': .05,
                        'endpoint_objective': 'se3_sqrt_error_no_degradation_pool_v6',
                        'update_unit': 'complete_case_candidate_pool_v1',
                        'quality_lr': opt.quality_lr,
                        'quality_ranking_weight': .1,
                        'sdpi_replay_weight': opt.sdpi_replay_weight,
                        'degradation_weight': 0.,
                        'quality_representation': CANDIDATE_QUALITY_CONTRACT,
                        'quality_target': 'native_crystal_log_geometry_v1',
                        'gradient_clip_norm': 10., 'node_dim': opt.node_dim,
                        'gradient_clip_contract': 'independent_motion_quality_groups_v1',
                        'near_target_frequency': opt.near_target_frequency,
                        'steric_contract': 'element_0p75_vdw_cross_and_ligand_gt3bonds_v1'}
    effective_motion_contract = (
        getattr(opt, 'motion_contract', None) or FLOW_MOTION_CONTRACT
        if opt.weights_from else
        payload['model_config'].get('motion_contract', FLOW_MOTION_CONTRACT)
        if payload else FLOW_MOTION_CONTRACT)
    if effective_motion_contract in (BOUNDED_REPAIR_MOTION_CONTRACT,
                                     QUALITY_GATED_REPAIR_MOTION_CONTRACT):
        training_options.update(
            motion_contract=effective_motion_contract,
            repair_action_target_contract='reachable_endpoint_step_target_v1',
            repair_limits=dict(BOUNDED_CORRECTION_LIMITS),
            repair_state_noise=dict(CORRECTION_STATE_NOISE))
    if effective_motion_contract == QUALITY_GATED_REPAIR_MOTION_CONTRACT:
        training_options.update(
            endpoint_objective='quality_gated_bounded_repair_v1',
            quality_gate_contract='initially_capped_predicted_geometry_error_v1',
            quality_gate_error_scale=QUALITY_GATE_ERROR_SCALE,
            quality_state_supervision='initial_and_terminal_v1')
    if opt.online_cache_size:
        training_options.update(online_cache_size=opt.online_cache_size,
            online_cache_contract='fill_new_then_uniform_fixed_slots_v1',
            hdock_candidates=opt.candidates, generator_steps_per_stage=opt.generator_steps_per_stage)
    if payload and payload['training_options'] != training_options:
        raise ValueError('resume model width or training objective differs')
    initial_data = None
    for i in owned_indices:
        entry = entries[i]
        if available is not None and entry['case'] not in available:
            continue
        try:
            initial_data = current_case(i)[0]
            break
        except FeatureCacheUnavailable:
            continue
    if initial_data is None:
        raise ValueError('training rank has no cached-only entries')
    model = model_for(opt, expert, initial_data[3], payload or transfer)
    initial_data = None
    feature_identity = (rank_state['feature_identity'].copy() if rank_state else {})
    for case, digest in feature_identity.items():
        if sha256_file(Path(opt.cache_dir)/(case+'.pt')) != digest:
            raise ValueError(f'resume feature cache identity differs: {case}')
    trainer = AssemblyTrainer(model, SDPIFlow(inference_steps=opt.steps), lr=opt.lr,
        device=opt.device, residual_weight=0., steric_loss_weight=.05, effective_batch_size=1)
    for group in trainer.optimizer.param_groups:
        if group['name'] == 'candidate_quality':
            group['lr'] = opt.quality_lr
    epoch, position, updates = 0, 0, 0
    samples_seen = payload.get('samples_seen', 0) if payload else 0
    if payload:
        trainer.optimizer.load_state_dict(payload['optimizer_state'])
        restore_rng_state(rank_state['rng_state'])
        epoch, position, updates = payload['epoch'], payload['position'], payload['updates']
        if changing_sampling:
            epoch, position = 0, 0
    sampling_origin = (payload.get('sampling_origin') if payload else None)
    if changing_sampling:
        sampling_origin = {'previous_checkpoint_sha256': sha256_file(opt.resume),
            'previous_sampling_contract': payload.get('sampling_contract', 'flat_entries_v1'),
            'previous_epoch': payload['epoch'], 'previous_position': payload['position'],
            'updates_retained': updates, 'new_epoch': 0, 'new_position': 0}
    cache_visits = (rank_state.get('cache_visits', {}).copy() if rank_state and not changing_sampling else {})
    if changing_sampling:
        # Previously generated slots remain immutable. Count only checkpointed
        # slots; an uncommitted pending slot is reused by the resumed visit.
        for state in payload.get('rank_states', [payload]):
            for path in state['bank_identity']:
                file = Path(path)
                if file.name.startswith('online_slot_') and file.suffix == '.pt':
                    number = int(file.stem.rsplit('_', 1)[1])+1
                    cache_visits[file.parent.name] = max(cache_visits.get(file.parent.name, 0), number)
    model.train()
    parameters = list(model.trifuse.parameters())
    synchronizer = GradientSynchronizer(opt.device) if world > 1 else None
    if world > 1:
        for parameter in parameters:
            dist.broadcast(parameter.data, 0)
    if not payload:
        torch.manual_seed(opt.seed+rank)
        np.random.seed(opt.seed+rank)
        random.seed(opt.seed+rank)
    # CPU docking runs ahead of GPU training, with stable case ownership.
    # Qualification does not leave a dataset-sized queue running behind it.
    prefetch = None
    if opt.online_cache_size and opt.hdock_workers > 1 and not opt.max_updates:
        prefetch = ThreadPoolExecutor(max_workers=opt.hdock_workers)
        ACTIVE_PREFETCH = prefetch
    rounds = (total_cases+world-1)//world
    while epoch < opt.epochs:
        if clusters:
            selected, shards, _ = cluster_epoch_plan(entries, clusters, available, opt.seed, epoch, world)
            epoch_cases = [e['case'] for e in selected]
            permutation = shards[rank]
            rounds = max(map(len, shards))
            if payload and not changing_sampling and epoch == payload['epoch'] and position and \
                    payload.get('epoch_cases') != epoch_cases:
                raise ValueError('resumed cluster choices differ from checkpoint')
        else:
            permutation = np.random.RandomState(opt.seed+epoch).permutation(len(tasks))
            epoch_cases = None
        if prefetch:
            futures = {entries[i]['case']: prefetch.submit(docking_job, opt, entries[i]) for i in permutation
                       if available is None or entries[i]['case'] in available}
        for cursor in range(position, rounds):
            update_started = time.monotonic()
            trainer.optimizer.zero_grad(set_to_none=True)
            active = cursor < len(permutation)
            error = None
            oom = False
            record = None
            try:
                if active:
                    i = permutation[cursor] if clusters else tasks[permutation[cursor]]
                    entry = entries[i]
                    if available is not None and entry['case'] not in available:
                        raise FeatureCacheUnavailable('excluded by pinned cached-only admission')
                    prepared, candidates = current_case(i)
                    e3, poi, ligand, _ = prepared
                    feature_identity[entry['case']] = sha256_file(Path(opt.cache_dir)/(entry['case']+'.pt'))
                    if opt.online_cache_size:
                        if prefetch:
                            futures[entry['case']].result()
                        visit = cache_visits.get(entry['case'], 0) if clusters else epoch
                        candidate, path, digest = online_candidate(opt, entry, expert, prepared, visit)
                        cache_visits[entry['case']] = visit+1
                        identity[str(path)] = digest
                        candidates = [candidate]
                    record = backward_case(opt, trainer, model, entry, prepared, candidates, updates)
                    if opt.online_cache_size:
                        record['online_cache_slot'] = path.name
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('nonfinite candidate training gradient; update aborted')
            except Exception as exc:
                known = (isinstance(exc, ConformerGenerationFailure) or
                    isinstance(exc, ValueError) and str(exc) ==
                        'full-ligand ETKDG embedding failed; no native-pose fallback')
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    trainer.optimizer.zero_grad(set_to_none=True)
                    active = False
                    oom = True
                    record = {'case': entry['case'], 'skipped': True,
                              'reason': str(exc), 'category': 'cuda_oom'}
                    print(f'SKIP OOM {entry["case"]}: discarded this batch gradient', flush=True)
                elif isinstance(exc, FeatureCacheUnavailable):
                    trainer.optimizer.zero_grad(set_to_none=True)
                    active = False
                    record = {'case': entry['case'], 'skipped': True,
                              'reason': str(exc), 'category': 'feature_cache_unavailable'}
                    print(f'SKIP CACHE {entry["case"]}: {exc}', flush=True)
                elif isinstance(exc, EndpointProjectionError):
                    # A sampled start can make the crystal endpoint unreachable
                    # under the rigid-plus-torsion update contract. Discard the
                    # whole local case gradient (including any earlier pool
                    # candidates) and let the synchronized update proceed with
                    # the other ranks. Other ValueErrors remain fatal.
                    trainer.optimizer.zero_grad(set_to_none=True)
                    active = False
                    diagnostics = getattr(exc, 'diagnostics', {})
                    projection = diagnostics.get('report', {})
                    torsion = projection.get('torsion_optimization', {})
                    record = {
                        'case': entry['case'], 'skipped': True,
                        'category': 'unreachable_endpoint_projection',
                        'reason': 'sampled target is not reachable by the configured rigid/torsion update',
                        'projection_rmsd_angstrom': projection.get(
                            'ligand_projection_rmsd_angstrom'),
                        'solver': torsion.get('solver'),
                        'solver_status': torsion.get('status'),
                        'solver_nfev': torsion.get('nfev'),
                    }
                    print(f'SKIP PROJECTION {entry["case"]}: '
                          f'rmsd={record["projection_rmsd_angstrom"]} A, '
                          f'nfev={record["solver_nfev"]}', flush=True)
                elif known and opt.online_cache_size:
                    path = mark_conformer_failure(opt, entry, expert, str(exc))
                    identity[str(path)] = sha256_file(path)
                    trainer.optimizer.zero_grad(set_to_none=True)
                    active = False
                    record = {'case': entry['case'], 'skipped': True,
                              'reason': str(exc), 'retry': False}
                    print(f'SKIP ETKDG {entry["case"]}: {exc}', flush=True)
                else:
                    error = exc
            if oom:
                # The exception traceback has now been released. Drop cached
                # case tensors before allocating NCCL reduction buckets.
                case_cache.clear()
                prepared = e3 = poi = ligand = candidates = candidate = None
                gc.collect()
                torch.cuda.empty_cache()
            if synchronizer:
                synchronizer.check_error(error)
                global_cases = synchronizer(parameters, int(active))
            elif error:
                raise error
            else:
                global_cases = int(active)
            norms = {}
            if global_cases:
                norms = {group['name']: float(torch.nn.utils.clip_grad_norm_(
                    group['params'], 10., error_if_nonfinite=True)) for group in trainer.optimizer.param_groups}
                trainer.optimizer.step()
            agreement = None
            if world > 1 and opt.max_updates:
                flat = torch.cat([p.detach().flatten() for p in parameters])
                reference = flat.clone()
                dist.broadcast(reference, 0)
                difference = (flat-reference).abs().max()
                dist.all_reduce(difference, op=dist.ReduceOp.MAX)
                agreement = float(difference)
                if agreement != 0.:
                    raise RuntimeError('distributed optimizer replicas diverged')
            updates += int(global_cases > 0)
            samples_seen += global_cases
            next_epoch, next_position = (epoch+1, 0) if cursor+1 == rounds else (epoch, cursor+1)
            local_state = {'bank_identity': identity, 'feature_identity': feature_identity,
                           'rng_state': capture_rng_state(), 'record': record, 'cache_visits': cache_visits}
            states = [None]*world
            if world > 1:
                dist.all_gather_object(states, local_state)
            else:
                states[0] = local_state
            if rank == 0:
                checkpoint = {'format': CANDIDATE_FRAMEWORK_VERSION,
                    'model_config': model.trifuse.config, 'model_state': model.trifuse.state_dict(),
                    'expert_identity': expert.provenance, 'optimizer_state': trainer.optimizer.state_dict(),
                    'bank_identity': identity, 'steps': opt.steps, 'seed': opt.seed,
                    'training_index_sha256': sha256_file(opt.index), 'training_options': training_options,
                    'weights_only_origin': (payload.get('weights_only_origin') if payload else
                        {'checkpoint_sha256': transfer_digest, 'updates': transfer['updates'],
                         'source_objective': transfer['training_options']['endpoint_objective'],
                         'migration_origin': transfer.get('migration_origin')} if transfer else None),
                    'feature_identity': feature_identity, 'epoch': next_epoch,
                    'position': next_position, 'updates': updates, 'rng_state': states[0]['rng_state'],
                    'world_size': world, 'rank_states': states, 'samples_seen': samples_seen,
                    'data_admission': admission,
                    'retained_generator_sha256': opt.reuse_generator_sha256,
                    'sampling_contract': sampling_contract, 'sampling_origin': sampling_origin,
                    'epoch_cases': epoch_cases}
                atomic_torch_save(checkpoint, opt.output/'progress.pt')
                if next_position == 0:
                    atomic_torch_save(checkpoint, opt.output/f'checkpoint_epoch{next_epoch}.pt')
                summary = {'epoch': epoch, 'update': updates, 'global_cases': global_cases,
                    'ranks': [s['record'] for s in states], 'gradient_norms': norms,
                    'seconds': time.monotonic()-update_started,
                    'qualification_parameter_difference': agreement,
                    'sampling_contract': sampling_contract,
                    'epoch_step': cursor+1, 'epoch_steps': rounds,
                    'epoch_sample_count': total_cases,
                    'phase_update': updates-(sampling_origin['updates_retained'] if sampling_origin else 0)}
                with (opt.output/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(summary)+'\n')
                print(json.dumps(summary), flush=True)
            if world > 1:
                dist.barrier()
            stop = torch.tensor(int(bool((opt.max_updates and updates >= opt.max_updates) or
                (opt.max_hours and time.monotonic()-training_started >= opt.max_hours*3600))), device=opt.device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop):
                finish_prefetch(prefetch)
                return
        epoch += 1
        position = 0
    if prefetch:
        finish_prefetch(prefetch)


def backward_case(opt, trainer, model, entry, prepared, candidates, updates):
    """Accumulate one real case before global sample-weighted reduction."""
    e3, poi, ligand, _ = prepared
    features, targets, records = [], [], []
    for j, candidate in enumerate(candidates):
        near = ({'near_target_zero': (updates // opt.near_target_frequency) % 2 == 0,
                 'near_target_start_t': 1.}
                if j == 0 and opt.near_target_frequency and updates % opt.near_target_frequency == 0 else {})
        result = trainer.compute_loss(e3, poi, ligand, case=entry['case'], source='hdock_matcha', task_mode='refine',
            coupling_override={'initial_state': {'ligand': candidate['ligand'],
                'poi': candidate['poi'], 'body_frame': candidate['rotation']}}, **near)
        # Accumulate the complete pool without retaining five rollout graphs.
        ((1-opt.sdpi_replay_weight)*result['loss_total']/len(candidates)).backward()
        features.append(result['candidate_quality_features'])
        targets.append(result['candidate_quality_target'])
        records.append({'candidate': j, 'task_mode': 'refine', 'source': 'hdock_matcha',
            'loss': float(result['loss_total'].detach()),
            'components': {key: value for key, value in result.items()
                if key.startswith('loss_') and key != 'loss_total'},
            'target_projection_rmsd_angstrom': result.get('target_projection_rmsd_angstrom')})
    replay_record = None
    if opt.sdpi_replay_weight:
        replay = trainer.compute_loss(e3, poi, ligand, case=entry['case'], source='sdpi', task_mode='assemble')
        (opt.sdpi_replay_weight*replay['loss_total']).backward()
        replay_record = {key: value for key, value in replay.items()
                         if key.startswith('loss_') and key != 'loss_total'}
        replay_record.update(task_mode='assemble', source='sdpi')
    predicted = torch.nn.functional.softplus(
        model.trifuse.controller.quality(torch.cat(features))).mean(-1)
    target = torch.cat(targets).mean(-1)
    delta = target[:, None]-target[None, :]
    pairs = torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1) & (delta.abs() > .1)
    difference = predicted[:, None]-predicted[None, :]
    ranking_loss = (torch.nn.functional.softplus(
        delta.abs().clamp_max(2.)-delta.sign()*difference)[pairs].mean()
        if pairs.any() else predicted.sum()*0.)
    (.1*ranking_loss).backward()
    pool_loss = sum(row['loss'] for row in records)/len(records)
    replay_loss = replay_record['loss_score'] if replay_record is not None else 0.
    record = {'case': entry['case'],
        'candidates': records, 'sdpi_replay': replay_record,
        'rollouts': len(candidates)+int(bool(opt.sdpi_replay_weight)),
        'loss': (1-opt.sdpi_replay_weight)*pool_loss+opt.sdpi_replay_weight*replay_loss+.1*float(ranking_loss.detach()),
        'loss_contract': 'weighted_pool_replay_ranking_v1',
        'candidate_pool_loss': pool_loss, 'sdpi_replay_loss': replay_loss,
        'quality_ranking_loss': float(ranking_loss.detach())}
    return record


def predict(opt, entries, expert, score_args):
    payload, checkpoint_digest = read_payload(opt.checkpoint, opt.device)
    if payload['training_options'].get('quality_target') != 'native_crystal_log_geometry_v1':
        raise ValueError('retired reachable-target scorer; train the crystal-quality framework')
    if payload['steps'] != opt.steps:
        raise ValueError('deployment must use the trained refinement step count')
    if not payload.get('updates', 0):
        raise ValueError('prediction requires a trained candidate-framework checkpoint')
    model = None
    for entry in entries:
        e3, poi, ligand, input_dim = load_case(opt, entry, expert, score_args)
        candidates, bank_digest = load_bank(opt, entry, expert)
        if model is None:
            model = model_for(opt, expert, input_dim, payload)
        predictions = model.refine_candidates(e3, poi, ligand, candidates, steps=opt.steps)
        scores = [float(p['candidate_score']) for p in predictions]
        learned_winner = int(np.argmax(scores)) if payload.get('updates', 0) else None
        winner = (learned_winner if opt.selection == 'learned' else
                  min(range(len(predictions)), key=lambda i:
                      predictions[i]['provenance']['protein_candidate_rank']))
        directory = opt.output/entry['case']
        directory.mkdir(parents=True, exist_ok=True)
        selection_path = directory/'selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text())['checkpoint_sha256'] != checkpoint_digest:
            raise ValueError('prediction output belongs to another checkpoint; use a new directory')
        atomic_torch_save([{key: value.cpu() if torch.is_tensor(value) else value
                           for key, value in p.items()} for p in predictions], directory/'predictions.pt')
        atomic_json({'case': entry['case'], 'candidate_scores': scores, 'selected_index': winner,
            'task_mode': 'refine',
            'selection_policy': opt.selection, 'experimental_learned_index': learned_winner,
            'quality_trained_updates': payload.get('updates', 0), 'bank_sha256': bank_digest,
            'checkpoint_sha256': checkpoint_digest, 'native_used_for_selection': False,
            'quality_calibration_validated': False,
            'checkpoint_migration': payload.get('checkpoint_migration'),
            'input_identity': input_identity(entry),
            'implementation_sha256': {
                'assembly_pipeline.py': sha256_file(Path(__file__).resolve()),
                'assembly.py': sha256_file(ROOT/'tricoopdiff/models/assembly.py'),
            }}, directory/'selection.json')
        # Export physical structures before optional scoring. Predictions and
        # winner were already fixed without access to a native complex.
        from tricoopdiff.metrics.dockq import (
            _records, _featurized_ca_coords, _fit_rigid, _replace_coords,
            compute_fullatom_dockq,
        )
        p1_lines, _, _ = _records(entry['protein1'], 'ATOM')
        p2_lines, p2_indices, p2_xyz = _records(entry['protein2'], 'ATOM')
        ca1, ca2 = _featurized_ca_coords(entry['protein1']), _featurized_ca_coords(entry['protein2'])
        frame_r, frame_t = _fit_rigid(e3['coords'][0].cpu().numpy(), ca1)
        metrics = []
        for index, prediction in enumerate(predictions):
            lig_xyz = prediction['ligand'][0].cpu().numpy() @ frame_r + frame_t
            poi_xyz = prediction['poi'][0].cpu().numpy() @ frame_r + frame_t
            rotation = prediction['rotation'][0].cpu().numpy().T @ frame_r
            translation = poi_xyz.mean(0)-ca2.mean(0) @ rotation
            protein2 = _replace_coords(p2_lines, p2_indices, p2_xyz @ rotation+translation, chain='B')
            mol = ligand['mol']
            mol = Chem.Mol(mol) if isinstance(mol, bytes) else Chem.Mol(mol)
            mol.RemoveAllConformers()
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for atom, point in enumerate(lig_xyz):
                conformer.SetAtomPosition(atom, point.tolist())
            mol.AddConformer(conformer)
            with Chem.SDWriter(str(directory/f'candidate{index+1}_ligand.sdf')) as writer:
                writer.write(mol)
            ligand_lines = [l[:21]+'C'+l[22:] for l in Chem.MolToPDBBlock(mol).splitlines()
                            if l.startswith(('ATOM', 'HETATM'))]
            records = [l[:21]+'A'+l[22:] for l in p1_lines if l.startswith('ATOM')]
            records += [l for l in protein2 if l.startswith('ATOM')]+ligand_lines
            (directory/f'candidate{index+1}.pdb').write_text('\n'.join(records)+'\nEND\n')
            if opt.native_root:
                native_path = opt.native_root/entry['case']/'gt_complex.pdb'
                row = compute_fullatom_dockq(prediction['ligand'][0].cpu().numpy(),
                    prediction['poi'][0].cpu().numpy(), e3['coords'][0].cpu().numpy(),
                    entry['protein1'], entry['protein2'], entry['ligand'], native_path, ROOT/'DeepTernary',
                    pred_poi_rotation=prediction['rotation'][0].cpu().numpy())
                initial = candidates[index]
                before = compute_fullatom_dockq(initial['ligand'][0].cpu().numpy(),
                    initial['poi'][0].cpu().numpy(), e3['coords'][0].cpu().numpy(),
                    entry['protein1'], entry['protein2'], entry['ligand'], native_path, ROOT/'DeepTernary',
                    pred_poi_rotation=initial['rotation'][0].cpu().numpy())
                initial_rmsd = (initial['ligand'].to(ligand['coords'])-ligand['coords']).square().sum(-1).mean().sqrt()
                rmsd = torch.linalg.vector_norm(prediction['ligand']-ligand['coords'], dim=-1).square().mean().sqrt()
                row.update(candidate_index=index, ligand_crystal_rmsd_angstrom=float(rmsd),
                           initial_dockq=before['dockq'], dockq_change=row['dockq']-before['dockq'],
                           initial_ligand_crystal_rmsd_angstrom=float(initial_rmsd),
                           ligand_rmsd_change_angstrom=float(rmsd-initial_rmsd),
                           protein_rotation_to_indexed_target_deg=float(np.rad2deg(np.arccos(np.clip(
                               (prediction['rotation'][0].cpu().numpy().trace()-1)/2, -1, 1)))),
                           initial_protein_rotation_to_indexed_target_deg=float(np.rad2deg(np.arccos(np.clip(
                               (initial['rotation'][0].cpu().numpy().trace()-1)/2, -1, 1)))),
                           protein_center_to_indexed_target_angstrom=float(torch.linalg.vector_norm(
                               prediction['poi'].mean(1)-poi['coords'].mean(1))),
                           ligand_rmsd_protocol='indexed_atom_order_e3_frame_no_ligand_alignment')
                metrics.append(row)
        if opt.native_root:
            atomic_json({'case': entry['case'], 'selected_index': winner,
                'selected': metrics[winner] if winner is not None else None,
                'candidates': metrics, 'native_used_for_selection': False}, directory/'metrics.json')
        print(f"PREDICTED {entry['case']} selected={winner}", flush=True)


@torch.no_grad()
def baseline(opt, entries, expert, score_args):
    """Score frozen HDOCK + joint Matcha candidates without ternary refinement."""
    from tricoopdiff.metrics.dockq import (
        _featurized_ca_coords, _fit_rigid, compute_fullatom_pose_dockq,
    )

    cases = []
    for entry in entries:
        directory = opt.output/entry['case']
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory/'metrics.json'
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved.get('input_identity') != input_identity(entry):
                raise ValueError(f'baseline input changed for {entry["case"]}')
            cases.append(saved)
            print(f'REUSED BASELINE {entry["case"]}', flush=True)
            continue
        try:
            e3, poi, ligand, _ = load_case(opt, entry, expert, score_args)
            candidates, bank_digest = load_bank(opt, entry, expert)
            native_path = entry['complex']
            if not Path(native_path).is_file():
                raise FileNotFoundError(native_path)
            frame_r, frame_t = _fit_rigid(e3['coords'][0].cpu().numpy(),
                                         _featurized_ca_coords(entry['protein1']))
            ranked = sorted(candidates, key=lambda value: value['provenance']['protein_candidate_rank'])
            metrics = []
            for candidate in ranked:
                rank = candidate['provenance']['protein_candidate_rank']
                ligand_xyz = candidate['ligand'][0].numpy() @ frame_r + frame_t
                molecule = Chem.Mol(ligand['mol'])
                molecule.RemoveAllConformers()
                conformer = Chem.Conformer(molecule.GetNumAtoms())
                for atom, point in enumerate(ligand_xyz):
                    conformer.SetAtomPosition(atom, point.tolist())
                molecule.AddConformer(conformer)
                ligand_path = directory/f'candidate{rank}_ligand.sdf'
                with Chem.SDWriter(str(ligand_path)) as writer:
                    writer.write(molecule)
                row = compute_fullatom_pose_dockq(
                    opt.bank/entry['case']/f'model_{rank}.pdb', native_path,
                    ROOT/'DeepTernary')
                row.update(rank=rank, ligand_rmsd_angstrom=float(torch.linalg.vector_norm(
                    candidate['ligand'].to(ligand['coords'])-ligand['coords'], dim=-1)
                    .square().mean().sqrt()), ligand_sdf=str(ligand_path),
                    protein_pdb=str(opt.bank/entry['case']/f'model_{rank}.pdb'))
                metrics.append(row)
            result = {'case': entry['case'], 'status': 'complete', 'input_identity': input_identity(entry),
                'bank_sha256': bank_digest, 'selection': 'HDOCK rank 1',
                'selected': metrics[0], 'oracle_top5': max(metrics, key=lambda row: row['dockq']),
                'candidates': metrics, 'native_used_for_selection': False,
                'oracle_uses_native': True,
                'ligand_rmsd_protocol': 'indexed_atom_order_e3_frame_no_ligand_alignment'}
        except Exception as error:
            result = {'case': entry['case'], 'status': 'failed',
                'input_identity': input_identity(entry),
                'error': f'{type(error).__name__}: {error}'}
        atomic_json(result, result_path)
        cases.append(result)
        print(f'BASELINE {entry["case"]} {result["status"]}', flush=True)
    complete = [case for case in cases if case['status'] == 'complete']
    summary = {'protocol': 'unbound_hdock_top5_joint_matcha_direct_fullatom_v2',
        'requested': len(entries), 'completed': len(complete),
        'failed': [case['case'] for case in cases if case['status'] != 'complete'],
        'top1_mean_dockq': float(np.mean([case['selected']['dockq'] for case in complete])) if complete else None,
        'oracle_top5_mean_dockq': float(np.mean([case['oracle_top5']['dockq'] for case in complete])) if complete else None,
        'top1_mean_ligand_rmsd_angstrom': float(np.mean([
            case['selected']['ligand_rmsd_angstrom'] for case in complete])) if complete else None,
        'cases': cases}
    atomic_json(summary, opt.output/'summary.json')
    print(f'BASELINE COMPLETE {len(complete)}/{len(entries)}', flush=True)


def migrate(opt, entries, expert, score_args):
    """Convert the selected SDPI EMA once; publish only after rollout parity."""
    from tricoopdiff.initialization import TernaryInitializer
    from tricoopdiff.runtime import evaluation_context
    from tricoopdiff.models.assembly import LEGACY_TORSION_REPRESENTATION

    parent, digest = read_payload(opt.baseline, opt.device)
    if parent['training_state']['epoch'] != 16 or opt.steps != 20:
        raise ValueError('this migration requires epoch16 EMA and the original 20-step execution')
    for key in ('stage_weights_sha256', 'source_tree_sha256', 'sequence_encoder_sha256',
                'torsion_contract', 'node_vector_contract', 'time_contract'):
        if parent['expert_base_identity'].get(key) != expert.provenance.get(key):
            raise ValueError(f'pretrained expert evidence differs: {key}')
    config = parent['model_config']
    legacy = TernaryAssemblyModel(expert, TernaryAssembly(**config), expert_input_grad=True).to(opt.device)
    old_weights = parent['ema_trifuse_state_dict']
    legacy.trifuse.load_state_dict(old_weights, strict=True)
    updated = production_model_config(config['node_dim'], config['input_dim'], True,
        expert_backend='matcha', expert_scalar_dim=expert.node_scalar_dim,
        expert_vector_dim=expert.node_vector_dim, candidate_driven=True,
        candidate_expert_frame_policy=expert.frame_policy,
        torsion_representation_contract=config.get('torsion_representation_contract', LEGACY_TORSION_REPRESENTATION),
        protein_local_geometry_contract=config.get('protein_local_geometry_contract', 'none'))
    model = TernaryAssemblyModel(expert, TernaryAssembly(**updated), expert_input_grad=True).to(opt.device)
    weights = model.trifuse.state_dict()
    copied = []
    for key, value in old_weights.items():
        if key == 'controller.vector_input.weight':
            weights[key].zero_()
            weights[key][:, :value.shape[1]] = value
        else:
            if key not in weights or weights[key].shape != value.shape:
                raise ValueError(f'cannot preserve pretrained tensor: {key}')
            weights[key] = value
        copied.append(key)
    for key in weights.keys()-old_weights.keys():
        if not key.startswith('controller.quality.'):
            weights[key].zero_()
    model.trifuse.load_state_dict(weights, strict=True)
    flow = SDPIFlow(inference_steps=20)
    rows = []
    with evaluation_context(legacy), evaluation_context(model), torch.no_grad():
        for entry in entries:
            e3, poi, ligand, _ = load_case(opt, entry, expert, score_args)
            l0, p0, r0, _ = TernaryInitializer(no_torsion=False).initialize(
                e3['coords'][0], poi['coords'][0], ligand, seed=73)
            starts = [('sdpi73', l0, p0, r0, 'sdpi')]
            bank_path = opt.bank/entry['case']/'candidates.pt'
            if bank_path.exists():
                # Pose-only comparison, not accepting this bank for training.
                bank = read_payload(bank_path)[0]
                if bank['protocol']['inputs'] != input_identity(entry):
                    raise ValueError('comparison bank belongs to different inputs')
                starts.extend((f'candidate{i+1}', *(c[k].to(l0) for k in ('ligand', 'poi', 'rotation')),
                               'hdock_matcha') for i, c in enumerate(bank['candidates']))
            for name, l0, p0, r0, source in starts:
                before = [x.clone() for x in (l0, p0, r0)]
                after = [x.clone() for x in (l0, p0, r0)]
                reference = {**poi, 'candidate_reference': {
                    'ligand': l0.clone(), 'poi': p0.clone(), 'body': r0.clone()}}
                max_velocity = 0.
                for t, dt in flow.reverse_times(l0):
                    a = flow.forward_state(legacy, e3, poi, ligand, *before, t, dt, source=0)
                    b = flow.forward_state(model, e3, reference, ligand, *after, t, dt, source=source,
                        task_mode='assemble' if source == 'sdpi' else 'refine')
                    for branch in ('e3_ligand', 'poi_ligand'):
                        for field in ('tr', 'rot'):
                            max_velocity = max(max_velocity, float((a['normalized_velocity'][branch][field]-
                                b['normalized_velocity'][branch][field]).abs().max()))
                    if a['normalized_velocity']['torsion'].numel():
                        max_velocity = max(max_velocity, float((a['normalized_velocity']['torsion']-
                            b['normalized_velocity']['torsion']).abs().max()))
                    before = list(flow.step(*before, flow.velocities(a['normalized_velocity'], e3['coords'], poi['coords']), dt, ligand))
                    after = list(flow.step(*after, flow.velocities(b['normalized_velocity'], e3['coords'], poi['coords']), dt, ligand))
                difference = [float((a-b).abs().max()) for a, b in zip(before, after)]
                passed = max(difference) <= 1e-3 and max_velocity <= 1e-4
                rows.append({'case': entry['case'], 'start': name, 'max_normalized_velocity_difference': max_velocity,
                             'terminal_max_difference': difference, 'passed': passed})
                print(json.dumps(rows[-1]), flush=True)
    audit = {'source_sha256': digest, 'source_epoch': 16, 'source_weights': 'ema_trifuse_state_dict',
             'copied_tensors': len(copied), 'steps': 20, 'comparisons': rows,
             'passed': bool(rows) and all(row['passed'] for row in rows)}
    atomic_json(audit, opt.output/'migration_audit.json')
    if not audit['passed']:
        raise RuntimeError('migration changed the pretrained rollout; no checkpoint published')
    atomic_torch_save({'format': CANDIDATE_FRAMEWORK_VERSION, 'model_config': model.trifuse.config,
        'model_state': model.trifuse.state_dict(), 'expert_identity': expert.provenance,
        'steps': 20, 'updates': 0,
        'training_options': {'endpoint_objective': 'sdpi_epoch16_ema_behavior_preserved_v1'},
        'migration_origin': audit}, opt.output/'epoch16_ema_candidate.pt')


def main(argv=None):
    opt = arguments(argv)
    signal.signal(signal.SIGTERM, stop_cpu_jobs)
    signal.signal(signal.SIGINT, stop_cpu_jobs)
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.set_num_threads(opt.cpu_threads)
    if opt.cuda_memory_fraction is not None:
        if not str(opt.device).startswith('cuda'):
            raise ValueError('CUDA memory fraction requires a CUDA device')
        torch.cuda.set_per_process_memory_fraction(opt.cuda_memory_fraction)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if opt.cpu_budget:
        available = sorted(os.sched_getaffinity(0))
        if opt.cpu_budget > len(available) or opt.cpu_budget % world:
            raise ValueError('CPU budget exceeds affinity or is not divisible by world size')
        share = opt.cpu_budget//world
        if opt.cpu_threads+opt.hdock_workers > share:
            raise ValueError('CPU threads plus docking workers exceed the per-rank CPU budget')
        affinity = available[rank*share:(rank+1)*share]
        # Torch may have created intra-op threads before this point.
        for thread in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(thread.name), affinity)
            except ProcessLookupError:
                pass
    if world > 1:
        if opt.mode != 'train':
            raise ValueError('distributed launch is training-only')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        opt.device = f'cuda:{local_rank}'
        # A rank may be waiting for bounded CPU docking (up to four hours),
        # not a hung GPU collective. Preserve error handling beyond that bound.
        dist.init_process_group('nccl', timeout=timedelta(hours=5))
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    entries = _load_index(opt.index)[0]
    if opt.cases:
        entries_by_case = {entry['case']: entry for entry in entries}
        missing = sorted(set(opt.cases)-set(entries_by_case))
        if missing:
            raise ValueError(f'cases are absent from the supplied index: {missing}')
        entries = [entries_by_case[case] for case in dict.fromkeys(opt.cases)]
    if opt.limit:
        entries = entries[:opt.limit]
    if not entries or len({e['case'] for e in entries}) != len(entries):
        raise ValueError('index must contain nonempty unique cases')
    opt.output.mkdir(parents=True, exist_ok=True)
    opt.bank.mkdir(parents=True, exist_ok=True)
    lock_root = opt.bank if opt.mode == 'generate' else opt.output
    with ExitStack() as stack:
        if rank == 0:
            locks = {(lock_root, opt.mode+'.lock')}
            if opt.mode == 'generate' or (opt.mode == 'train' and opt.online_cache_size):
                locks.add((opt.bank, 'writer.lock'))
            if opt.mode == 'train':
                feature_root = Path(opt.cache_dir)
                feature_root.mkdir(parents=True, exist_ok=True)
                locks.add((feature_root, 'writer.lock'))
            for root, name in sorted(locks):
                lock = stack.enter_context((root/name).open('a'))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError('another process is writing this pipeline directory') from error
        if world > 1:
            dist.barrier()
        policy = opt.frame_policy
        source = opt.baseline if opt.mode == 'migrate' else (opt.resume or opt.weights_from or opt.checkpoint)
        if source:
            metadata = read_payload(source)[0]
            policy = (metadata['expert_base_identity'].get('frame_policy', 'legacy_argmax_v1')
                      if opt.mode == 'migrate' else
                      metadata['model_config'].get('candidate_expert_frame_policy', 'backbone_rigid_transport_v1'))
        expert = FrozenMatcha(opt.matcha_source_dir, opt.matcha_weights_dir, opt.device, frame_policy=policy)
        score_args = _load_score_model_args(opt.score_model_dir)
        if opt.mode == 'train' and opt.cached_only:
            objects = [cached_admission(opt, entries) if rank == 0 else None]
            if world > 1:
                dist.broadcast_object_list(objects, src=0)
            opt.data_admission = objects[0]
            if rank == 0:
                print(f'CACHED-ONLY available={len(opt.data_admission["available_cases"])} '
                      f'missing={len(opt.data_admission["missing_cache_cases"])} '
                      f'total={len(entries)}; signatures validated on read', flush=True)
        try:
            {'generate': generate, 'train': train, 'predict': predict,
             'baseline': baseline, 'migrate': migrate}[opt.mode](opt, entries, expert, score_args)
        finally:
            # ThreadPoolExecutor's atexit join otherwise hides a fatal training
            # exception behind hours of outstanding CPU prefetch work.
            finish_prefetch(ACTIVE_PREFETCH)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
