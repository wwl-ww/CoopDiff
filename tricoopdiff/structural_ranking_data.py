"""Strict archive adapter: labels never enter structural model inputs."""

import hashlib
import json
import math
from pathlib import Path

import torch

from .graphs import build_trifuse_graph
from .initial_selector import POOL_LABEL_VERSION
from .ranking_archive import ARCHIVE_VERSION, state_hash


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_graph(inputs, state):
    """Use candidate geometry, never template POI/ligand coordinates."""
    e, p, l = (inputs[name] for name in ('e3_data', 'poi_data', 'ligand_data'))
    graph = build_trifuse_graph(
        e['coords'][0], e['features'][0], state['poi'], p['features'][0],
        state['ligand'], l['features'][0], lig_edge_index=l['edge_index'])
    return {key: graph[key] for key in
            ('node_features', 'node_coords', 'node_types', 'lig_edge_index')}


def adapt_initial_pool(archive, labels=None):
    """All generated initial states, without consulting refinement outcomes."""
    if archive['protocol'] != ARCHIVE_VERSION or not archive['pool']:
        raise ValueError('nonempty compatible initial pool required')
    graphs, sources, identities, seen = [], [], [], set()
    for index, candidate in enumerate(archive['pool']):
        identity = tuple(candidate['candidate_id'])
        if (not identity or identity in seen or candidate['source'] not in ('sdpi', 'bitseed')
                or identity[0] != candidate['source']):
            raise ValueError('duplicate candidate identity or invalid actual source')
        if state_hash(candidate['state']) != candidate['state_sha256']:
            raise ValueError('initial pool state identity mismatch')
        seen.add(identity)
        graphs.append(state_graph(archive['input_graphs'], candidate['state']))
        sources.append(candidate['source']); identities.append((index, 'initial'))
    result = {'case': archive['case'], 'group': archive['group'], 'graphs': graphs, 'sources': sources,
            'stages': ['initial'] * len(graphs), 'identities': identities,
            'pool_size': len(graphs), 'labeled_pairs': 0, 'labeled_initials': 0}
    if labels is not None:
        rows = labels['rows']
        if (labels['case'] != archive['case'] or len(rows) != len(graphs)
                or [row['pool_index'] for row in rows] != list(range(len(graphs)))):
            raise ValueError('initial labels must cover the whole pool in identity order')
        values = []
        for row, candidate in zip(rows, archive['pool']):
            if (row['source'] != candidate['source']
                    or tuple(row['candidate_id']) != tuple(candidate['candidate_id'])
                    or row['state_sha256'] != candidate['state_sha256']):
                raise ValueError('initial label source/state identity mismatch')
            value = float(row['dockq']['dockq'])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid initial DockQ label')
            values.append(value)
        result.update(labels=torch.tensor(values), labeled_initials=len(values))
    return result


def dockq_components(label):
    """Return bounded DockQ decomposition, or None for old scalar-only labels."""
    keys = ('fnat', 'irms', 'lrms')
    if not any(key in label for key in keys):
        return None
    if not all(key in label for key in keys):
        raise ValueError('partial DockQ component labels')
    fnat, irms, lrms = (float(label[key]) for key in keys)
    if (not all(math.isfinite(x) for x in (fnat, irms, lrms))
            or not 0 <= fnat <= 1 or irms < 0 or lrms < 0):
        raise ValueError('invalid DockQ component labels')
    values = [fnat, (1. / math.hypot(1., irms / 1.5)) ** 2,
              (1. / math.hypot(1., lrms / 8.5)) ** 2]
    if abs(sum(values) / 3 - float(label['dockq'])) > 1e-5:
        raise ValueError('DockQ scalar disagrees with its components')
    return values


def adapt_case(archive, labels=None):
    """Adapt archived pairs; optional training labels stay outside graph inputs."""
    if archive['protocol'] != ARCHIVE_VERSION or (labels is not None and archive['case'] != labels['case']):
        raise ValueError('archive protocol/case mismatch')
    pairs = {p['pool_index']: p for p in archive['pairs']}
    truth = {p['pool_index']: p for p in labels['labels']} if labels is not None else None
    if (not pairs or len(pairs) != len(archive['pairs']) or (truth is not None
            and (len(truth) != len(labels['labels']) or pairs.keys() != truth.keys()))):
        raise ValueError('labels must cover each archived pair exactly once')
    inputs = archive['input_graphs']
    graphs, stages, sources, targets, identities = [], [], [], [], []
    components, component_mask = [], []
    for index, pair in sorted(pairs.items()):
        if type(index) is not int or not 0 <= index < len(archive['pool']):
            raise ValueError('invalid pool index')
        candidate = archive['pool'][index]
        target = truth[index] if truth is not None else None
        if (pair['source'] not in ('sdpi', 'bitseed')
                or pair['source'] != candidate['source']
                or tuple(pair['candidate_id']) != tuple(candidate['candidate_id'])
                or candidate['candidate_id'][0] != pair['source']
                or type(pair['valid']) is not bool
                or (target is not None and pair['valid'] != target['valid'])):
            raise ValueError('source, candidate identity or validity mismatch')
        initial = candidate['state']
        if state_hash(initial) != pair['initial_state_sha256'] or state_hash(initial) != candidate['state_sha256']:
            raise ValueError('initial state identity mismatch')
        if pair['valid']:
            if (pair['final'] is None or (target is not None and target['final'] is None)
                    or state_hash(pair['final']) != pair['final_state_sha256']):
                raise ValueError('valid final state missing or corrupted')
        elif (pair['final'] is not None or (target is not None and target['final'] is not None)
              or pair['final_state_sha256'] is not None):
            raise ValueError('failed trajectory must not supply a final state')
        for stage, state in [('initial', initial)] + ([('final', pair['final'])] if pair['valid'] else []):
            if target is not None:
                value = float(target[stage]['dockq'])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('invalid DockQ label')
                targets.append(value)
                terms = dockq_components(target[stage])
                components.append(terms if terms is not None else [0., 0., 0.])
                component_mask.append(terms is not None)
            graphs.append(state_graph(inputs, state))
            stages.append(stage); sources.append(pair['source'])
            identities.append((index, stage))
    result = {'case': archive['case'], 'group': archive['group'], 'graphs': graphs,
            'stages': stages, 'sources': sources,
            'identities': identities, 'pool_size': len(archive['pool']),
            'archived_pairs': len(pairs), 'labeled_pairs': len(pairs) if labels is not None else 0,
            'labeled_initials': len(pairs) if labels is not None else 0}
    if labels is not None:
        result['labels'] = torch.tensor(targets)
        result['component_targets'] = torch.tensor(components)
        result['component_mask'] = torch.tensor(component_mask, dtype=torch.bool)
    return result


class StructuralArchiveDataset:
    """Lazy trusted-local PT loading; verify seals before consuming labels."""

    def __init__(self, archives, labels, split, *, candidate_scope='archived-pairs'):
        if candidate_scope not in ('archived-pairs', 'initial-pool'):
            raise ValueError('invalid training candidate scope')
        self.candidate_scope = candidate_scope
        self.archives, self.labels = Path(archives), Path(labels)
        self.split = split
        path = self.archives / 'manifest.json'
        self.manifest = json.loads(path.read_text())
        self.manifest_digest = file_digest(path)
        label_manifest = json.loads((self.labels / 'manifest.json').read_text())
        if candidate_scope == 'initial-pool' and label_manifest.get('protocol') != POOL_LABEL_VERSION:
            raise ValueError('whole-pool initial label protocol required')
        self.label_pool_size = label_manifest.get('pool_size')
        self.label_manifest_digest = file_digest(self.labels / 'manifest.json')
        self.seals = {}
        if label_manifest['archive_manifest_sha256'] != self.manifest_digest:
            raise ValueError('label manifest does not match archive')
        self.cases = self.manifest['cases']
        if not self.cases or len(set(self.cases)) != len(self.cases):
            raise ValueError('empty or duplicate archive cases')

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        path = self.archives / f'{case}.pt'
        digest = file_digest(path)
        receipt = json.loads((self.archives / f'{case}.complete.json').read_text())
        label_path = self.labels / f'{case}.json'
        labels = json.loads(label_path.read_text())
        seal = {'archive_sha256': digest, 'labels_sha256': file_digest(label_path)}
        if case in self.seals and self.seals[case] != seal:
            raise ValueError('archive or labels changed after dataset audit')
        if receipt['archive_sha256'] != digest or labels['archive_sha256'] != digest:
            raise ValueError('archive seal mismatch')
        archive = torch.load(path, map_location='cpu', weights_only=False)
        if (archive['case'] != case or archive['split'] != self.split
                or archive['manifest'] != self.manifest):
            raise ValueError('archive split/manifest mismatch')
        if self.candidate_scope == 'initial-pool':
            if len(archive['pool']) != self.label_pool_size:
                raise ValueError('initial label pool-size contract mismatch')
            result = adapt_initial_pool(archive, labels)
        else:
            result = adapt_case(archive, labels)
        self.seals[case] = seal
        return result


class StructuralJointDataset:
    """One quality list: all initials plus available archived finals, no repeats."""

    candidate_scope = 'joint'

    def __init__(self, paired, initial):
        if (paired.candidate_scope != 'archived-pairs' or initial.candidate_scope != 'initial-pool'
                or paired.manifest_digest != initial.manifest_digest
                or paired.cases != initial.cases or paired.split != initial.split):
            raise ValueError('joint datasets must refer to the same archive and split')
        self.paired, self.initial = paired, initial
        self.cases, self.split, self.manifest = paired.cases, paired.split, paired.manifest
        self.manifest_digest = paired.manifest_digest
        self.label_manifest_digest = hashlib.sha256(json.dumps(
            [paired.label_manifest_digest, initial.label_manifest_digest]).encode()).hexdigest()

    @property
    def seals(self):
        return {case: {**seal, 'initial_labels_sha256': self.initial.seals[case]['labels_sha256']}
                for case, seal in self.paired.seals.items() if case in self.initial.seals}

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        paired, initial = self.paired[index], self.initial[index]
        case = self.cases[index]
        if self.paired.seals[case]['archive_sha256'] != self.initial.seals[case]['archive_sha256']:
            raise ValueError('archive changed between joint label reads')
        finals, selected = {}, set()
        for row, (pool_index, stage) in enumerate(paired['identities']):
            selected.add(pool_index)
            if stage == 'initial':
                if abs(float(paired['labels'][row]) - float(initial['labels'][pool_index])) > 1e-6:
                    raise ValueError('same initial state has inconsistent DockQ labels')
            else:
                finals[pool_index] = row
        result = {key: initial[key] for key in ('case', 'group', 'pool_size', 'labeled_initials')}
        for key in ('graphs', 'stages', 'sources', 'identities'):
            result[key] = []
        targets, eligible = [], []
        for pool_index in range(initial['pool_size']):
            rows = [(initial, pool_index)]
            if pool_index in finals:
                rows.append((paired, finals[pool_index]))
            for dataset, row in rows:
                for key in ('graphs', 'stages', 'sources', 'identities'):
                    result[key].append(dataset[key][row])
                targets.append(dataset['labels'][row])
                eligible.append(pool_index in selected)
        result.update(labels=torch.stack(targets), union_eligible=torch.tensor(eligible, dtype=torch.bool),
                      labeled_pairs=paired['labeled_pairs'], archived_pairs=paired['archived_pairs'])
        return result


def future_utility_case(case, *, target='final'):
    """Initial-only inputs supervised by observed final/Union utility.

    Union uses max(initial, final); final uses final DockQ, never improvement
    delta. Unrefined/failed candidates retain unknown utility, never fake zeros.
    """
    if target not in ('final', 'union'):
        raise ValueError('future target must be final or union')
    rows = {tuple(identity): i for i, identity in enumerate(case['identities'])}
    if len(rows) != len(case['identities']):
        raise ValueError('duplicate identities in future supervision')
    initial, final = [], []
    for (identity, stage), j in rows.items():
        if stage != 'final':
            continue
        i = rows.get((identity, 'initial'))
        if i is None or case['sources'][i] != case['sources'][j]:
            raise ValueError('future label lacks same-source initial state')
        initial.append(i)
        final.append(j)
    if not initial:
        return None
    result = {key: [case[key][i] for i in initial]
              for key in ('graphs', 'stages', 'sources', 'identities')}
    labels = case['labels'][final]
    if target == 'union':
        labels = torch.maximum(labels, case['labels'][initial])
    result.update(case=case['case'], labels=labels, ranking_task='future')
    return result


class StructuralArchiveCollection:
    """Ordered read-only shards; never rewrite or merge original manifests."""

    def __init__(self, datasets):
        if not datasets:
            raise ValueError('at least one archive shard required')
        self.datasets = list(datasets)
        self.candidate_scope = getattr(datasets[0], 'candidate_scope', 'archived-pairs')
        self.manifest = datasets[0].manifest
        self.cases, self.locations, pdbs = [], [], set()
        for shard_index, dataset in enumerate(datasets):
            if getattr(dataset, 'candidate_scope', 'archived-pairs') != self.candidate_scope:
                raise ValueError('mixed candidate scopes in archive collection')
            if dataset.split != datasets[0].split:
                raise ValueError('mixed split archive collection')
            for key in ('checkpoint_sha256', 'steps', 'pool', 'selection'):
                if dataset.manifest[key] != self.manifest[key]:
                    raise ValueError(f'archive shard protocol mismatch: {key}')
            for local_index, case in enumerate(dataset.cases):
                pdb = case.split('_')[0].upper()
                if pdb in pdbs:
                    raise ValueError('duplicate PDB in archive collection')
                pdbs.add(pdb)
                self.cases.append(case)
                self.locations.append((shard_index, local_index))
        self.split = datasets[0].split
        # Ordered corpus identities, not hashes of an invented merged file.
        self.manifest_digest = hashlib.sha256(json.dumps(
            [d.manifest_digest for d in datasets]).encode()).hexdigest()
        self.label_manifest_digest = hashlib.sha256(json.dumps(
            [d.label_manifest_digest for d in datasets]).encode()).hexdigest()

    @property
    def seals(self):
        return {case: seal for dataset in self.datasets for case, seal in dataset.seals.items()}

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        shard, local = self.locations[index]
        return self.datasets[shard][local]


def audit_datasets(train, val):
    """Reject protocol drift and PDB/group leakage before any optimization."""
    if getattr(train, 'candidate_scope', 'archived-pairs') != getattr(val, 'candidate_scope', 'archived-pairs'):
        raise ValueError('train/validation candidate-scope mismatch')
    for key in ('checkpoint_sha256', 'steps', 'pool'):
        if train.manifest[key] != val.manifest[key]:
            raise ValueError(f'generator protocol mismatch: {key}')
    selection_changed = train.manifest['selection'] != val.manifest['selection']
    if selection_changed:
        policies = (train.manifest['selection'], val.manifest['selection'])
        allowed = ('complete_existing_pool_v1', 'uniform_without_replacement_within_actual_source_balanced_v1')
        if any(policy not in allowed for policy in policies) or 'complete_existing_pool_v1' not in policies:
            raise ValueError('generator protocol mismatch: selection')
        # Historical collection used a random selector with walkers=pool size.
        # Its policy name may differ, but only exhaustive valid outcomes make
        # that difference irrelevant. Do not waive unknown generator versions.
        for key in ('expert_assets', 'diffusion_seed'):
            if not train.manifest.get(key) or train.manifest.get(key) != val.manifest.get(key):
                raise ValueError(f'complete-pool generator mismatch: {key}')
        for name in ('tricoopdiff/sampler.py', 'tricoopdiff/sdpi_flow.py',
                     'tricoopdiff/consensus.py', 'tricoopdiff/geometry_encoders.py',
                     'scripts/evaluation_runner.py'):
            digest = train.manifest.get('runtime_sha256', {}).get(name)
            if not digest or digest != val.manifest.get('runtime_sha256', {}).get(name):
                raise ValueError(f'complete-pool runtime mismatch: {name}')
    rows = []
    for dataset in (train, val):
        split_rows = []
        for i in range(len(dataset)):
            case = dataset[i]
            if selection_changed and (case.get('archived_pairs') != case['pool_size']
                                      or case['stages'].count('final') != case['pool_size']):
                raise ValueError('different collection policies require complete valid pool outcomes')
            split_rows.append({key: value for key, value in case.items() if key in
                               ('case', 'group', 'pool_size', 'labeled_pairs', 'labeled_initials')})
        rows.append(split_rows)
    for key in ('case', 'group'):
        if {r[key] for r in rows[0]} & {r[key] for r in rows[1]}:
            raise ValueError(f'train/validation {key} leakage')
    if ({r['case'].split('_')[0].upper() for r in rows[0]}
            & {r['case'].split('_')[0].upper() for r in rows[1]}):
        raise ValueError('train/validation PDB leakage')
    return [[{k: r[k] for k in ('case', 'group', 'pool_size', 'labeled_pairs', 'labeled_initials') if k in r}
             for r in split] for split in rows]
