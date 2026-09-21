"""Fixed-checkpoint diagnostics and train-only fitted CPU geometry baselines."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat  # noqa: E402,F401
import torch

from train_structural_ranker import ordering_metrics
from tricoopdiff.structural_ranker import StructuralQualityRanker
from tricoopdiff.structural_ranking_data import StructuralArchiveDataset, audit_datasets, file_digest


def summarize(case, scores):
    """Canonical order breaks score ties; labels are evaluation-only here."""
    result = {}
    labels = case['labels']
    for stage in ('initial', 'final', 'union'):
        ids = [i for i, s in enumerate(case['stages']) if stage == 'union' or s == stage]
        if not ids:
            continue
        order = sorted(ids, key=lambda i: (-float(scores[i]), i))
        groups = [(case['stages'][i], case['sources'][i]) for i in ids]
        row = {'top1': float(labels[order[0]]), 'oracle': float(labels[ids].max()),
               'top1_source': case['sources'][order[0]],
               'within_stage_source': ordering_metrics(scores[ids], labels[ids], groups=groups)}
        if stage == 'initial':
            selected = {case['identities'][i][0] for i in order[:64]}
            paired = [i for i, identity in enumerate(case['identities']) if identity[0] in selected]
            row['selected64_union_oracle'] = float(labels[paired].max())
        row['sources'] = {}
        for source in ('sdpi', 'bitseed'):
            members = [i for i in ids if case['sources'][i] == source]
            if not members:
                row['sources'][source] = {'states': 0}
                continue
            ranked = [i for i in order if i in set(members)]
            row['sources'][source] = {
                'states': len(members), 'label_mean': float(labels[members].mean()),
                'oracle': float(labels[members].max()), 'top1': float(labels[ranked[0]]),
                'hits23_states': int((labels[members] >= .23).sum()),
                'hits49_states': int((labels[members] >= .49).sum()),
                'top64_count': sum(case['sources'][i] == source for i in order[:64])}
        result[stage] = row
    return result


def geometry_features(graph):
    """C-alpha/ligand distance proxies, NOT all-atom clashes or binding energies."""
    coords, types = graph['node_coords'], graph['node_types']
    values = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        distances = torch.cdist(coords[types == a], coords[types == b])
        nearest = distances.min(1).values
        values.extend([float(distances.min()), float(nearest.mean()),
                       float((distances < 4.).float().mean()),
                       float((distances < 8.).any(1).float().mean()),
                       float((distances < 8.).any(0).float().mean())])
    return torch.tensor(values, dtype=torch.float64)


def fit_ridge(cases, features, alpha=1.):
    """Train-only standardization; each case has equal total regression weight."""
    x = torch.cat(features)
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    design = torch.cat(((x - mean) / scale, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    y = torch.cat([c['labels'] for c in cases]).double()
    weights = torch.cat([torch.full((len(f),), 1. / len(f), dtype=x.dtype) for f in features])
    penalty = torch.eye(design.shape[1], dtype=x.dtype) * alpha
    penalty[-1, -1] = 0.
    beta = torch.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                              design.T @ (weights * y))
    return mean, scale, beta


def transform_graph(graph, mode):
    """Rigid global control and local destructive translations, without relabeling."""
    coords = graph['node_coords'].clone()
    if mode == 'global_rigid':
        rotation = coords.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        coords = coords @ rotation.T + coords.new_tensor([11., -7., 3.])
    elif mode == 'separate_poi':
        coords[graph['node_types'] == 1] += coords.new_tensor([100., 0., 0.])
    elif mode == 'separate_ligand':
        coords[graph['node_types'] == 2] += coords.new_tensor([100., 0., 0.])
    else:
        raise ValueError('unknown geometry intervention')
    return {**graph, 'node_coords': coords}


def local_quality_audit(case, scores):
    """Label-free nearest neighbors within stage/source, in the fixed E3 frame.

    Distance equally weights POI C-alpha and ligand atom mean squared distances.
    No native alignment, no new perturbed labels, and no cross-case neighbors.
    """
    rows = []
    for stage in ('initial', 'final'):
        for source in ('sdpi', 'bitseed'):
            ids = [i for i, pair in enumerate(zip(case['stages'], case['sources']))
                   if pair == (stage, source)]
            if len(ids) < 2:
                continue
            reference = case['graphs'][ids[0]]
            matrices = []
            for kind in (1, 2):
                mask = reference['node_types'] == kind
                coords = torch.stack([case['graphs'][i]['node_coords'][mask] for i in ids]).double()
                matrices.append(coords.flatten(1) / (2 * int(mask.sum())) ** .5)
            features = torch.cat(matrices, 1)
            distances = torch.cdist(features, features)
            distances.fill_diagonal_(float('inf'))
            nearest = distances.argmin(1).tolist()
            pairs = sorted({tuple(sorted((i, j))) for i, j in enumerate(nearest)})
            records = []
            for a, b in pairs:
                i, j = ids[a], ids[b]
                delta = float(case['labels'][i] - case['labels'][j])
                score_delta = float(scores[i] - scores[j])
                accuracy = None if abs(delta) <= 1e-4 else (.5 if score_delta == 0 else float(delta * score_delta > 0))
                records.append({'a': case['identities'][i], 'b': case['identities'][j],
                                'distance_angstrom': float(distances[a, b]), 'dockq_delta': delta,
                                'score_delta': score_delta, 'accuracy': accuracy})
            positive = [i for i in ids if case['labels'][i] >= .23]
            negative = [i for i in ids if case['labels'][i] < .23]
            discrimination = None
            if positive and negative:
                differences = scores[positive, None] - scores[negative][None, :]
                discrimination = float(((differences > 0).float() + .5 * (differences == 0)).mean())
            rows.append({'stage': stage, 'source': source, 'states': len(ids),
                         'positive_states': len(positive), 'negative_states': len(negative),
                         'positive_negative_accuracy': discrimination, 'nearest_pairs': records})
    return rows


@torch.no_grad()
def structural_embeddings(model, case):
    """Frozen pre-readout structural tokens, excluding both condition embeddings."""
    features = []
    def capture(module, inputs):
        # readout concatenates global, E3-L, POI-L, ligand, stage, source.
        features.append(inputs[0][:, :4 * model.encoder.hidden_dim].detach().cpu().double())
    handle = model.readout.register_forward_pre_hook(capture)
    try:
        score(model, case)
    finally:
        handle.remove()
    return torch.cat(features)


@torch.no_grad()
def score(model, case, sources=None):
    return model.forward_list(case['graphs'], stages=case['stages'],
                              sources=case['sources'] if sources is None else sources)['score'].cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=('support', 'intervention', 'local', 'probe'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('use a new output')
    torch.set_num_threads(2)
    digest = file_digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['metadata']['arguments']
    if len(config['train_archives']) != 1 or config['candidate_scope'] != 'archived-pairs':
        raise ValueError('one archived-pairs training shard required')
    datasets = {
        'train': StructuralArchiveDataset(config['train_archives'][0], config['train_labels'][0], 'train'),
        'val': StructuralArchiveDataset(config['val_archives'], config['val_labels'], 'val')}
    coverage = audit_datasets(datasets['train'], datasets['val'])
    for name, dataset in datasets.items():
        if dict(dataset.seals) != checkpoint['metadata'][name + '_case_seals']:
            raise ValueError('data changed relative to checkpoint')
    report = {'mode': args.mode, 'checkpoint_sha256': digest, 'epoch': checkpoint['epoch'],
              'scope': 'reused internal validation, not independent test; diagnostics do not select checkpoints',
              'coverage': coverage, 'rows': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.out.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.out)

    if args.mode == 'support':
        cached = {}
        for split, dataset in datasets.items():
            cases, features = [], []
            for i in range(len(dataset)):
                case = dataset[i]
                features.append(torch.stack([geometry_features(g) for g in case['graphs']]))
                cases.append({k: v for k, v in case.items() if k != 'graphs'})
                report['rows'].append({'case': case['case'], 'split': split,
                    'ligand_atoms': int((case['graphs'][0]['node_types'] == 2).sum()),
                    'label_support': summarize(case, torch.zeros_like(case['labels']))})
                print(f'FEATURES {split} {i + 1}/{len(dataset)} {case["case"]}', flush=True)
                save()
            cached[split] = cases, features
        mean, scale, beta = fit_ridge(*cached['train'])
        report['baseline_protocol'] = 'ridge alpha=1 fixed, no flags, no validation tuning; simple normalized C-alpha contact rule'
        report['ridge_parameters'] = {'mean': mean.tolist(), 'scale': scale.tolist(), 'beta': beta.tolist()}
        for split, (cases, features) in cached.items():
            for case, feature in zip(cases, features):
                scores = ((feature - mean) / scale) @ beta[:-1] + beta[-1]
                row = next(r for r in report['rows'] if r['case'] == case['case'] and r['split'] == split)
                row['ridge'] = summarize(case, scores)
                row['contact_rule'] = summarize(case, feature[:, [3, 4, 8, 9, 13, 14]].mean(1))
        report['warning'] = 'label_support ordering is canonical dummy order, not a scored baseline; protein family homology not audited'
    elif args.mode == 'probe':
        model = StructuralQualityRanker(**checkpoint['config']).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.requires_grad_(False)
        cached = {}
        for split, dataset in datasets.items():
            cases, features = [], []
            for i in range(len(dataset)):
                case = dataset[i]
                features.append(structural_embeddings(model, case))
                cases.append({k: v for k, v in case.items() if k != 'graphs'})
                print(f'PROBE {split} {i + 1}/{len(dataset)} {case["case"]}', flush=True)
            cached[split] = cases, features
        mean, scale, beta = fit_ridge(*cached['train'])
        report['warning'] = 'frozen supervised encoder, pre-readout four structural tokens only; ridge alpha=1 fixed; train-only fitting, not proof of representation sufficiency'
        report['ridge_parameters'] = {'mean': mean.tolist(), 'scale': scale.tolist(), 'beta': beta.tolist()}
        for split, (cases, features) in cached.items():
            for case, feature in zip(cases, features):
                scores = ((feature - mean) / scale) @ beta[:-1] + beta[-1]
                report['rows'].append({'case': case['case'], 'split': split,
                                      'probe': summarize(case, scores)})
    elif args.mode == 'local':
        model = StructuralQualityRanker(**checkpoint['config']).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        report['warning'] = 'nearest does not necessarily mean close; inspect distance bins; same-case correlated pairs, no native-based neighbor selection'
        for split, dataset in datasets.items():
            for i in range(len(dataset)):
                case = dataset[i]
                scores = score(model, case)
                report['rows'].append({'case': case['case'], 'split': split,
                                      'local': local_quality_audit(case, scores)})
                print(f'LOCAL {split} {i + 1}/{len(dataset)} {case["case"]}', flush=True)
                save()
    else:
        model = StructuralQualityRanker(**checkpoint['config']).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        dataset = datasets['val']
        report['warning'] = 'flag interventions are out-of-distribution diagnostics, not a retrained no-source model; perturbed structures have no new DockQ labels'
        for i in range(len(dataset)):
            case = dataset[i]
            original = score(model, case)
            row = {'case': case['case'], 'original': summarize(case, original), 'interventions': {}}
            for mode in ('swapped', 'all_sdpi', 'zero_embedding'):
                sources = ([('bitseed' if s == 'sdpi' else 'sdpi') for s in case['sources']]
                           if mode == 'swapped' else ['sdpi'] * len(case['sources']))
                saved = model.source_embedding.weight.detach().clone()
                if mode == 'zero_embedding':
                    with torch.no_grad():
                        model.source_embedding.weight.zero_()
                try:
                    changed = score(model, case, sources)
                finally:
                    with torch.no_grad():
                        model.source_embedding.weight.copy_(saved)
                row['interventions'][mode] = {'metrics': summarize(case, changed),
                    'mean_abs_score_change': float((changed - original).abs().mean())}
            row['geometry'] = []
            # Canonical first state in each stage/source, selected without native labels.
            seen = set()
            for j, (stage, source) in enumerate(zip(case['stages'], case['sources'])):
                if (stage, source) in seen:
                    continue
                seen.add((stage, source))
                for mode in ('global_rigid', 'separate_poi', 'separate_ligand'):
                    with torch.no_grad():
                        changed = model(**transform_graph(case['graphs'][j], mode), stage=stage, source=source)['score']
                    row['geometry'].append({'identity': case['identities'][j], 'source': source,
                        'mode': mode, 'score_delta': float(changed[0] - original[j])})
            report['rows'].append(row)
            print(f'INTERVENTION {i + 1}/{len(dataset)} {case["case"]}', flush=True)
            save()
    if file_digest(args.checkpoint) != digest:
        raise ValueError('checkpoint changed during audit')
    report['complete'] = True
    save()
    print(f'DONE {args.out}', flush=True)


if __name__ == '__main__':
    main()
