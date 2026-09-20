"""Pair identity, actual geometry, missing-state and split safety checks."""

from pathlib import Path
import copy
import json
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tricoopdiff.ranking_archive import ARCHIVE_VERSION, state_hash
from tricoopdiff.initial_selector import POOL_LABEL_VERSION
from tricoopdiff.structural_ranking_data import (
    StructuralArchiveCollection, StructuralArchiveDataset, adapt_case, adapt_initial_pool, audit_datasets, file_digest,
)
from tricoopdiff.structural_ranker import StructuralQualityRanker, structural_ranking_loss
from test_structural_ranker import fixture


def archive_fixture():
    graph = fixture()
    state = {'poi': graph['node_coords'][5:11].clone(),
             'ligand': graph['node_coords'][11:].clone(), 'body_frame': torch.eye(3)}
    final = {k: v.clone() for k, v in state.items()}
    final['ligand'] += .3
    inputs = {}
    for name, start, end in [('e3_data', 0, 5), ('poi_data', 5, 11), ('ligand_data', 11, 15)]:
        inputs[name] = {'coords': graph['node_coords'][None, start:end] + 99,
                        'features': graph['node_features'][None, start:end]}
    inputs['ligand_data']['edge_index'] = graph['lig_edge_index']
    pair = {'pool_index': 0, 'source': 'sdpi', 'candidate_id': ['sdpi', 0],
            'valid': True, 'initial_state_sha256': state_hash(state),
            'final': final, 'final_state_sha256': state_hash(final)}
    archive = {'protocol': ARCHIVE_VERSION, 'case': '1ABC_A_B_L', 'group': 'g1',
               'input_graphs': inputs, 'pairs': [pair],
               'pool': [{'source': 'sdpi', 'candidate_id': ['sdpi', 0],
                         'state': state, 'state_sha256': state_hash(state)}]}
    labels = {'case': archive['case'], 'labels': [
        {'pool_index': 0, 'valid': True, 'initial': {'dockq': .2}, 'final': {'dockq': .4}}]}
    return archive, labels


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('invalid archive was accepted')


def pool_labels(archive):
    return {'case': archive['case'], 'rows': [
        {'pool_index': i, 'candidate_id': candidate['candidate_id'], 'source': candidate['source'],
         'state_sha256': candidate['state_sha256'], 'dockq': {'dockq': .2 + .1 * i}}
        for i, candidate in enumerate(archive['pool'])]}


def test_pair_geometry_and_training_step():
    archive, labels = archive_fixture()
    case = adapt_case(archive, labels)
    assert torch.equal(case['graphs'][0]['node_coords'][5:11], archive['pool'][0]['state']['poi'])
    assert case['identities'] == [(0, 'initial'), (0, 'final')]
    assert all(set(g) == {'node_coords', 'node_features', 'node_types', 'lig_edge_index'} for g in case['graphs'])
    model = StructuralQualityRanker(16, 16)
    before = model.rank_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    out = model.forward_list(case['graphs'], stages=case['stages'], sources=case['sources'])
    loss = structural_ranking_loss(out, case['labels'], quality_weight=.1)
    loss.backward(); optimizer.step()
    assert not torch.equal(before, model.rank_head.weight)


def test_failure_retains_initial_only():
    archive, labels = archive_fixture()
    archive['pairs'][0].update(valid=False, final=None, final_state_sha256=None)
    labels['labels'][0].update(valid=False, final=None)
    assert adapt_case(archive, labels)['stages'] == ['initial']


def test_corruption_and_duplicate_labels_rejected():
    for change in ('source', 'hash', 'duplicate', 'validity'):
        archive, labels = archive_fixture()
        if change == 'source': archive['pairs'][0]['source'] = 'bitseed'
        if change == 'hash': archive['pairs'][0]['final']['ligand'][0, 0] += 1
        if change == 'duplicate': labels['labels'] *= 2
        if change == 'validity': labels['labels'][0]['valid'] = False
        rejected(lambda: adapt_case(archive, labels))


def test_split_and_protocol_isolation():
    class Dataset:
        manifest = dict(checkpoint_sha256='x', steps=20, pool={}, selection={})
        def __init__(self, case, group): self.row = dict(case=case, group=group, pool_size=256, labeled_pairs=32)
        def __len__(self): return 1
        def __getitem__(self, index): return self.row
    train = Dataset('1ABC_A_B_L', 'g1')
    val = Dataset('2ABC_A_B_L', 'g2')
    assert len(audit_datasets(train, val)) == 2
    rejected(lambda: audit_datasets(train, Dataset('1ABC_C_D_L', 'g2')))
    rejected(lambda: audit_datasets(train, Dataset('2ABC_A_B_L', 'g1')))
    val.manifest = {**val.manifest, 'steps': 10}
    rejected(lambda: audit_datasets(train, val))


def test_evaluation_reports_missing_final_without_inventing_pose():
    from scripts.train_structural_ranker import evaluate
    archive, labels = archive_fixture()
    archive['pairs'][0].update(valid=False, final=None, final_state_sha256=None)
    labels['labels'][0].update(valid=False, final=None)
    case = adapt_case(archive, labels)
    report = evaluate(StructuralQualityRanker(16, 16), [case], 'cpu')
    assert report['final']['available_cases'] == 0
    assert report['final']['mean_top1_available'] is None
    assert report['final']['success_rate_0.23_all'] == 0
    assert report['final']['hit_rate_given_oracle_0.23'] is None
    assert report['union']['oracle_opportunities_0.23'] == 0
    assert report['union']['mean_top1_regret_available'] == 0
    assert report['union']['available_cases'] == 1
    assert report['rows'][0]['union']['ranking'] == [(0, 'initial')]


def test_full_order_metrics_distinguish_lower_rank_errors():
    from scripts.train_structural_ranker import ordering_metrics
    labels = torch.tensor([.8, .5, .2, .01])
    perfect = ordering_metrics(torch.tensor([3., 2., 1., 0.]), labels)
    swapped = ordering_metrics(torch.tensor([3., 2., 0., 1.]), labels)
    assert perfect == {'comparable_pairs': 6, 'pair_accuracy': 1.}
    assert swapped['pair_accuracy'] < 1.
    assert ordering_metrics(torch.zeros(4), labels)['pair_accuracy'] == .5
    assert ordering_metrics(torch.zeros(4), torch.zeros(4))['pair_accuracy'] is None


def test_training_entrypoint_checkpoint_roundtrip():
    from scripts.train_structural_ranker import main, validate_resume
    from scripts.rank_structural_archive import rank_archive
    with tempfile.TemporaryDirectory(prefix='structural-ranker-test-') as temporary:
        root = Path(temporary)
        arguments = []
        for split, name, group in [('train', '1ABC_A_B_L', 'g1'), ('val', '2ABC_A_B_L', 'g2')]:
            archive, labels = archive_fixture()
            extra = copy.deepcopy(archive['pool'][0])
            extra['candidate_id'] = ['bitseed', 1]
            extra['source'] = 'bitseed'
            extra['state']['ligand'] += 1
            extra['state_sha256'] = state_hash(extra['state'])
            archive['pool'].append(extra)  # No refinement or label for this candidate.
            archive.update(case=name, group=group, split=split)
            labels['case'] = name
            manifest = dict(cases=[name], checkpoint_sha256='test', steps=20, pool={}, selection={})
            archive['manifest'] = manifest
            poses, targets = root / (split + '_archives'), root / (split + '_labels')
            poses.mkdir(); targets.mkdir()
            (poses / 'manifest.json').write_text(json.dumps(manifest))
            torch.save(archive, poses / f'{name}.pt')
            digest = file_digest(poses / f'{name}.pt')
            (poses / f'{name}.complete.json').write_text(json.dumps({'archive_sha256': digest}))
            labels['archive_sha256'] = digest
            (targets / f'{name}.json').write_text(json.dumps(labels))
            (targets / 'manifest.json').write_text(json.dumps({
                'archive_manifest_sha256': file_digest(poses / 'manifest.json')}))
            arguments.extend(['--' + split + '-archives', str(poses), '--' + split + '-labels', str(targets)])
        out = root / 'output'
        main(arguments + ['--out', str(out), '--epochs', '1', '--hidden-dim', '16',
                          '--quality-weight', '.1', '--top-weight', '.5'])
        checkpoint = torch.load(out / 'best.pt', weights_only=False)
        restored = StructuralQualityRanker(**checkpoint['config'])
        restored.load_state_dict(checkpoint['state_dict'], strict=True)
        assert checkpoint['epoch'] == 1
        assert checkpoint['metadata']['future_head_trained'] is False
        assert checkpoint['metadata']['arguments']['top_weight'] == .5
        assert checkpoint['validation']['union']['available_cases'] == 1
        assert len((out / 'history.jsonl').read_text().splitlines()) == 1
        assert checkpoint['validation']['geometry_shuffle']['repeats'] == 32
        common = arguments + ['--epochs', '2', '--hidden-dim', '16', '--quality-weight', '.1', '--top-weight', '.5']
        continuous, resumed = root / 'continuous', root / 'resumed'
        main(common + ['--out', str(continuous)])
        main(common + ['--out', str(resumed), '--resume', str(out / 'last.pt')])
        direct = torch.load(continuous / 'last.pt', weights_only=False)
        continued = torch.load(resumed / 'last.pt', weights_only=False)
        assert direct['epoch'] == continued['epoch'] == 2
        assert all(torch.equal(value, continued['state_dict'][key]) for key, value in direct['state_dict'].items())
        assert direct['validation'] == continued['validation']
        for index, state in direct['optimizer']['state'].items():
            for key, value in state.items():
                other = continued['optimizer']['state'][index][key]
                assert torch.equal(value, other) if torch.is_tensor(value) else value == other
        assert direct['resume_state']['best_payload']['epoch'] == continued['resume_state']['best_payload']['epoch']
        assert len((resumed / 'history.jsonl').read_text().splitlines()) == 1
        for key in ('train_case_seals', 'generator_protocol', 'encoder_initialization', 'implementation_sha256'):
            changed_metadata = copy.deepcopy(continued['metadata'])
            changed_metadata[key] = 'changed'
            rejected(lambda: validate_resume(continued, changed_metadata, continued['config'], 3))
        changed_metadata = copy.deepcopy(continued['metadata'])
        changed_metadata['arguments']['top_weight'] = 9.
        rejected(lambda: validate_resume(continued, changed_metadata, continued['config'], 3))
        rejected(lambda: validate_resume(checkpoint, checkpoint['metadata'], checkpoint['config'], 2))
        initial_arguments = []
        for split, name in [('train', '1ABC_A_B_L'), ('val', '2ABC_A_B_L')]:
            poses, targets = root / (split + '_archives'), root / (split + '_initial_labels')
            targets.mkdir()
            archive = torch.load(poses / f'{name}.pt', weights_only=False)
            label = {**pool_labels(archive), 'archive_sha256': file_digest(poses / f'{name}.pt')}
            (targets / f'{name}.json').write_text(json.dumps(label))
            (targets / 'manifest.json').write_text(json.dumps({
                'protocol': POOL_LABEL_VERSION, 'pool_size': len(archive['pool']),
                'archive_manifest_sha256': file_digest(poses / 'manifest.json')}))
            initial_arguments.extend(['--' + split + '-archives', str(poses), '--' + split + '-labels', str(targets)])
        initial_out = root / 'initial_only'
        main(initial_arguments + ['--candidate-scope', 'initial-pool', '--out', str(initial_out),
                                  '--hidden-dim', '16', '--epochs', '1', '--shuffle-repeats', '0'])
        initial_checkpoint = torch.load(initial_out / 'best.pt', weights_only=False)
        assert 'union' not in initial_checkpoint['validation'] and 'final' not in initial_checkpoint['validation']
        assert initial_checkpoint['validation']['initial_selection']['mean_oracle_retention'] == 1.
        rejected(lambda: rank_archive(initial_out / 'best.pt', root / 'val_archives' / '2ABC_A_B_L.pt'))
        assert len(rank_archive(initial_out / 'best.pt', root / 'val_archives' / '2ABC_A_B_L.pt',
                                candidate_scope='initial-pool')['rankings']['initial']) == 2
        # Label directories are unavailable to the inference entry point.
        for split in ('train', 'val'):
            (root / (split + '_labels')).rename(root / (split + '_labels_unavailable'))
        report = rank_archive(out / 'best.pt', root / 'val_archives' / '2ABC_A_B_L.pt')
        assert len(report['rankings']['union']) == 2
        assert report['available_states'] == 2
        assert all('dockq' not in row for row in report['rankings']['union'])
        expected = checkpoint['validation']['rows'][0]['union']['ranking']
        assert [(row['pool_index'], row['stage']) for row in report['rankings']['union']] == expected
        truncated = rank_archive(out / 'best.pt', root / 'val_archives' / '2ABC_A_B_L.pt', top_k=1)
        assert truncated['rankings']['union'] == report['rankings']['union'][:1]
        pool_report = rank_archive(out / 'best.pt', root / 'val_archives' / '2ABC_A_B_L.pt',
                                   candidate_scope='initial-pool')
        assert set(pool_report['rankings']) == {'initial'}
        assert {row['pool_index'] for row in pool_report['rankings']['initial']} == {0, 1}
        assert pool_report['available_states'] == 2
        assert len(report['rankings']['initial']) == 1, 'pool scope leaked into archived protocol'
        archived_initial = report['rankings']['initial'][0]
        assert next(row for row in pool_report['rankings']['initial'] if row['pool_index'] == 0) == archived_initial
        estimates = rank_archive(out / 'best.pt', root / 'val_archives' / '2ABC_A_B_L.pt',
                                  include_quality_estimates=True)
        assert estimates['head_capabilities']['quality_aux']
        assert estimates['head_capabilities']['probabilities_calibrated'] is False
        for row, original in zip(estimates['rankings']['union'], report['rankings']['union']):
            assert row['score'] == original['score']
            assert row['pool_index'] == original['pool_index'] and row['stage'] == original['stage']
            aux = row['uncalibrated_quality_estimates']
            assert 0 <= aux['p49'] <= aux['p23'] <= 1
        legacy = {k: v for k, v in checkpoint.items() if k not in ('head_supervision', 'optimizer_steps')}
        torch.save(legacy, out / 'legacy.pt')
        legacy_report = rank_archive(out / 'legacy.pt', root / 'val_archives' / '2ABC_A_B_L.pt')
        assert legacy_report['rankings'] == report['rankings']
        rejected(lambda: rank_archive(out / 'legacy.pt', root / 'val_archives' / '2ABC_A_B_L.pt',
                                       include_quality_estimates=True))
        checkpoint['metadata']['generator_protocol']['steps'] = 99
        torch.save(checkpoint, out / 'incompatible.pt')
        rejected(lambda: rank_archive(out / 'incompatible.pt', root / 'val_archives' / '2ABC_A_B_L.pt'))
        dataset = StructuralArchiveDataset(root / 'val_archives', root / 'val_labels_unavailable', 'val')
        dataset[0]
        assert dataset.seals == checkpoint['metadata']['val_case_seals']
        label_path = root / 'val_labels_unavailable' / '2ABC_A_B_L.json'
        changed = json.loads(label_path.read_text())
        changed['labels'][0]['initial']['dockq'] = .99
        label_path.write_text(json.dumps(changed))
        rejected(lambda: dataset[0])


def test_native_free_adapter_matches_labeled_graphs():
    archive, labels = archive_fixture()
    unlabelled, labelled = adapt_case(archive), adapt_case(archive, labels)
    assert 'labels' not in unlabelled
    assert unlabelled['labeled_pairs'] == 0 and unlabelled['archived_pairs'] == 1
    assert unlabelled['identities'] == labelled['identities']
    for left, right in zip(unlabelled['graphs'], labelled['graphs']):
        assert all(torch.equal(left[key], right[key]) for key in left)


def test_initial_pool_never_reads_refinement_and_rejects_corruption():
    archive, _ = archive_fixture()
    before = adapt_initial_pool(archive)
    del archive['pairs']
    after = adapt_initial_pool(archive)
    assert after['identities'] == [(0, 'initial')]
    assert 'labels' not in after
    for left, right in zip(before['graphs'], after['graphs']):
        assert all(torch.equal(left[key], right[key]) for key in left)
    archive['pool'].append(copy.deepcopy(archive['pool'][0]))
    rejected(lambda: adapt_initial_pool(archive))
    archive['pool'].pop()
    archive['pool'][0]['state']['ligand'][0, 0] += 1
    rejected(lambda: adapt_initial_pool(archive))


def test_checkpoint_capabilities_reject_untrained_or_inconsistent_heads():
    from scripts.rank_structural_archive import checkpoint_capabilities
    checkpoint = {'architecture': 'structural_quality_ranker', 'epoch': 1,
                  'optimizer_steps': 1,
                  'metadata': {'task': 'quality', 'arguments': {'quality_weight': 0.}},
                  'head_supervision': {'rank': True, 'quality_aux': False, 'future': False}}
    capabilities = checkpoint_capabilities(checkpoint)
    assert capabilities['rank'] and not capabilities['quality_aux'] and not capabilities['future']
    for alteration in ('no_steps', 'untrained', 'wrong_aux', 'future'):
        bad = copy.deepcopy(checkpoint)
        if alteration == 'no_steps': bad['optimizer_steps'] = 0
        if alteration == 'untrained': bad['epoch'] = 0
        if alteration == 'wrong_aux': bad['head_supervision']['quality_aux'] = True
        if alteration == 'future': bad['head_supervision']['future'] = True
        rejected(lambda: checkpoint_capabilities(bad))


def test_pretrained_encoder_selection_is_explicit_and_isolated():
    from scripts.train_structural_ranker import initialize_encoder
    model = StructuralQualityRanker(16, 16)
    before_head = model.rank_head.weight.detach().clone()
    main = {f'pooling.{key}': value.detach().clone() for key, value in model.encoder.state_dict().items()}
    ema = {key: value + .1 for key, value in main.items()}
    with tempfile.TemporaryDirectory(prefix='ranker-encoder-test-') as temporary:
        path = Path(temporary) / 'refiner.pt'
        torch.save({'trifuse_state_dict': main, 'ema_trifuse_state_dict': ema}, path)
        metadata = initialize_encoder(model, path, 'ema')
        assert metadata['weights'] == 'ema' and metadata['checkpoint_sha256'] == file_digest(path)
        assert all(torch.equal(value, ema['pooling.' + key]) for key, value in model.encoder.state_dict().items())
        assert torch.equal(before_head, model.rank_head.weight), 'refiner initialization overwrote ranking head'
        initialize_encoder(model, path, 'model')
        assert all(torch.equal(value, main['pooling.' + key]) for key, value in model.encoder.state_dict().items())
        with torch.no_grad():
            model.encoder.protein_feature_proj.weight.add_(1)
        reloaded = torch.load(path, weights_only=False)
        assert torch.equal(reloaded['trifuse_state_dict']['pooling.protein_feature_proj.weight'],
                           main['pooling.protein_feature_proj.weight'])
        rejected(lambda: initialize_encoder(StructuralQualityRanker(16, 32), path, 'ema'))
        torch.save({'trifuse_state_dict': main}, path)
        rejected(lambda: initialize_encoder(model, path, 'ema'))


def test_failed_checkpoint_save_keeps_previous_file():
    from scripts.train_structural_ranker import atomic_checkpoint
    with tempfile.TemporaryDirectory(prefix='ranker-atomic-test-') as temporary:
        path = Path(temporary) / 'last.pt'
        atomic_checkpoint({'epoch': 1}, path)
        previous = path.read_bytes()
        with patch('scripts.train_structural_ranker.torch.save', side_effect=OSError('simulated disk failure')):
            try:
                atomic_checkpoint({'epoch': 2}, path)
            except OSError:
                pass
            else:
                raise AssertionError('checkpoint failure was hidden')
        assert path.read_bytes() == previous
        assert torch.load(path, weights_only=False)['epoch'] == 1


def test_archive_collection_order_and_protocol_guards():
    class Shard:
        split = 'train'
        manifest = dict(checkpoint_sha256='x', steps=20, pool={}, selection={})
        def __init__(self, case):
            self.cases = [case]
            self.manifest_digest = case
            self.label_manifest_digest = 'label-' + case
            self.seals = {}
        def __getitem__(self, index):
            case = self.cases[index]
            self.seals[case] = {'archive_sha256': case}
            return {'case': case}
    a, b = Shard('1ABC_A_B_L'), Shard('2ABC_A_B_L')
    combined = StructuralArchiveCollection([a, b])
    assert combined.cases == a.cases + b.cases
    assert len(combined) == 2 and combined[1]['case'] == b.cases[0]
    assert b.cases[0] in combined.seals
    assert combined.manifest_digest != StructuralArchiveCollection([b, a]).manifest_digest
    rejected(lambda: StructuralArchiveCollection([a, Shard('1ABC_C_D_L')]))
    b.manifest = {**b.manifest, 'steps': 10}
    rejected(lambda: StructuralArchiveCollection([a, b]))
    rejected(lambda: StructuralArchiveCollection([]))


def test_initial_pool_label_alignment_and_no_feature_leakage():
    archive, _ = archive_fixture()
    labels = pool_labels(archive)
    labels['rows'][0]['geometry'] = [float('nan')]
    labelled = adapt_initial_pool(archive, labels)
    unlabelled = adapt_initial_pool(archive)
    assert labelled['labeled_initials'] == 1 and labelled['labeled_pairs'] == 0
    assert torch.equal(labelled['labels'], torch.tensor([.2]))
    assert all(torch.equal(labelled['graphs'][0][k], unlabelled['graphs'][0][k]) for k in unlabelled['graphs'][0])
    for kind in ('missing', 'duplicate', 'source', 'state', 'case'):
        broken = copy.deepcopy(labels)
        if kind == 'missing': broken['rows'] = []
        if kind == 'duplicate': broken['rows'] *= 2
        if kind == 'source': broken['rows'][0]['source'] = 'bitseed'
        if kind == 'state': broken['rows'][0]['state_sha256'] = 'wrong'
        if kind == 'case': broken['case'] = 'wrong'
        rejected(lambda: adapt_initial_pool(archive, broken))


if __name__ == '__main__':
    test_pair_geometry_and_training_step()
    test_failure_retains_initial_only()
    test_corruption_and_duplicate_labels_rejected()
    test_split_and_protocol_isolation()
    test_evaluation_reports_missing_final_without_inventing_pose()
    test_full_order_metrics_distinguish_lower_rank_errors()
    test_training_entrypoint_checkpoint_roundtrip()
    test_native_free_adapter_matches_labeled_graphs()
    test_initial_pool_never_reads_refinement_and_rejects_corruption()
    test_checkpoint_capabilities_reject_untrained_or_inconsistent_heads()
    test_pretrained_encoder_selection_is_explicit_and_isolated()
    test_failed_checkpoint_save_keeps_previous_file()
    test_archive_collection_order_and_protocol_guards()
    test_initial_pool_label_alignment_and_no_feature_leakage()
    print('Structural ranking data: 14 tests passed')
