"""One model, expanded training list, unchanged archived-walker Union."""

import copy
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch

from test_structural_ranking_data import archive_fixture, pool_labels, rejected
from tricoopdiff.ranking_archive import state_hash
from tricoopdiff.structural_ranking_data import adapt_case, adapt_initial_pool, StructuralJointDataset
from scripts.train_structural_ranker import evaluate, geometry_shuffle_diagnostic


def joint_fixture():
    archive, labels = archive_fixture()
    extra = copy.deepcopy(archive['pool'][0])
    extra['candidate_id'] = ['sdpi', 1]
    extra['state']['ligand'] += 2
    extra['state_sha256'] = state_hash(extra['state'])
    archive['pool'].append(extra)
    initials = pool_labels(archive)
    initials['rows'][1]['dockq']['dockq'] = .9

    class Dataset:
        manifest_digest = 'same'
        label_manifest_digest = 'labels'
        cases = [archive['case']]
        split = 'train'
        manifest = {}
        seals = {archive['case']: {'archive_sha256': 'same', 'labels_sha256': 'label'}}
        def __init__(self, scope, case): self.candidate_scope, self.case = scope, case
        def __getitem__(self, index): return self.case

    paired = Dataset('archived-pairs', adapt_case(archive, labels))
    initial = Dataset('initial-pool', adapt_initial_pool(archive, initials))
    return StructuralJointDataset(paired, initial)


def test_joint_list_and_evaluation_scopes():
    dataset = joint_fixture()
    case = dataset[0]
    assert case['identities'] == [(0, 'initial'), (0, 'final'), (1, 'initial')]
    assert case['union_eligible'].tolist() == [True, True, False]
    class Scorer:
        def eval(self): return self
        def forward_list(self, *args, **kwargs): return {'score': torch.tensor([0., 1., 10.])}
    result = evaluate(Scorer(), dataset, 'cpu', shuffle_repeats=4)
    assert abs(result['initial']['mean_top1_available'] - .9) < 1e-6
    assert abs(result['union']['mean_top1_available'] - .4) < 1e-6
    assert result['rows'][0]['union']['ranking'] == [(0, 'final'), (0, 'initial')]
    assert result['initial_selection']['mean_selected_source_counts']['sdpi'] == 2


def test_joint_rejects_conflicting_initial_labels():
    dataset = joint_fixture()
    dataset.initial.case['labels'][0] = .8
    rejected(lambda: dataset[0])


def test_union_shuffle_cannot_return_unselected_initial():
    report = geometry_shuffle_diagnostic(
        torch.tensor([0., 1., 10.]), torch.tensor([.2, .4, .9]),
        ['initial', 'final', 'initial'], ['sdpi'] * 3, repeats=32, seed=9,
        union_eligible=torch.tensor([True, True, False]))
    assert report['union']['shuffled_mean_top1'] < .401


if __name__ == '__main__':
    test_joint_list_and_evaluation_scopes()
    test_joint_rejects_conflicting_initial_labels()
    test_union_shuffle_cannot_return_unselected_initial()
    print('Joint ranking: 3 contract tests passed')
