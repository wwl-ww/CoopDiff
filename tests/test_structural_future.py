"""Future utility is a trainable initial-only task, not final-state leakage."""
import copy
import json
import tempfile
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from test_structural_ranker import fixture
from tricoopdiff.structural_ranking_data import future_utility_case
from tricoopdiff.structural_ranker import StructuralQualityRanker, structural_ranking_loss
from tricoopdiff.ranking_selection import refinement_selection_metrics
from scripts.train_structural_ranker import score_case
from scripts.rank_structural_archive import checkpoint_capabilities


def example():
    graph = fixture(); graph.pop('stage'); graph.pop('source')
    graphs = [copy.deepcopy(graph) for _ in range(5)]
    graphs[2]['node_coords'][5:11] += 2
    return {'case': 'test', 'graphs': graphs,
            'identities': [(0,'initial'), (0,'final'), (1,'initial'), (1,'final'), (2,'initial')],
            'stages': ['initial','final','initial','final','initial'],
            'sources': ['sdpi']*5, 'labels': torch.tensor([.8,.01,.1,.9,.5])}


def test_future_uses_only_initial_graphs_and_observed_final_labels():
    case = example(); future = future_utility_case(case)
    assert future['identities'] == [(0,'initial'), (1,'initial')]
    assert torch.equal(future['labels'], case['labels'][[1,3]])
    assert future['graphs'][0] is case['graphs'][0]
    model = StructuralQualityRanker(16, 16)
    before = score_case(model, future, 'cpu')['score']
    case['graphs'][1]['node_coords'] *= 100
    case['graphs'][3]['node_coords'] *= -100
    after = score_case(model, future_utility_case(case), 'cpu')['score']
    assert torch.equal(before, after), 'future prediction leaked refined coordinates'
    structural_ranking_loss({'score': before}, future['labels']).backward()
    assert model.future_head.weight.grad.abs().sum() > 0
    assert model.encoder.protein_feature_proj.weight.grad.abs().sum() > 0
    assert model.rank_head.weight.grad is None


def test_future_scores_allocate_compute_quality_scores_choose_output():
    case = example()
    quality = torch.tensor([9.,1.,8.,10.,11.])
    allocation = torch.tensor([1.,999.,2.,-999.,100.])
    result = refinement_selection_metrics(quality, case['labels'], case['identities'],
                                         case['sources'], (1,), selection_scores=allocation)
    selected = result['budgets']['1']
    assert selected['selected_pool_ids'] == [1]
    assert selected['final_top1'] > .89 and selected['union_top1'] > .89
    assert result['unpaired_initials'] == 1


def test_dual_head_checkpoint_requires_explicit_supervision():
    checkpoint = {'architecture': 'structural_quality_ranker', 'epoch': 1, 'optimizer_steps': 1,
                  'metadata': {'task':'quality', 'arguments': {'quality_weight':0., 'future_weight':1.}},
                  'head_supervision': {'rank':True, 'quality_aux':False, 'future':True}}
    assert checkpoint_capabilities(checkpoint)['future']
    checkpoint['metadata']['arguments']['future_weight'] = 0.
    try:
        checkpoint_capabilities(checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError('unsupervised future head accepted')


def test_accumulated_task_gradients_match_joint_objective():
    case = example(); future = future_utility_case(case)
    accumulated = StructuralQualityRanker(16,16)
    combined = copy.deepcopy(accumulated)
    structural_ranking_loss(score_case(accumulated,case,'cpu'),case['labels']).backward()
    (.7*structural_ranking_loss(score_case(accumulated,future,'cpu'),future['labels'])).backward()
    loss = (structural_ranking_loss(score_case(combined,case,'cpu'),case['labels'])
            + .7*structural_ranking_loss(score_case(combined,future,'cpu'),future['labels']))
    loss.backward()
    for (name, left), (_, right) in zip(accumulated.named_parameters(),combined.named_parameters()):
        if right.grad is not None:
            assert left.grad is not None and torch.allclose(left.grad,right.grad,atol=1e-6),name


def test_joint_training_and_native_free_future_inference(target='final'):
    from test_structural_ranking_data import archive_fixture, state_hash, file_digest
    from scripts.train_structural_ranker import main
    from scripts.rank_structural_archive import rank_archive
    with tempfile.TemporaryDirectory(prefix='future-ranking-test-') as temporary:
        root = Path(temporary)
        arguments = []
        for split, name, group in [('train','1ABC_A_B_L','g1'), ('val','2ABC_A_B_L','g2')]:
            archive, labels = archive_fixture()
            candidate = copy.deepcopy(archive['pool'][0])
            candidate['candidate_id'] = ['sdpi',1]
            candidate['state']['poi'] += 2
            candidate['state_sha256'] = state_hash(candidate['state'])
            pair = copy.deepcopy(archive['pairs'][0])
            pair.update(pool_index=1, candidate_id=['sdpi',1],
                        initial_state_sha256=candidate['state_sha256'])
            pair['final']['poi'] -= 1
            pair['final_state_sha256'] = state_hash(pair['final'])
            archive['pool'].append(candidate); archive['pairs'].append(pair)
            archive.update(case=name, group=group, split=split)
            labels['case'] = name
            labels['labels'].append({'pool_index':1, 'valid':True,
                                     'initial':{'dockq':.5}, 'final':{'dockq':.1}})
            manifest = dict(cases=[name], checkpoint_sha256='test', steps=20, pool={}, selection={})
            archive['manifest'] = manifest
            poses, targets = root/(split+'_archives'), root/(split+'_labels')
            poses.mkdir(); targets.mkdir()
            (poses/'manifest.json').write_text(json.dumps(manifest))
            torch.save(archive, poses/f'{name}.pt')
            digest = file_digest(poses/f'{name}.pt')
            (poses/f'{name}.complete.json').write_text(json.dumps({'archive_sha256':digest}))
            labels['archive_sha256'] = digest
            (targets/f'{name}.json').write_text(json.dumps(labels))
            (targets/'manifest.json').write_text(json.dumps({'archive_manifest_sha256':file_digest(poses/'manifest.json')}))
            arguments += ['--'+split+'-archives',str(poses),'--'+split+'-labels',str(targets)]
        out = root/'out'
        main(arguments+['--out',str(out),'--epochs','1','--hidden-dim','16','--future-weight','1',
                        '--future-target',target])
        rejected_out = root/'budget_out'
        try:
            main(arguments+['--out',str(rejected_out),'--epochs','1','--hidden-dim','16',
                            '--future-weight','1','--budget-weight','1'])
        except ValueError as error:
            assert 'complete training outcomes' in str(error)
            assert not rejected_out.exists(), 'invalid budget experiment must fail before output creation'
        else:
            raise AssertionError('all-candidates selection passed as a useful top64 budget experiment')
        checkpoint = torch.load(out/'best.pt',weights_only=False)
        torch.manual_seed(7315)
        fresh = StructuralQualityRanker(16,16)
        assert not torch.equal(fresh.future_head.weight,checkpoint['state_dict']['future_head.weight'])
        assert not torch.equal(fresh.rank_head.weight,checkpoint['state_dict']['rank_head.weight'])
        assert checkpoint['head_supervision']['future']
        assert checkpoint['metadata']['future_target'] == f'observed_{target}_dockq'
        assert checkpoint['metadata']['arguments']['objective'] == 'decision'
        assert checkpoint['validation']['future_refinement_selection']['complete_pair_coverage_cases'] == 1
        report = rank_archive(out/'best.pt',root/'val_archives/2ABC_A_B_L.pt',
                              task='future', candidate_scope='initial-pool')
        assert report['task'] == f'future_{target}_utility'
        assert len(report['rankings']['initial']) == 2
        assert all(row['stage'] == 'initial' for row in report['rankings']['initial'])
        from tricoopdiff.structural_ranking_runtime import StructuralRankingRuntime
        archived = torch.load(root/'val_archives/2ABC_A_B_L.pt',weights_only=False)
        runtime = StructuralRankingRuntime(out/'best.pt',archived['manifest'],task='future')
        selected, details = runtime.select(archived['input_graphs'],archived['pool'],1)
        assert selected == [report['rankings']['initial'][0]['pool_index']]
        assert details['randomized_inclusion_probabilities'] is False
        paired = [pair for pair in archived['pairs'] if pair['pool_index'] in selected]
        post = runtime.rank_pairs(archived['input_graphs'],archived['pool'],paired)
        assert post['task'] == 'quality'
        assert post['rankings']['final'][0]['pool_index'] == selected[0]
        assert len(post['rankings']['union']) == 2
        for key, value in [('steps',21),('checkpoint_sha256','wrong'),('pool',{'changed':True})]:
            try:
                StructuralRankingRuntime(out/'best.pt',{**archived['manifest'],key:value},task='future')
            except ValueError:
                pass
            else:
                raise AssertionError('changed generator contract accepted')


def test_union_training_and_native_free_future_inference():
    test_joint_training_and_native_free_future_inference(target='union')


if __name__ == '__main__':
    test_future_uses_only_initial_graphs_and_observed_final_labels()
    test_future_scores_allocate_compute_quality_scores_choose_output()
    test_dual_head_checkpoint_requires_explicit_supervision()
    test_accumulated_task_gradients_match_joint_objective()
    test_joint_training_and_native_free_future_inference()
    test_union_training_and_native_free_future_inference()
    print('Future utility: 6 tests passed')
