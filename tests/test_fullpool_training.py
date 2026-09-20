"""Exercise the real trainer on 256 small synthetic candidates, not benchmark efficacy."""

import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch

from test_structural_ranking_data import archive_fixture, file_digest, state_hash
from scripts.train_structural_ranker import main, selection_score
from tricoopdiff.structural_ranker import StructuralQualityRanker


def test_fullpool_budget_training_and_checkpoint():
    with tempfile.TemporaryDirectory(prefix='fullpool-training-') as temporary:
        root = Path(temporary)
        arguments = []
        for split, case in (('train', '1ABC_A_B_L'), ('val', '2ABC_A_B_L')):
            base, _ = archive_fixture()
            archive = {**base, 'case': case, 'split': split, 'group': case, 'pool': [], 'pairs': []}
            targets = []
            for index in range(256):
                source = 'sdpi' if index % 2 else 'bitseed'
                candidate = copy.deepcopy(base['pool'][0])
                candidate.update(source=source, candidate_id=[source, index])
                candidate['state']['poi'] += index / 256.
                candidate['state']['ligand'] += index / 512.
                candidate['state_sha256'] = state_hash(candidate['state'])
                pair = copy.deepcopy(base['pairs'][0])
                pair.update(pool_index=index, source=source, candidate_id=[source, index],
                            initial_state_sha256=candidate['state_sha256'])
                pair['final']['poi'] += index / 512.
                pair['final_state_sha256'] = state_hash(pair['final'])
                archive['pool'].append(candidate)
                archive['pairs'].append(pair)
                targets.append({'pool_index': index, 'valid': True,
                                'initial': {'dockq': .1 + .2 * index / 256.},
                                'final': {'dockq': .8 * (255 - index) / 256.}})
            manifest = {'cases': [case], 'checkpoint_sha256': 'synthetic', 'steps': 20,
                        'pool': {'size': 256}, 'selection': 'complete_existing_pool_v1'}
            archive['manifest'] = manifest
            poses, labels = root / f'{split}_archives', root / f'{split}_labels'
            poses.mkdir(); labels.mkdir()
            (poses / 'manifest.json').write_text(json.dumps(manifest))
            torch.save(archive, poses / f'{case}.pt')
            digest = file_digest(poses / f'{case}.pt')
            (poses / f'{case}.complete.json').write_text(json.dumps({'archive_sha256': digest}))
            (labels / f'{case}.json').write_text(json.dumps({'case': case, 'archive_sha256': digest,
                                                           'labels': targets}))
            (labels / 'manifest.json').write_text(json.dumps({
                'archive_manifest_sha256': file_digest(poses / 'manifest.json')}))
            arguments += [f'--{split}-archives', str(poses), f'--{split}-labels', str(labels)]
        out = root / 'run'
        main(arguments + ['--out', str(out), '--epochs', '3', '--max-hours', '0.000000001',
                          '--save-epoch-checkpoints', '--hidden-dim', '16',
                          '--shuffle-repeats', '0', '--future-weight', '1', '--future-target', 'union',
                          '--budget-weight', '1', '--checkpoint-task', 'future', '--backward-chunk-size', '4',
                          '--quality-walker-scope', 'future-top64'])
        checkpoint = torch.load(out / 'best.pt', weights_only=False)
        assert checkpoint['epoch'] == 1 and (out / 'epoch_001.pt').exists()
        assert not (out / 'epoch_002.pt').exists(), 'soft deadline must stop at the saved epoch boundary'
        record = json.loads((out / 'history.jsonl').read_text())
        assert record['budget_train_loss'] > 0
        assert checkpoint['metadata']['primary'] == 'future_top64_union_selected_best'
        metrics = checkpoint['validation']
        assert metrics['future_refinement_selection']['complete_pair_coverage_cases'] == 1
        assert selection_score(metrics, 'archived-pairs', 'union') == (
            metrics['future_refinement_selection']['budgets']['64']['mean_union_selected_best'])
        torch.manual_seed(7315)
        initial = StructuralQualityRanker(16, 16)
        for name in ('rank_head.weight', 'future_head.weight', 'encoder.protein_feature_proj.weight'):
            assert not torch.equal(initial.state_dict()[name], checkpoint['state_dict'][name]), name
        assert torch.isfinite(torch.tensor(record['train_loss']))
        print(f'Full-pool integration: 256 candidates, 512 quality states; budget loss={record["budget_train_loss"]:.6f}')


if __name__ == '__main__':
    test_fullpool_budget_training_and_checkpoint()
