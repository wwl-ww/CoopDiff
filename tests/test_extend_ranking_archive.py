"""An extension must preserve original candidates and observed trajectories."""
import copy
from pathlib import Path
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from scripts.extend_ranking_archive import validate_archive, validate_extension, atomic_save
from test_structural_ranking_data import archive_fixture, rejected


def test_missing_and_failed_trajectories_are_distinct():
    archive, _ = archive_fixture()
    archive['split'] = 'train'
    archive['pool'].append(copy.deepcopy(archive['pool'][0]))
    archive['pool'][1]['candidate_id'] = ['sdpi', 1]
    assert validate_archive(archive) == [1]
    archive['pairs'][0].update(valid=False, final=None, final_state_sha256=None)
    assert validate_archive(archive) == [1], 'failed trajectories must not silently be retried'
    archive['split'] = 'val'
    rejected(lambda: validate_archive(archive))


def test_existing_results_cannot_change():
    parent, _ = archive_fixture()
    parent['split'] = 'train'
    extended = copy.deepcopy(parent)
    validate_extension(parent, extended)
    extended['pairs'][0]['source'] = 'bitseed'
    rejected(lambda: validate_extension(parent, extended))
    extended = copy.deepcopy(parent)
    extended['pairs'] = []
    rejected(lambda: validate_extension(parent, extended))
    extended = copy.deepcopy(parent)
    extended['pool'][0]['state']['ligand'] += 1
    rejected(lambda: validate_extension(parent, extended))


def test_atomic_roundtrip_preserves_resume_state():
    parent, _ = archive_fixture()
    parent['split'] = 'train'
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'partial.pt'
        atomic_save(parent, path)
        validate_extension(parent, torch.load(path, weights_only=False))
        assert len(list(Path(temporary).iterdir())) == 1


if __name__ == '__main__':
    test_missing_and_failed_trajectories_are_distinct()
    test_existing_results_cannot_change()
    test_atomic_roundtrip_preserves_resume_state()
    print('Archive extension: 3 tests passed')
