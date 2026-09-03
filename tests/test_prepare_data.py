"""Lightweight checks for the Phase 0 ternary-data index pipeline."""

import json
import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_data.py'
_SPEC = importlib.util.spec_from_file_location('prepare_data', _SCRIPT)
prepare_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_data)
split_ternary_entries = prepare_data.split_ternary_entries
write_training_splits = prepare_data.write_training_splits

_DEEPTERNARY_SCRIPT = (
    Path(__file__).resolve().parents[1] / 'scripts' / 'make_deepternary_index.py'
)
_DEEPTERNARY_SPEC = importlib.util.spec_from_file_location(
    'make_deepternary_index', _DEEPTERNARY_SCRIPT,
)
make_deepternary_index = importlib.util.module_from_spec(_DEEPTERNARY_SPEC)
_DEEPTERNARY_SPEC.loader.exec_module(make_deepternary_index)


def _entries(n_pdb=10, copies=2):
    return [
        {
            'case': f'{pdb_id}_A_B_LIG_{copy}',
            'pdb_id': pdb_id,
            'protein1': f'data/{pdb_id}/protein1.pdb',
            'protein2': f'data/{pdb_id}/protein2.pdb',
            'ligand': f'data/{pdb_id}/ligand.pdb',
        }
        for pdb_id in (f'{index:04d}' for index in range(n_pdb))
        for copy in range(copies)
    ]


def test_split_is_deterministic_and_groups_pdb_ids():
    entries = _entries()
    first = split_ternary_entries(entries, seed=7)
    second = split_ternary_entries(list(reversed(entries)), seed=7)

    first_cases = {key: sorted(x['case'] for x in value) for key, value in first.items()}
    second_cases = {key: sorted(x['case'] for x in value) for key, value in second.items()}
    assert first_cases == second_cases, 'split must not depend on input ordering'

    pdb_sets = {
        split: {entry['pdb_id'] for entry in split_entries}
        for split, split_entries in first.items()
    }
    assert pdb_sets['train'].isdisjoint(pdb_sets['val'])
    assert pdb_sets['train'].isdisjoint(pdb_sets['test'])
    assert pdb_sets['val'].isdisjoint(pdb_sets['test'])
    assert sum(len(value) for value in first.values()) == len(entries)


def test_write_training_splits_uses_loader_schema(tmp_path):
    splits = split_ternary_entries(_entries(n_pdb=5, copies=1), seed=3)
    write_training_splits(splits, str(tmp_path))

    for split in ('train', 'val', 'test'):
        payload = json.loads((tmp_path / f'ternary_{split}.json').read_text())
        assert list(payload) == ['entries']
        for entry in payload['entries']:
            assert {'case', 'protein1', 'protein2', 'ligand'} <= set(entry)


def test_split_rejects_invalid_fractions():
    try:
        split_ternary_entries(_entries(), val_fraction=0.6, test_fraction=0.4)
    except ValueError as error:
        assert 'sum to < 1' in str(error)
    else:
        raise AssertionError('invalid split fractions must fail')


def test_deepternary_cluster_selection_uses_published_representatives():
    clusters = [
        {'representative': ['rep_a', 'rep_b'], 'items': ['a', 'b']},
        {'representative': [], 'items': ['fallback', 'other']},
        {'representative': [], 'items': []},
    ]
    selected = make_deepternary_index.select_cluster_cases(
        clusters, mode='representative',
    )
    assert selected == ['rep_a', 'fallback']
    assert make_deepternary_index.select_cluster_cases(clusters, mode='all') == [
        'a', 'b', 'fallback', 'other',
    ]
    assert make_deepternary_index.select_cluster_cases(
        clusters, mode='representative', excluded={'rep_a', 'fallback'},
    ) == ['rep_b', 'other']


def test_deepternary_case_list_reads_first_csv_column(tmp_path):
    split = tmp_path / 'split.csv'
    split.write_text('case_a,unbound1,unbound2\ncase_b,x,y\n', encoding='utf-8')
    assert make_deepternary_index.read_case_list(split) == ['case_a', 'case_b']


def test_deepternary_heavy_atom_count_excludes_hydrogen(tmp_path):
    pdb = tmp_path / 'ligand.pdb'
    pdb.write_text(
        'HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C  \n'
        'HETATM    2  H1  LIG A   1       1.000   0.000   0.000  1.00  0.00           H  \n'
        'HETATM    3  O1  LIG A   1       0.000   1.000   0.000  1.00  0.00           O  \n',
        encoding='utf-8',
    )
    assert make_deepternary_index.count_pdb_heavy_atoms(pdb) == 2
