"""Lightweight checks for the official DeepTernary index pipeline."""

import importlib.util
from pathlib import Path


_DEEPTERNARY_SCRIPT = (
    Path(__file__).resolve().parents[1] / 'scripts' / 'make_deepternary_index.py'
)
_DEEPTERNARY_SPEC = importlib.util.spec_from_file_location(
    'make_deepternary_index', _DEEPTERNARY_SCRIPT,
)
make_deepternary_index = importlib.util.module_from_spec(_DEEPTERNARY_SPEC)
_DEEPTERNARY_SPEC.loader.exec_module(make_deepternary_index)


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
