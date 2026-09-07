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


def test_ligand_graph_gate_rejects_bondless_input(tmp_path):
    pdb = tmp_path / 'ligand.pdb'
    pdb.write_text(
        'HETATM    1 ZN   ZN  A   1       0.000   0.000   0.000  1.00  0.00          ZN  \n',
        encoding='utf-8',
    )
    try:
        make_deepternary_index.validate_ligand_graph(pdb)
    except RuntimeError as error:
        assert 'no inferred bonds' in str(error)
    else:
        raise AssertionError('bondless ligand must be rejected before training')


def test_ligand_graph_repairs_plausible_missing_pdb_bond(tmp_path):
    pdb = tmp_path / 'ligand.pdb'
    pdb.write_text(
        'HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C  \n'
        'HETATM    2  C2  LIG A   1       1.500   0.000   0.000  1.00  0.00           C  \n'
        'HETATM    3  C3  LIG A   1       3.320   0.000   0.000  1.00  0.00           C  \n'
        'HETATM    4  C4  LIG A   1       4.820   0.000   0.000  1.00  0.00           C  \n'
        'END\n',
        encoding='utf-8',
    )
    make_deepternary_index.validate_ligand_graph(pdb)
    assert 'CONECT    2    3' in pdb.read_text(encoding='utf-8')


def test_ligand_graph_rejects_distant_fragments(tmp_path):
    pdb = tmp_path / 'ligand.pdb'
    pdb.write_text(
        'HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C  \n'
        'HETATM    2  C2  LIG A   1       1.500   0.000   0.000  1.00  0.00           C  \n'
        'HETATM    3  C3  LIG A   1      10.000   0.000   0.000  1.00  0.00           C  \n'
        'HETATM    4  C4  LIG A   1      11.500   0.000   0.000  1.00  0.00           C  \n'
        'END\n',
        encoding='utf-8',
    )
    try:
        make_deepternary_index.validate_ligand_graph(pdb)
    except RuntimeError as error:
        assert 'disconnected fragments' in str(error)
    else:
        raise AssertionError('distant ligand fragments must not be joined')


def test_joint_cluster_intersection_merges_relations_and_excludes_pdbs():
    protac = [
        {'representative': ['a_A_B_X'], 'items': ['a_A_B_X', 'b_A_B_X']},
        {'representative': [], 'items': ['c_A_B_X']},
    ]
    mgd = [
        {'representative': ['b_A_B_X'], 'items': ['b_A_B_X', 'c_A_B_X']},
        {'representative': [], 'items': ['not_shared_A_B_X']},
    ]
    merged = make_deepternary_index.merge_cluster_intersection(
        [protac, mgd], excluded_pdb_ids={'C'},
    )
    assert merged == [{
        'representative': ['b_A_B_X'],
        'items': ['b_A_B_X'],
    }]


def test_cluster_split_is_deterministic_and_disjoint():
    clusters = [
        {'representative': [f'case_{i}'], 'items': [f'case_{i}']}
        for i in range(10)
    ]
    train_a, val_a = make_deepternary_index.split_clusters(clusters, 0.2, 7)
    train_b, val_b = make_deepternary_index.split_clusters(clusters, 0.2, 7)
    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 8
    assert len(val_a) == 2
    assert {c['items'][0] for c in train_a}.isdisjoint(
        {c['items'][0] for c in val_a}
    )


def test_repository_joint_pool_contract():
    root = Path(__file__).resolve().parents[1]
    protac_path = root / 'DeepTernary/data/PROTAC/train_clusters_poc6.json'
    mgd_path = (
        root / 'DeepTernary/data/MolecularGlue/train_clusters_poc3_exUNL.json'
    )
    if not protac_path.is_file() or not mgd_path.is_file():
        return
    import json
    protac = json.loads(protac_path.read_text())
    mgd = json.loads(mgd_path.read_text())
    benchmark_cases = []
    for path in (
        root / 'DeepTernary/data/PROTAC/protac22.txt',
        root / 'DeepTernary/data/MolecularGlue/test_all.txt',
    ):
        benchmark_cases.extend(make_deepternary_index.read_case_list(path))
    merged = make_deepternary_index.merge_cluster_intersection(
        [protac, mgd],
        {make_deepternary_index.case_pdb_id(case) for case in benchmark_cases},
    )
    assert len(merged) == 653
    assert sum(len(cluster['items']) for cluster in merged) == 7198
