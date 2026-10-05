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
    except make_deepternary_index.UnsupportedLigandGraphError as error:
        assert 'disconnected fragments' in str(error)
    else:
        raise AssertionError('distant ligand fragments must not be joined')


def test_build_index_classifies_unsupported_ligand_graph():
    def reject(*args, **kwargs):
        raise make_deepternary_index.UnsupportedLigandGraphError('disconnected')

    original = make_deepternary_index.export_case
    try:
        make_deepternary_index.export_case = reject
        entries, rejected = make_deepternary_index.build_index(
            ['case_A_B_LIG'], '.', '.',
        )
    finally:
        make_deepternary_index.export_case = original
    assert entries == []
    assert rejected == [{
        'case': 'case_A_B_LIG',
        'code': 'unsupported_ligand_graph',
        'reason': 'disconnected',
    }]


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
        {'representative': [f'{i:04d}_A_B_L'], 'items': [f'{i:04d}_A_B_L']}
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


def test_written_index_preserves_explicit_split_label(tmp_path):
    destination = tmp_path / 'ternary_val.json'
    make_deepternary_index.write_index(
        destination, [], [], {'protocol': 'test', 'split': 'val'})
    import json
    payload = json.loads(destination.read_text(encoding='utf-8'))
    assert payload['metadata']['split'] == 'val'


def test_pdb_groups_preserve_transitive_source_clusters():
    clusters = [
        {'items': ['1ABC_A_B_L', '2ABC_A_B_L'], 'representative': ['2ABC_A_B_L']},
        {'items': ['1abc_C_D_L', '3ABC_A_B_L'], 'representative': []},
        {'items': ['4ABC_A_B_L'], 'representative': []},
    ]
    train, val = make_deepternary_index.split_clusters(clusters, .5, 0)
    assert sorted(len(c['items']) for c in train + val) == [1, 4]
    payloads = [{'entries': [{'case': c} for c in make_deepternary_index.cluster_members(groups)],
                 'clusters': groups} for groups in (train, val)]
    report = make_deepternary_index.audit_partition(*payloads, source_clusters=clusters)
    assert report['pdb_overlap'] == 0 and report['source_clusters_checked'] == 3


def test_partition_gate_rejects_pdb_and_source_cluster_leakage():
    left = {'entries': [{'case': '1ABC_A_B_L'}]}
    for right_case, sources, expected in (
            ('1abc_C_D_L', [], 'PDB overlap'),
            ('2ABC_A_B_L', [{'items': ['1ABC_A_B_L', '2ABC_A_B_L']}], 'published cluster')):
        try:
            make_deepternary_index.audit_partition(
                left, {'entries': [{'case': right_case}]}, source_clusters=sources)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError('leaking partitions were accepted')


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


def test_ligand_centroid_matches_retained_graph_atoms(tmp_path):
    import sys
    import numpy as np
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from featurize_common import _ligand_centroid

    # RemoveHs retains isotope hydrogens; a heavy-only centroid is incorrect.
    mol = Chem.AddHs(Chem.MolFromSmiles('[2H]C'))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(index, Point3D(float(index * 2), 0., 0.))
    mol.AddConformer(conformer)
    path = tmp_path / 'isotope.sdf'
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    for remove_hs in (True, False):
        retained = Chem.RemoveHs(mol) if remove_hs else mol
        expected = retained.GetConformer().GetPositions().mean(axis=0)
        assert np.allclose(_ligand_centroid(str(path), remove_hs), expected), \
            'native ligand frame must use the same atoms as the input graph'


def test_protein_local_geometry_is_intrinsic_masked_and_sidechain_sensitive():
    import sys
    import numpy as np
    import torch
    import prody as pr
    from unittest.mock import patch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from featurize_common import protein_local_geometry

    pdb = pr.AtomGroup('monomer')
    xyz = np.array([[-1., 0., 0.], [0., 0., 0.], [1., 1., 0.], [2., 1., 0.],
                    [0., -1., 1.], [1., -2., 1.], [3., 0., 0.], [4., 0., 0.], [5., 1., 0.]])
    pdb.setCoords(xyz)
    pdb.setNames(['N', 'CA', 'C', 'O', 'CB', 'OG', 'N', 'CA', 'C'])
    pdb.setResnames(['SER'] * 6 + ['GLY'] * 3)
    pdb.setResnums([1] * 6 + [2] * 3)
    pdb.setChids(['A'] * 9)
    receptor = {'coords': torch.tensor(xyz[[1, 7]])[None]}
    with patch('prody.parsePDB', return_value=pdb):
        original = protein_local_geometry('unused.pdb', receptor)
        assert original.shape == (1, 2, 32) and torch.isfinite(original).all()
        assert original[0, 1, 12] == 0 and original[0, 1, 28:].sum() == 0
        assert original[0, 0, 28] == 1, 'present SER chi must not be masked'
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        pdb.setCoords(xyz @ rotation + 37.)
        moved = protein_local_geometry('unused.pdb', {'coords': receptor['coords'] @ torch.tensor(rotation) + 9.})
        assert torch.allclose(original, moved, atol=1e-10)
        changed = xyz.copy()
        changed[5] += [1., .5, .8]
        pdb.setCoords(changed)
        assert not torch.allclose(original, protein_local_geometry('unused.pdb', receptor))
        try:
            protein_local_geometry('unused.pdb', {'coords': receptor['coords'].flip(1)})
        except ValueError:
            pass
        else:
            raise AssertionError('misordered CA template accepted local sidechain features')
