"""Prepare data regression checks. Run with pytest or execute this file directly."""

if __name__ == "__main__":
    from helpers import run_tests

    run_tests(__file__)

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from rdkit import Chem

import featurize_common
import make_deepternary_index
import prepare_local_mgd_panel
from helpers import chemistry_example
from prepare_validation_index import prepare_training_index
from restore_ligand_chemistry import restore_ligand


# Prepare data


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


def test_written_index_preserves_explicit_split_label(tmp_path):
    destination = tmp_path / 'ternary_val.json'
    make_deepternary_index.write_index(
        destination, [], [], {'protocol': 'test', 'split': 'val'})
    import json
    payload = json.loads(destination.read_text(encoding='utf-8'))
    assert payload['metadata']['split'] == 'val'


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
    import numpy as np
    from rdkit import Chem
    from rdkit.Geometry import Point3D

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
    import numpy as np
    import torch
    import prody as pr
    from unittest.mock import patch

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


# Validation index

def test_filter_validation_clusters_without_changing_training(tmp_path):
    config = {}
    inputs = {
        'train_index': {'entries': [{'case': 'AAAA_A_B_L'}],
                        'clusters': [{'items': ['AAAA_A_B_L']}]},
        'val_index': {'entries': [{'case': c} for c in
                                 ['AAAA_X_Y_L', 'BBBB_A_B_L', 'CCCC_A_B_L']],
                      'clusters': [{'items': ['AAAA_X_Y_L', 'BBBB_A_B_L']},
                                   {'items': ['CCCC_A_B_L']}]},
        'protac_index': {'entries': [{'case': 'DDDD_A_B_L'}]},
        'mgd_index': {'entries': [{'case': 'EEEE_A_B_L'}]},
    }
    for key, payload in inputs.items():
        path = tmp_path / (key + '.json')
        path.write_text(json.dumps(payload))
        config[key] = str(path)
    output = tmp_path / 'derived'
    output.mkdir()
    prepared, report = prepare_training_index(config, output)
    assert json.loads(Path(prepared['train_index']).read_text()) == inputs['train_index']
    validation = json.loads(Path(prepared['val_index']).read_text())
    assert [e['case'] for e in validation['entries']] == ['CCCC_A_B_L']
    assert report['removed_cases'] == ['AAAA_X_Y_L', 'BBBB_A_B_L']
    for key, payload in inputs.items():
        assert json.loads(Path(config[key]).read_text()) == payload
    original_outputs = {path.name: path.read_bytes() for path in output.iterdir()}
    try:
        prepare_training_index(config, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError('existing split outputs must not be overwritten')
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original_outputs


def test_filter_without_clusters_excludes_benchmarks_and_rejects_empty(tmp_path):
    config = {}
    for name, cases in (
        ('train', ['AAAA_A_B_L']), ('val', ['aaaa_X_Y_L', 'BBBB_A_B_L', 'CCCC_A_B_L']),
        ('protac', ['BBBB_X_Y_L']), ('mgd', ['DDDD_A_B_L']),
    ):
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps({'entries': [{'case': case} for case in cases]}))
        config[f'{name}_index'] = str(path)
    prepared, _ = prepare_training_index(config, tmp_path / 'derived')
    assert json.loads(Path(prepared['val_index']).read_text())['entries'] == [
        {'case': 'CCCC_A_B_L'}]
    Path(config['val_index']).write_text(json.dumps({'entries': [{'case': 'BBBB_A_B_L'}]}))
    try:
        prepare_training_index(config, tmp_path / 'empty')
    except ValueError as error:
        assert 'no validation samples' in str(error)
    else:
        raise AssertionError('empty validation must not be accepted')
    assert not (tmp_path / 'empty').exists(), 'failed preparation must not write outputs'


# Prepare local mgd panel


def test_select_panel_uses_predeclared_evenly_spaced_positions():
    cases = [f"case_{index:02d}" for index in range(94)]
    assert prepare_local_mgd_panel.select_panel(cases) == [
        "case_00", "case_23", "case_46", "case_69", "case_93",
    ]


def test_select_panel_rejects_changed_official_cohort():
    try:
        prepare_local_mgd_panel.select_panel(["case"] * 94)
    except ValueError as error:
        assert "94 unique" in str(error)
    else:
        raise AssertionError("duplicate official MGD cases must be rejected")


def test_full_entries_record_explicit_pdb_fallback(tmp_path):
    ternary = tmp_path / "ternary"
    restored = tmp_path / "restored"
    case = "case"
    (ternary / case).mkdir(parents=True)
    for name in ("protein1.pdb", "protein2.pdb", "ligand.pdb", "gt_complex.pdb"):
        (ternary / case / name).write_text("placeholder")
    entries, fallbacks = prepare_local_mgd_panel.build_entries(
        [case], ternary, restored, allow_pdb_fallback=True,
    )
    assert fallbacks == [case]
    assert entries[0]["ligand"].endswith("ligand.pdb")
    assert entries[0]["ligand_chemistry"] == "pdb_inferred_fallback"


# Paired cache


def test_read_only_featurization_follower_does_not_generate_missing_case():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs = []
        for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
            path = root / name; path.write_text(name); inputs.append(path.resolve())
        cache = root / 'cache'; cache.mkdir()
        try:
            featurize_common.featurize_complex(
                'demo', *inputs, Namespace(receptor_radius=30), 'cpu',
                cache_dir=cache, cache_read_only=True, cache_wait_seconds=0.)
        except TimeoutError as error:
            assert 'read-only featurization follower' in str(error)
        else:
            raise AssertionError('read-only follower attempted cold featurization')
        assert not list(cache.iterdir())


def test_failed_featurization_publish_preserves_cache_and_removes_partial_file(tmp_path):
    inputs = []
    for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
        path = tmp_path / name
        path.write_text(name)
        inputs.append(str(path.resolve()))
    cache = tmp_path / 'cache'
    cache.mkdir()
    destination = cache / 'demo.pt'
    torch.save({'cache_version': -1}, destination)
    original = destination.read_bytes()
    data = {'coords': torch.zeros(1, 3, 3), 'features': torch.ones(1, 3, 2)}

    def fail_save(payload, path):
        Path(path).write_bytes(b'partial cache')
        raise OSError('intentional interrupted write')

    with patch.object(featurize_common, 'generate_esm_embeddings', return_value={path: [] for path in inputs}), \
            patch.object(featurize_common, 'featurize_subgraph', return_value=SimpleNamespace(success=[True])), \
            patch.object(featurize_common, 'extract_receptor', side_effect=lambda *args: dict(data)), \
            patch.object(featurize_common, 'extract_ligand', return_value=dict(data)), \
            patch.object(featurize_common, 'protein_local_geometry', return_value=torch.ones(1, 3, 32)), \
            patch.object(featurize_common, '_true_relative_offsets', return_value=(np.zeros(3), np.zeros(3))), \
            patch('torch.save', side_effect=fail_save):
        try:
            featurize_common.featurize_complex('demo', *inputs, Namespace(remove_hs=True), 'cpu', cache_dir=cache)
        except OSError as error:
            assert 'intentional' in str(error)
        else:
            raise AssertionError('cache write failure was hidden')
    assert destination.read_bytes() == original, 'failed write replaced the original cache'
    assert list(cache.iterdir()) == [destination], 'failed write left a partial cache behind'


def test_v8_cache_enrichment_preserves_embeddings_without_esm(tmp_path):
    inputs = []
    for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
        path = tmp_path / name
        path.write_text(name)
        inputs.append(str(path.resolve()))
    args = Namespace(remove_hs=True)
    data = {'coords': torch.zeros(3, 3), 'features': torch.ones(3, 2)}
    payload = {'cache_version': 8,
               'source_signature': featurize_common._featurize_signature(*inputs, args, cache_version=None),
               'e3_data': dict(data), 'poi_data': dict(data), 'ligand_data': dict(data),
               'receptor_feat_dim': 2}
    torch.save(payload, tmp_path / 'demo.pt')
    with patch.object(featurize_common, 'generate_esm_embeddings', side_effect=AssertionError('ESM was rebuilt')), \
            patch.object(featurize_common, 'protein_local_geometry', return_value=torch.ones(1, 3, 32)) as geometry:
        e3, poi, _, _ = featurize_common.featurize_complex('demo', *inputs, args, 'cpu', cache_dir=tmp_path)
        assert geometry.call_count == 2
        assert e3['local_geometry'].shape == poi['local_geometry'].shape == (1, 3, 32)
        again = featurize_common.featurize_complex('demo', *inputs, args, 'cpu', cache_dir=tmp_path)
        assert geometry.call_count == 2 and torch.equal(again[0]['features'], e3['features'])
    upgraded = torch.load(tmp_path / 'demo.pt', weights_only=False)
    assert upgraded['cache_version'] == 9
    assert upgraded['source_signature'] != payload['source_signature']
    assert upgraded['e3_data']['local_geometry'].shape == (3, 32)


def test_cache_only_never_rebuilds_missing_or_stale_features(tmp_path):
    inputs = []
    for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
        path = tmp_path / name
        path.write_text(name)
        inputs.append(str(path.resolve()))
    with patch.object(featurize_common, 'generate_esm_embeddings', side_effect=AssertionError('ESM must not run')):
        for stale in (False, True):
            if stale:
                torch.save({'cache_version': -1}, tmp_path / 'demo.pt')
            try:
                featurize_common.featurize_complex('demo', *inputs, Namespace(remove_hs=True), 'cpu',
                                           cache_dir=tmp_path, cache_only=True)
            except FileNotFoundError as error:
                assert 'ESM rebuilding disabled' in str(error)
            else:
                raise AssertionError('missing/stale cache was accepted')


def test_v7_cache_only_corrects_center_without_rebuilding_esm(tmp_path):
    inputs = []
    for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
        path = tmp_path / name
        path.write_text(name)
        inputs.append(str(path.resolve()))
    args = Namespace(remove_hs=True)
    data = {'coords': torch.arange(9).float().reshape(3, 3), 'features': torch.ones(3, 2)}
    torch.save({'cache_version': 7,
        'source_signature': featurize_common._featurize_signature(*inputs, args, cache_version=None),
        'e3_data': dict(data), 'poi_data': dict(data), 'ligand_data': dict(data), 'receptor_feat_dim': 2}, tmp_path / 'demo.pt')
    with patch.object(featurize_common, 'generate_esm_embeddings', side_effect=AssertionError('ESM must not run')), \
            patch.object(featurize_common, 'protein_local_geometry', return_value=torch.ones(1, 3, 32)), \
            patch.object(featurize_common, '_ligand_centroid', return_value=np.array([4., 5., 6.])), \
            patch.object(featurize_common, '_protein_ca_centroid', return_value=np.array([1., 1., 1.])):
        e3, _, ligand, _ = featurize_common.featurize_complex('demo', *inputs, args, 'cpu',
                                                     cache_dir=tmp_path, cache_only=True)
    assert torch.equal(e3['features'][0], data['features'])
    assert torch.equal(ligand['coords'].mean(1)[0], torch.tensor([3., 4., 5.]))
    assert torch.equal(torch.cdist(ligand['coords'][0], ligand['coords'][0]), torch.cdist(data['coords'], data['coords']))


# Restore ligand chemistry

def test_restores_bond_order_and_charge_without_reordering(tmp_path):
    pdb, ccd, positions = chemistry_example(tmp_path)
    output = tmp_path / 'restored.sdf'
    report = restore_ligand(pdb, ccd, output)
    molecule = Chem.SDMolSupplier(str(output), removeHs=True)[0]
    assert molecule is not None
    assert Chem.GetFormalCharge(molecule) == -1
    assert molecule.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.DOUBLE
    assert report['atom_names_in_order'] == ['C2', 'C1', 'O1', 'O2']
    assert np.allclose(molecule.GetConformer().GetPositions(), positions, atol=1e-5)
    assert molecule.HasProp('chemistry_provenance')


def test_missing_atoms_are_not_invented(tmp_path):
    pdb, ccd, _ = chemistry_example(tmp_path)
    pdb.write_text('\n'.join(line for line in pdb.read_text().splitlines()
                              if line[12:16].strip() != 'O2') + '\n')
    output = tmp_path / 'must_not_exist.sdf'
    try:
        restore_ligand(pdb, ccd, output)
    except ValueError as error:
        assert 'missing' in str(error) and 'O2' in str(error)
    else:
        raise AssertionError('incomplete component was silently reconstructed')
    assert not output.exists()


def test_uppercase_ccd_halogen_symbols(tmp_path):
    pdb, ccd, _ = chemistry_example(tmp_path)
    ccd.write_text(ccd.read_text().replace('O2 O -1', 'O2 CL 0'))
    lines = pdb.read_text().splitlines()
    pdb.write_text('\n'.join(line[:76] + 'CL' + line[78:]
                              if line[12:16].strip() == 'O2' else line
                              for line in lines) + '\n')
    output = tmp_path / 'halogen.sdf'
    restore_ligand(pdb, ccd, output)
    molecule = Chem.SDMolSupplier(str(output))[0]
    assert molecule.GetAtomWithIdx(3).GetSymbol() == 'Cl'
    assert Chem.GetFormalCharge(molecule) == 0


def test_uses_explicit_ccd_aliases_without_guessing(tmp_path):
    pdb, ccd, coordinates = chemistry_example(tmp_path)
    content = ccd.read_text().replace('_chem_comp_atom.type_symbol',
                                    '_chem_comp_atom.alt_atom_id\n_chem_comp_atom.type_symbol')
    for before, after in [('O2 O -1', 'O2 O2 O -1'), ('C1 C 0', 'CX C1 C 0'),
                          ('C2 C 0', 'C2 C2 C 0'), ('O1 O 0', 'O1 O1 O 0'),
                          ('C1 O1', 'CX O1'), ('C1 C2', 'CX C2'), ('C1 O2', 'CX O2')]:
        content = content.replace(before, after)
    ccd.write_text(content)
    output = tmp_path / 'aliased.sdf'
    report = restore_ligand(pdb, ccd, output)
    assert report['atom_name_aliases'] == {'C1': 'CX'}
    assert report['atom_names_in_order'] == ['C2', 'C1', 'O1', 'O2']
    molecule = Chem.SDMolSupplier(str(output))[0]
    assert np.allclose(molecule.GetConformer().GetPositions(), coordinates)


def test_undefined_bond_endpoint_is_not_treated_as_hydrogen(tmp_path):
    pdb, ccd, _ = chemistry_example(tmp_path)
    ccd.write_text(ccd.read_text() + 'C1 UNKNOWN SING\n')
    output = tmp_path / 'must_not_exist.sdf'
    try:
        restore_ligand(pdb, ccd, output)
    except ValueError as error:
        assert 'undefined atom' in str(error)
    else:
        raise AssertionError('malformed CCD bond was silently ignored')
    assert not output.exists()
