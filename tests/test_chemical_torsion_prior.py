"""Native independence and exact dihedral transfer, not an accuracy claim."""

from pathlib import Path
import sys

import torch
from rdkit import Chem

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_anchored_torsion import fixture
from tricoopdiff.anchored_flow import AnchoredTorsionFlow, wrap_angles
from tricoopdiff.chemical_torsion_prior import independent_conformers, transfer_dihedrals, ordered_graph_sha256


def test_chemical_prior_ignores_all_input_coordinates():
    mol = Chem.MolFromSmiles('CCCCCCCC')
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(index, (index * 20., 19., -9.))
    mol.AddConformer(conformer)
    actual, audit = independent_conformers(mol, seed=7315, count=4)
    mol.RemoveAllConformers()
    expected, other = independent_conformers(mol, seed=7315, count=4)
    assert torch.equal(actual, expected)
    assert audit == other and audit['input_coordinates_used'] is False
    assert actual.shape[1:] == (8, 3)


def test_graph_identity_detects_atom_reordering_not_coordinates():
    mol = Chem.MolFromSmiles('CCCOCCCN')
    identity = ordered_graph_sha256(mol)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    assert ordered_graph_sha256(mol) == identity
    assert ordered_graph_sha256(Chem.RenumberAtoms(mol, list(reversed(range(8))))) != identity


def test_conditioned_prior_preserves_template_origin_and_explicit_random_stream():
    from tricoopdiff.bounded_binary_flow import BoundedBinaryTorsionFlow
    from tricoopdiff.chemical_prior_contract import chemical_prior_config, conditioned_chemical_prior

    strict = AnchoredTorsionFlow(*fixture())
    flow = BoundedBinaryTorsionFlow(strict, (0, 1, 2), (5, 6, 7),
                                   translation_bounds=[.5, .5], rotation_bounds=[.2, .2])
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    conformers = torch.stack([strict.decode(strict.ligand.new_tensor(q))[0]
                              for q in ([.2, -.3, .7], [-.7, .1, -.9])])
    def draw():
        return conditioned_chemical_prior(flow, edges, conformers,
            generator=torch.Generator().manual_seed(7315), config=chemical_prior_config())
    state, audit = draw()
    other, other_audit = draw()
    assert torch.equal(state, other) and audit == other_audit
    assert torch.count_nonzero(state[3:]) == 0, 'known binary pose was noised unnecessarily'
    ligand, poi, _ = flow.decode(state)
    assert torch.allclose(ligand[:3], strict.ligand[:3], atol=1e-12)
    assert torch.allclose(torch.cdist(ligand[5:], poi), torch.cdist(strict.ligand[5:], strict.poi), atol=1e-10)


def test_library_rejects_changed_graph_and_artifact(tmp_path):
    import hashlib
    import json
    from tricoopdiff.chemical_prior_contract import ChemicalPriorLibrary, chemical_prior_config, deployment_conformers

    molecule = Chem.MolFromSmiles('CCCOCCCN')
    config = chemical_prior_config(count=2)
    coords, audit = deployment_conformers(molecule, config)
    artifact = tmp_path / 'library.pt'
    torch.save({'coordinates': coords, 'audit': audit,
                'ordered_graph_sha256': ordered_graph_sha256(molecule)}, artifact)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'root_seed': config['seed'], 'count': 2, 'index_sha256': 'a' * 64,
        'cases': [{'case': 'fixture', 'status': 'prepared', 'path': str(artifact),
                   'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}]}))
    library = ChemicalPriorLibrary(manifest, config, index_sha256='a' * 64)
    assert torch.equal(library.get('fixture', molecule), coords)
    assert torch.equal(library.get('fixture', molecule.ToBinary()), coords)
    try:
        library.get('fixture', Chem.MolFromSmiles('NCCOCCCC'))
    except RuntimeError as error:
        assert 'graph/provenance' in str(error)
    else:
        raise AssertionError('wrong atom order or chemistry was accepted')
    artifact.write_bytes(b'changed')
    try:
        ChemicalPriorLibrary(manifest, config, index_sha256='a' * 64).get('fixture', molecule)
    except RuntimeError as error:
        assert 'hash mismatch' in str(error)
    else:
        raise AssertionError('modified prior artifact was accepted')


def test_mixed_prior_keeps_both_modes_without_perturbing_binary_origin():
    from tricoopdiff.bounded_binary_flow import BoundedBinaryTorsionFlow
    from tricoopdiff.chemical_prior_contract import conditioned_chemical_prior, chemical_prior_config
    strict = AnchoredTorsionFlow(*fixture())
    flow = BoundedBinaryTorsionFlow(strict, (0, 1, 2), (5, 6, 7),
                                   translation_bounds=[.5, .5], rotation_bounds=[.2, .2])
    generator = torch.Generator().manual_seed(7315)
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    components = []
    for _ in range(32):
        state, audit = conditioned_chemical_prior(flow, edges, strict.ligand[None], generator=generator,
            config=chemical_prior_config(uniform_probability=.5))
        assert torch.count_nonzero(state[3:]) == 0
        components.append(audit['component'])
        if audit['component'] == 'chemical':
            assert torch.allclose(state[:3], torch.zeros_like(state[:3]), atol=1e-12)
        else:
            assert state[:3].abs().sum() > 0 and audit['conformer_index'] is None
    assert set(components) == {'chemical', 'uniform'}


def test_transfer_matches_chart_increments_and_preserves_binary_geometry():
    flow = AnchoredTorsionFlow(*fixture())
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    values = flow.ligand.new_tensor([[2.8, -2.9, .7], [-.3, .5, -1.2]])
    conformers = torch.stack([flow.decode(q)[0] for q in values])
    recovered = transfer_dihedrals(flow.ligand, flow.plan, edges, conformers)
    assert torch.allclose(wrap_angles(recovered - values), torch.zeros_like(values), atol=1e-10)
    # Independent frame and internal scale changes must not leak into the chart.
    rotation = flow.ligand.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transformed = conformers @ rotation.T * 1.4 + 22.
    shifted = transfer_dihedrals(flow.ligand, flow.plan, edges, transformed)
    assert torch.allclose(wrap_angles(shifted - values), torch.zeros_like(values), atol=1e-10)
    for q in shifted:
        ligand, poi, _ = flow.decode(q)
        assert torch.equal(ligand[:3], flow.ligand[:3])
        assert torch.allclose(torch.cdist(ligand[5:], poi), torch.cdist(flow.ligand[5:], flow.poi), atol=1e-10)


if __name__ == '__main__':
    test_graph_identity_detects_atom_reordering_not_coordinates()
    test_conditioned_prior_preserves_template_origin_and_explicit_random_stream()
    test_chemical_prior_ignores_all_input_coordinates()
    test_transfer_matches_chart_increments_and_preserves_binary_geometry()
    print('Chemical torsion prior checks passed')
