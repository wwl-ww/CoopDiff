"""Explicit experimental transfer of conserved binary cores into conditioning.

Preserves trusted sides and the original ligand graph used by the model. A
tautomer-normalized graph is used only to certify new chemical correspondences.
The resulting local template hypotheses are not calibrated pose guarantees.
"""

import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .binary_pose_constraints import BinaryPoseSide
from .condition_features import SideCondition, build_condition_features
from .conserved_binary_cores import conserved_binary_cores
from .conserved_core_geometry import core_geometry_evidence
from .tautomer_binary_cores import tautomer_conserved_binary_cores, canonical_tautomer_with_identity


CORE_CONDITION_POLICY = 'experimental_conserved_binary_core_conditioning_v1'


def merge_side_features(base, addition, side_index):
    """Merge one explicitly absent side; do not infer linker from its complement."""
    if side_index not in (0, 1) or bool(base['sides'][side_index, :3].any()):
        raise ValueError('core augmentation can only fill an absent side')
    result = {key: value.clone() for key, value in base.items()}
    result['ligand'][:, side_index] = addition['ligand'][:, side_index]
    column = 4 + 5 * side_index
    result['ligand'][:, column:column + 5] = addition['ligand'][:, column:column + 5]
    if bool((result['ligand'][:, 0] * result['ligand'][:, 1]).any()):
        raise ValueError('conserved binary core overlaps the other known end')
    result['ligand'][:, 3] = (result['ligand'][:, :3].sum(1) == 0).to(result['ligand'])
    result['sides'][side_index] = addition['sides'][side_index]
    for name in ('protein', 'binary'):
        result[f'{name}{side_index + 1}'] = addition[f'{name}{side_index + 1}'].clone()
    return result


def augment_conserved_core_conditions(features, sides, molecule, *, case, case_dir,
                                       chemistry_report, read_protein_coordinates):
    """Recompute graph evidence from hashed binary inputs; never load audit maps.

    Caller must opt in explicitly, retain the returned audit in predictions,
    and run normal constrained-embedding/finite-bound checks. No fallback to a
    relaxed MCS or native-dependent fit is permitted.
    """
    case_dir, chemistry_report = Path(case_dir), Path(chemistry_report)
    records = json.loads(chemistry_report.read_text())['cases']
    matches = [row for row in records if row['case'] == case]
    if len(matches) != 1:
        raise ValueError('chemical core source must identify the case uniquely')
    record = matches[0]
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(case_dir / 'ligand.sdf') != record['full_ligand_sha256']:
        raise ValueError('conserved core full-ligand source hash mismatch')
    source_full = Chem.SDMolSupplier(str(case_dir / 'ligand.sdf'), removeHs=True)[0]
    full = Chem.Mol(molecule)
    if (source_full is None or source_full.GetNumAtoms() != full.GetNumAtoms()
            or source_full.GetNumBonds() != full.GetNumBonds()):
        # Full structural identity, including atom ordering, is checked below;
        # the counts check gives an early diagnostic for a stale feature cache.
        raise ValueError('core source molecule and feature cache counts differ')
    cleaned_stereo = {}
    for name, graph in (('source', source_full), ('feature_cache', full)):
        graph.RemoveAllConformers()
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
        before = [atom.GetChiralTag() for atom in graph.GetAtoms()]
        # Clean before adding identity labels: labels can make chemically
        # identical methyl groups look different and preserve a spurious tag.
        Chem.AssignStereochemistry(graph, cleanIt=True, force=True)
        cleaned_stereo[name] = [index for index, atom in enumerate(graph.GetAtoms())
                               if atom.GetChiralTag() != before[index]]
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    if Chem.MolToSmiles(source_full, isomericSmiles=True) != Chem.MolToSmiles(full, isomericSmiles=True):
        raise ValueError('core source molecule and feature cache atom identities differ')
    for atom in full.GetAtoms():
        atom.SetAtomMapNum(0)
    result, augmented = features, list(sides)
    audit = {'policy': CORE_CONDITION_POLICY, 'experimental': True,
             'chemistry_report_sha256': digest(chemistry_report),
             'nonstereogenic_tag_cleanup_before_identity_labels': cleaned_stereo,
             'native_ternary_coordinates_used': False, 'calibrated_template_uncertainty': False,
             'added_sides': [], 'rejected_sides': []}
    sizes = [features['protein1'].shape[0], features['protein2'].shape[0]]
    for index, side in enumerate(sides):
        if side is not None:
            continue
        source = record['sides'][str(index + 1)]
        path = Path(source['restored_sdf'])
        if digest(path) != source['restored_sha256']:
            raise ValueError('conserved core binary source hash mismatch')
        binary = Chem.SDMolSupplier(str(path), removeHs=True)[0]
        if binary is None:
            raise ValueError('unreadable binary core source')
        options = [(discover(full, binary), normalized) for discover, normalized in
                   ((conserved_binary_cores, False), (tautomer_conserved_binary_cores, True))]
        options = [(result, normalized) for result, normalized in options if result[0]]
        if not options:
            audit['rejected_sides'].append({'side': index + 1, 'reason': 'no_verified_core'})
            continue
        (cores, chemical_audit), normalized = max(options, key=lambda item: len(item[0][0][0]['source_atom_indices']))
        if len(cores) != 1:
            raise ValueError('multiple distinct core graphs require an explicit hypothesis mixture')
        protein_path = case_dir / f'unbound_protein{index + 1}.pdb'
        geometry = core_geometry_evidence(cores[0], binary, read_protein_coordinates(protein_path))
        if not geometry['orientation_defined'] or geometry['core_contact_atoms'] < 3:
            raise ValueError('core does not define a contacted binary pose')
        proof_full = (canonical_tautomer_with_identity(full)[0] if normalized else
                      rdMolStandardize.Uncharger().uncharge(full))
        condition = SideCondition(provenance='template',
            evidence_id=f'{CORE_CONDITION_POLICY}:{digest(path)}:{digest(protein_path)}',
            confidence=.5, fragment=Chem.MolFromSmiles(geometry['fragment_smiles']),
            atom_maps=tuple(tuple(m) for m in geometry['atom_maps']), mapping_method='exact_stereo',
            pocket_probabilities=geometry['pocket_probabilities'],
            binary_fragment_coords=geometry['binary_fragment_coords'],
            binary_protein_coords=geometry['binary_protein_coords'])
        local = [None, None]
        local[index] = condition
        addition = build_condition_features(proof_full, sizes, local, linker_indices=())
        result = merge_side_features(result, addition, index)
        augmented[index] = BinaryPoseSide(condition.binary_protein_coords,
                                          condition.binary_fragment_coords, condition.atom_maps)
        audit['added_sides'].append({'side': index + 1, 'chemistry': chemical_audit,
            'source_atom_indices': geometry['source_atom_indices_in_smiles_order'],
            'atom_maps': geometry['atom_maps'], 'source_contact_coverage': geometry['source_contact_coverage'],
            'confidence_feature': .5, 'confidence_is_calibrated': False,
            'binary_fragment_sha256': digest(path), 'binary_protein_sha256': digest(protein_path)})
    if not audit['added_sides']:
        raise ValueError('explicit core augmentation did not add any verified condition')
    audit['available_sides'] = [side is not None for side in augmented]
    return result, tuple(augmented), audit
