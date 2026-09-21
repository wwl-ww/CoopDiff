"""Deployment core conditions, built only from validated local binary inputs."""

import json
from pathlib import Path

import torch
from rdkit import Chem

from .binding_core import CORE_POLICY, CONTACT_CORE_POLICY, select_binding_core
from .binary_pose_constraints import BinaryPoseSide
from .condition_features import SideCondition, build_condition_features
from .condition_manifest import load_condition_manifest


def load_core_conditions(path, *, case, molecule, protein_sizes, all_contact_blocks=False):
    # Validate all original chemistry/provenance before narrowing its scope.
    original_features, original_audit = load_condition_manifest(path, case=case, molecule=molecule,
                                               protein_sizes=protein_sizes)
    payload = json.loads(Path(path).read_text())
    conditions, poses, audits = [], [], []
    for raw in payload['sides']:
        if (raw is None or raw.get('binary_fragment_coords') is None
                or raw.get('binary_protein_coords') is None):
            raise ValueError('dual_core deployment requires two validated binary poses')
        maps = raw.get('atom_maps') or [raw['atom_map']]
        coords = torch.tensor(raw['binary_fragment_coords'], dtype=torch.float32)
        protein = torch.tensor(raw['binary_protein_coords'], dtype=torch.float32)
        core = select_binding_core(Chem.MolFromSmiles(raw['fragment_smiles']), maps[0], coords, protein,
                                   all_contact_blocks=all_contact_blocks)
        indices = list(core['local_indices'])
        core_maps = tuple(tuple(mapping[i] for i in indices) for mapping in maps)
        core_coords = coords[indices]
        pocket = (torch.cdist(protein, core_coords).min(1).values <= 6.).float()
        conditions.append(SideCondition(provenance=raw['provenance'], evidence_id=raw['evidence_id'],
            confidence=raw.get('confidence', 1.), fragment=core['fragment'], atom_maps=core_maps,
            mapping_method=raw.get('mapping_method', 'exact_stereo'),
            audit_only=raw.get('audit_only', False), pocket_probabilities=pocket,
            binary_fragment_coords=core_coords, binary_protein_coords=protein))
        poses.append(BinaryPoseSide(protein, core_coords, core_maps))
        audits.append({**core['audit'], 'protected_atom_maps': [list(m) for m in core_maps]})
    build_condition_features(molecule, protein_sizes, tuple(conditions))
    return original_features, tuple(poses), {**original_audit,
                                   'core_policy': CONTACT_CORE_POLICY if all_contact_blocks else CORE_POLICY,
                                   'full_binary_template_retained_as_soft_condition': True,
                                   'binding_cores': audits, 'native_ternary_geometry_read': False}
