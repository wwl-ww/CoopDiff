"""Direct ternary interaction and legal joint-pose prediction.

The HDOCK profile consumes all three current components directly and has no
binary expert modules. Older checkpoint readers retain a separate compatibility
path, but the pure-ternary production profile cannot construct one.
"""

from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tricoopdiff.inputs.graphs import build_binary_graph, build_trifuse_graph, crop_receptor_to_ligand
from tricoopdiff.inputs.static_cache import tensor_view_key
from tricoopdiff.models.expert import _set_time_embeddings, matcha_stage_clock
from tricoopdiff.sdpi_flow import SDPIFlow, horizontal_torsion_spin
from tricoopdiff.diffdock.constants import lig_feature_dims


def protein_local_geometry_features(data, *, required=False):
    """Return node-aligned intrinsic scalars plus measured-data availability.

    Intrinsic monomer geometry stays unchanged under rigid assembly motion.
    Missing structure data is represented by an explicit zero availability bit,
    distinct from a measured geometry vector whose values happen to be zero.
    """
    reference = data['coords'][0]
    values = data.get('local_geometry')
    if values is None:
        if required:
            raise ValueError('protein local_geometry is required by this model contract')
        return reference.new_zeros(len(reference), 33)
    if not torch.is_tensor(values) or values.shape != (1, len(reference), 32):
        raise ValueError('protein local_geometry must be node-aligned [1,N,32]')
    values = values[0].to(reference)
    if not torch.isfinite(values).all():
        raise ValueError('protein local_geometry must be finite')
    available = data.get('local_geometry_available')
    if available is None:
        available = reference.new_ones(len(reference), 1)
    else:
        if not torch.is_tensor(available) or available.shape != (1, len(reference), 1):
            raise ValueError('protein local_geometry_available must be [1,N,1]')
        available = available[0].to(reference)
        if not torch.isfinite(available).all() or not ((available == 0) | (available == 1)).all():
            raise ValueError('protein geometry availability must be binary')
    if required and not available.bool().all():
        raise ValueError('required protein local_geometry contains unavailable residues')
    return torch.cat((values * available, available), -1)


class LigandFeatureEncoder(nn.Module):
    """Embed DiffDock categorical ligand features without treating IDs as floats."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        categorical_dims, scalar_dim = lig_feature_dims
        if scalar_dim:
            raise ValueError("TriFuse expects categorical-only DiffDock ligand features")
        self.categorical_dims = tuple(categorical_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, hidden_dim) for size in self.categorical_dims
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] < len(self.embeddings):
            raise ValueError(
                f"ligand features require {len(self.embeddings)} categorical columns"
            )
        encoded = features.new_zeros(features.shape[0], self.norm.normalized_shape[0])
        for column, (embedding, size) in enumerate(zip(
            self.embeddings, self.categorical_dims,
        )):
            indices = features[:, column].long().clamp(0, size - 1)
            encoded = encoded + embedding(indices)
        return self.norm(encoded / math.sqrt(len(self.embeddings)))


BRANCHES = ('e3_ligand', 'poi_ligand')


def bounded_vector(value):
    """Isotropic compression, retaining direction when the raw norm overflows."""
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    finite = torch.isfinite(norm)
    ordinary = value / (1 + torch.where(finite, norm, torch.zeros_like(norm)))
    scale = value.abs().amax(dim=-1, keepdim=True).clamp_min(1.)
    scaled = value / scale
    stable = scaled / (scale.reciprocal()
                       + torch.linalg.vector_norm(scaled, dim=-1, keepdim=True))
    # Ordinary inputs retain the historical float arithmetic exactly. Scaling
    # is only needed when squaring finite components overflowed the raw norm.
    return torch.where(finite, ordinary, stable)


def two_sided_contact_direction_field(logits, coords, types):
    """Turn target-pocket ranks into equivariant ligand and receptor directions."""
    ligand = types == 2
    field = coords.new_zeros(len(coords), 2, 3)
    for side, branch in enumerate(BRANCHES):
        receptor = types == side
        scores = logits[branch]
        delta = coords[receptor][None] - coords[ligand][:, None]
        if scores.shape != delta.shape[:2]:
            raise ValueError(f'{branch} contact field shape disagrees with ternary nodes')
        # A far-away ligand still receives a direction even when all absolute
        # contact logits are negative: only relative pocket preference matters.
        field[ligand, side] = bounded_vector(
            (scores.softmax(-1)[..., None] * delta).sum(1) / 10.)
        field[receptor, side] = bounded_vector(
            -(scores.softmax(0)[..., None] * delta).sum(0) / 10.)
    return field


def two_sided_anchor_relation_field(logits, roles, coords, types):
    """Equivariant AA/AN/NN messages for both protein-ligand interfaces."""
    ligand = types == 2
    field = coords.new_zeros(len(coords), 6, 3)
    for side, branch in enumerate(BRANCHES):
        receptor = types == side
        scores = logits[branch]
        ligand_role = roles[branch]['ligand'].sigmoid()[:, None]
        receptor_role = roles[branch]['protein'].sigmoid()[None, :]
        delta = coords[receptor][None] - coords[ligand][:, None]
        if scores.shape != delta.shape[:2] or ligand_role.shape[0] != len(delta) or \
                receptor_role.shape[1] != delta.shape[1]:
            raise ValueError(f'{branch} anchor roles disagree with ternary nodes')
        relation = torch.stack((
            ligand_role * receptor_role,
            ligand_role * (1-receptor_role) + (1-ligand_role) * receptor_role,
            (1-ligand_role) * (1-receptor_role),
        ), -1)
        direction = bounded_vector(delta / 10.)
        ligand_weight = scores.softmax(-1)
        receptor_weight = scores.softmax(0)
        for kind in range(3):
            pair_role = relation[..., kind]
            field[ligand, side*3+kind] = (
                ligand_weight[..., None] * pair_role[..., None] * direction).sum(1)
            field[receptor, side*3+kind] = -(
                receptor_weight[..., None] * pair_role[..., None] * direction).sum(0)
    return field


def ligand_rotor_quads(edges, rotor_mask):
    """Choose one signed dihedral per directed legal rotor, retaining rotor indices."""
    adjacency = {}
    for a, b in edges.detach().cpu().T.tolist():
        adjacency.setdefault(a, set()).add(b)
    quads, indices = [], []
    for index, (a, b) in enumerate(edges[:, rotor_mask.bool()].detach().cpu().T.tolist()):
        left, right = sorted(adjacency[a]-{b}), sorted(adjacency[b]-{a})
        if left and right:
            quads.append((left[0], a, b, right[0]))
            indices.append(index)
    return (torch.tensor(quads, device=edges.device, dtype=torch.long).reshape(-1, 4),
            torch.tensor(indices, device=edges.device, dtype=torch.long))


def torsion_phase_pairs(ligand, quads):
    """SE(3)-invariant signed dihedral phases in directed rotating-bond order."""
    left, first, second, right = ligand[:, quads].unbind(-2)
    axis = torch.nn.functional.normalize(second-first, dim=-1)
    u, v = left-first, right-second
    u, v = u-(u*axis).sum(-1, keepdim=True)*axis, v-(v*axis).sum(-1, keepdim=True)*axis
    valid = (u.norm(dim=-1) > 1e-4) & (v.norm(dim=-1) > 1e-4)
    u = torch.nn.functional.normalize(u, dim=-1, eps=1e-6)
    v = torch.nn.functional.normalize(v, dim=-1, eps=1e-6)
    return torch.stack(((u*v).sum(-1),
        (torch.linalg.cross(u, v, dim=-1)*axis).sum(-1)), -1).mul(valid[..., None])


def ligand_frame_triplet(edges, ligand):
    """Choose a noncollinear bonded triplet, invariant to rigid and torsion motion."""
    neighbors = {}
    for a, b in edges.detach().cpu().T.tolist():
        neighbors.setdefault(a, set()).add(b)
    xyz = ligand[0]
    for center in sorted(neighbors):
        adjacent = sorted(neighbors[center])
        for first in range(len(adjacent)):
            for second in range(first + 1, len(adjacent)):
                a, b = adjacent[first], adjacent[second]
                u, v = xyz[a] - xyz[center], xyz[b] - xyz[center]
                if torch.linalg.vector_norm(torch.linalg.cross(u, v)) > .05:
                    return (a, center, b)
    raise ValueError('ligand has no stable bonded three-atom frame')


def _orthogonal_frame(first, second):
    x = torch.nn.functional.normalize(first, dim=-1)
    y = second - (second*x).sum(-1, keepdim=True)*x
    if bool((torch.linalg.vector_norm(y, dim=-1) < 1e-6).any()):
        raise ValueError('degenerate three-atom frame')
    y = torch.nn.functional.normalize(y, dim=-1)
    return torch.stack((x, y, torch.linalg.cross(x, y, dim=-1)), -1)


def e3_reference_frame(e3):
    """Equivariant receptor frame from geometry-only, rotation-invariant choices."""
    if e3.ndim != 3 or e3.shape[0] != 1 or e3.shape[1] < 3:
        raise ValueError('E3 reference frame requires one protein with >=3 residues')
    center = e3.mean(1)
    offsets = e3 - center[:, None]
    first = offsets[0, offsets[0].square().sum(-1).argmax()]
    axis = torch.nn.functional.normalize(first, dim=-1)
    transverse = offsets[0] - (offsets[0]*axis).sum(-1, keepdim=True)*axis
    second = transverse[transverse.square().sum(-1).argmax()]
    return center, _orthogonal_frame(first[None], second[None])


def ligand_body_frame(ligand, edges):
    a, center, b = ligand_frame_triplet(edges, ligand)
    xyz = ligand[:, (a, center, b)]
    return xyz[:, 1], _orthogonal_frame(xyz[:, 0]-xyz[:, 1], xyz[:, 2]-xyz[:, 1])


def ligand_target_frame_label(e3, ligand, edges):
    """A pose target constant across legal start perturbations, in the E3 frame."""
    origin, basis = e3_reference_frame(e3)
    ligand_origin, ligand_basis = ligand_body_frame(ligand, edges)
    local_origin = ((ligand_origin-origin)[:, None] @ basis).squeeze(1) / 10.
    local_basis = basis.transpose(-1, -2) @ ligand_basis
    return torch.cat((local_origin, local_basis[..., 0], local_basis[..., 1]), -1)


def place_ligand_in_target_frame(ligand, e3, edges, predicted_pose):
    """Apply the learned E3-relative target frame as an exact ligand rigid motion."""
    if predicted_pose.shape != (1, 9):
        raise ValueError('target ligand frame must contain nine local coordinates')
    origin, basis = e3_reference_frame(e3)
    target_local = _orthogonal_frame(predicted_pose[:, 3:6], predicted_pose[:, 6:9])
    target_basis = basis @ target_local
    target_origin = origin + ((predicted_pose[:, :3] * 10.)[:, None]
                              @ basis.transpose(-1, -2)).squeeze(1)
    current_origin, current_basis = ligand_body_frame(ligand, edges)
    rotation = target_basis @ current_basis.transpose(-1, -2)
    return (ligand-current_origin[:, None]) @ rotation.transpose(-1, -2) + target_origin[:, None]


ARCHITECTURE = 'residual_evidence_v3'
DIRECT_ARCHITECTURE = 'ternary_topological_joint_pose_v14'
PURE_TERNARY_ARCHITECTURE = 'ternary_topological_joint_pose_no_experts_v2'
PREVIOUS_DIRECT_ARCHITECTURE = 'ternary_nodefield_independent_v4'
ASSEMBLY_CHECKPOINT_VERSION = 64
EXPERT_CROP_POLICY = 'full_receptor_cross_taper_v1'
LEGACY_RANK_FEATURE_STATE = 'final_pose_t0_dt0_v1'
DIRECT_RANK_FEATURE_STATE = 'final_pose_evidence_t0_dt0_v2'
INTERFACE_OBJECTIVE_CHECKPOINT_VERSION = 22
OBJECTIVE_CONTRACT_CHECKPOINT_VERSION = 21
ENDPOINT_PROTOCOL_CHECKPOINT_VERSION = 20
LEGACY_CONSENSUS_CHECKPOINT_VERSION = 18
PREVIOUS_CONSENSUS_CHECKPOINT_VERSION = 19
SOURCE_NAMES = ('sdpi', 'bitseed')
CANDIDATE_SOURCE_NAMES = (*SOURCE_NAMES, 'hdock_matcha')
PURE_CANDIDATE_SOURCE_NAMES = ('sdpi', 'hdock_etkdg')
CANDIDATE_FRAMEWORK_VERSION = 'hdock_etkdg_ternary_v1'
QUALITY_COMPONENTS = ('protein', 'ligand', 'e3_ligand', 'poi_ligand', 'clash')
CANDIDATE_GEOMETRY_CONTRACT = 'smooth_allatom_cross_proximity_v1'
CANDIDATE_QUALITY_CONTRACT = 'static_identity_current_contact_quality_v4'
CANDIDATE_CONDITION_CONTRACT = 'candidate_reference_fullatom_source_clock_film_v2'
CANDIDATE_REFERENCE_CONTRACT = 'immutable_pose_displacement_transport_v1'
PURE_CANDIDATE_REFERENCE_CONTRACT = 'immutable_three_center_directions_v2'
TASK_NAMES = ('assemble', 'refine')
CANDIDATE_TASK_CONTRACT = 'explicit_task_cross_source_v1'
SOURCE_CONTRACT_VERSION = 'candidate_source_conditioning_v1'
PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v4'
VECTOR_EXPERT_MODEL_PROFILE = 'ternary_endpoint_v14'
PURE_TERNARY_MODEL_PROFILE = 'ternary_only_hdock_v2'
FLOW_MOTION_CONTRACT = 'sdpi_flow_velocity_v1'
DRIFTING_MOTION_CONTRACT = 'conditional_drifting_candidate_step_v5'
CONTACT_GUIDANCE_CONTRACT = 'two_sided_atom_residue_contact_field_v1'
ANCHOR_RELATION_CONTRACT = 'two_sided_anchor_role_relations_v2'
TORSION_PHASE_CONTRACT = 'predicted_target_rotor_phase_local_v3'
TARGET_LIGAND_FRAME_CONTRACT = 'anchor_driven_joint_pose_v3'
BOUNDED_REPAIR_MOTION_CONTRACT = 'bounded_iterative_correction_v1'
QUALITY_GATED_REPAIR_MOTION_CONTRACT = 'quality_gated_bounded_repair_v1'
QUALITY_GATE_ERROR_SCALE = 0.5
MATCHA_CONDITIONING_CONTRACT = 'stage_specific_chemical_interface_v5'
INFORMATION_CONTRACT = 'conditional_node_evidence_without_rigid_advice_v2'
RIGID_DECODER_CONTRACT = 'independent_horizontal_rotor_coupling_v4'
LEGACY_TORSION_REPRESENTATION = 'directed_rotor_v1'
TORSION_REPRESENTATION_CONTRACT = 'horizontal_equivalent_rotor_v2'
PROTEIN_LOCAL_GEOMETRY_CONTRACT = 'intrinsic32_masked_v1'
PROTEIN_LOCAL_GEOMETRY_CONTRACTS = ('none', PROTEIN_LOCAL_GEOMETRY_CONTRACT, 'intrinsic32_required_v1')
PROTAC_ANCHOR_CONTRACT = 'dual_bound_anchor_node_context_v1'


def production_model_config(node_dim, input_dim, source_conditioning=False, *,
                            model_profile=VECTOR_EXPERT_MODEL_PROFILE,
                            expert_scalar_dim=58, expert_vector_dim=20,
                            expert_backend='diffdock', conditioning_contract=MATCHA_CONDITIONING_CONTRACT,
                            information_contract=INFORMATION_CONTRACT,
                            rigid_decoder_contract=RIGID_DECODER_CONTRACT,
                            torsion_representation_contract=TORSION_REPRESENTATION_CONTRACT,
                            protein_local_geometry_contract=PROTEIN_LOCAL_GEOMETRY_CONTRACT,
                            protac_anchor_contract=None,
                            candidate_driven=False, candidate_expert_frame_policy='backbone_rigid_transport_v1',
                            motion_contract=FLOW_MOTION_CONTRACT,
                            contact_guidance_contract='auto', torsion_phase_contract=None,
                            target_ligand_frame_contract=None):
    """Construct only the implemented controller; do not synthesize retired recipes."""
    if model_profile not in (VECTOR_EXPERT_MODEL_PROFILE, PURE_TERNARY_MODEL_PROFILE):
        raise ValueError(f'retired model profile: {model_profile}')
    if information_contract != INFORMATION_CONTRACT:
        raise ValueError('retired evidence contract: use conditional node evidence without rigid advice')
    if rigid_decoder_contract != RIGID_DECODER_CONTRACT:
        raise ValueError('retired action contract: use physical velocities without inverse-time gain')
    if motion_contract not in (FLOW_MOTION_CONTRACT, DRIFTING_MOTION_CONTRACT, BOUNDED_REPAIR_MOTION_CONTRACT,
                               QUALITY_GATED_REPAIR_MOTION_CONTRACT):
        raise ValueError('unknown motion contract')
    if motion_contract == DRIFTING_MOTION_CONTRACT and (
            model_profile != PURE_TERNARY_MODEL_PROFILE or not candidate_driven):
        raise ValueError('drifting requires the pure ternary candidate generator')
    eligible_contact_guidance = (motion_contract == DRIFTING_MOTION_CONTRACT
        and model_profile == PURE_TERNARY_MODEL_PROFILE and candidate_driven)
    if contact_guidance_contract == 'auto':
        contact_guidance_contract = ANCHOR_RELATION_CONTRACT if eligible_contact_guidance else None
    if contact_guidance_contract not in (None, CONTACT_GUIDANCE_CONTRACT, ANCHOR_RELATION_CONTRACT):
        raise ValueError('unknown two-sided contact guidance contract')
    if contact_guidance_contract is not None and not eligible_contact_guidance:
        raise ValueError('two-sided contact guidance requires pure ternary drifting')
    if torsion_phase_contract not in (None, TORSION_PHASE_CONTRACT):
        raise ValueError('unknown torsion phase encoding contract')
    if torsion_phase_contract is not None and not eligible_contact_guidance:
        raise ValueError('explicit torsion phase encoding requires pure ternary drifting')
    if target_ligand_frame_contract not in (None, TARGET_LIGAND_FRAME_CONTRACT):
        raise ValueError('unknown target ligand frame contract')
    if target_ligand_frame_contract and torsion_phase_contract != TORSION_PHASE_CONTRACT:
        raise ValueError('target ligand frame requires the target rotor-phase controller')
    if target_ligand_frame_contract and contact_guidance_contract != ANCHOR_RELATION_CONTRACT:
        raise ValueError('target ligand frame requires two-sided anchor roles')
    config = dict(node_dim=node_dim, input_dim=input_dim, information_contract=information_contract,
                rigid_decoder_contract=rigid_decoder_contract,
                source_conditioning=bool(source_conditioning),
                direct_ternary_motion=True,
                model_profile=model_profile,
                motion_contract=motion_contract)
    if motion_contract == FLOW_MOTION_CONTRACT:
        config.pop('motion_contract')
    if contact_guidance_contract is not None:
        config['contact_guidance_contract'] = contact_guidance_contract
    if torsion_phase_contract is not None:
        config['torsion_phase_contract'] = torsion_phase_contract
    if target_ligand_frame_contract is not None:
        config['target_ligand_frame_contract'] = target_ligand_frame_contract
    if model_profile == PURE_TERNARY_MODEL_PROFILE:
        if (candidate_driven and not source_conditioning) or protac_anchor_contract is not None:
            raise ValueError('pure ternary candidate model requires source conditioning and no external anchors')
        config['expert_free'] = True
        config['model_profile'] = PURE_TERNARY_MODEL_PROFILE
        expert_scalar_dim = 0
        expert_vector_dim = 0
        config.pop('expert_backend', None)
        config.pop('conditioning_contract', None)
    if candidate_driven:
        if (model_profile != PURE_TERNARY_MODEL_PROFILE
                and (expert_backend != 'matcha' or not source_conditioning)):
            raise ValueError('legacy candidate framework requires Matcha and source conditioning')
        if model_profile == PURE_TERNARY_MODEL_PROFILE and not source_conditioning:
            raise ValueError('HDOCK candidate framework requires explicit source conditioning')
        config['candidate_driven'] = True
        config['candidate_geometry_contract'] = CANDIDATE_GEOMETRY_CONTRACT
        if motion_contract != DRIFTING_MOTION_CONTRACT:
            config['candidate_quality_contract'] = CANDIDATE_QUALITY_CONTRACT
        config['candidate_condition_contract'] = CANDIDATE_CONDITION_CONTRACT
        config['candidate_reference_contract'] = (PURE_CANDIDATE_REFERENCE_CONTRACT
            if model_profile == PURE_TERNARY_MODEL_PROFILE else CANDIDATE_REFERENCE_CONTRACT)
        config['candidate_task_contract'] = CANDIDATE_TASK_CONTRACT
        if candidate_expert_frame_policy not in ('legacy_argmax_v1', 'backbone_rigid_transport_v1'):
            raise ValueError('unknown candidate expert frame policy')
        if model_profile != PURE_TERNARY_MODEL_PROFILE:
            config['candidate_expert_frame_policy'] = candidate_expert_frame_policy
    for name, value in (('expert_scalar_dim', expert_scalar_dim), ('expert_vector_dim', expert_vector_dim)):
        if model_profile == PURE_TERNARY_MODEL_PROFILE:
            config[name] = 0
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
        config[name] = value
    if torsion_representation_contract not in (LEGACY_TORSION_REPRESENTATION, TORSION_REPRESENTATION_CONTRACT):
        raise ValueError('unknown torsion representation contract')
    if torsion_representation_contract != LEGACY_TORSION_REPRESENTATION:
        config['torsion_representation_contract'] = torsion_representation_contract
    if protein_local_geometry_contract not in PROTEIN_LOCAL_GEOMETRY_CONTRACTS:
        raise ValueError('unknown protein local geometry contract')
    if protein_local_geometry_contract != 'none':
        config['protein_local_geometry_contract'] = protein_local_geometry_contract
    if protac_anchor_contract is not None:
        if protac_anchor_contract != PROTAC_ANCHOR_CONTRACT:
            raise ValueError('unknown PROTAC anchor conditioning contract')
        config['protac_anchor_contract'] = protac_anchor_contract
    if model_profile == PURE_TERNARY_MODEL_PROFILE:
        return config
    if expert_backend not in ('diffdock', 'matcha'):
        raise ValueError('unknown binary expert backend')
    if expert_backend == 'matcha':
        if conditioning_contract != MATCHA_CONDITIONING_CONTRACT:
            raise ValueError('retired Matcha conditioning: distance must enter nodewise values')
        config['expert_backend'] = expert_backend
        config['conditioning_contract'] = conditioning_contract
    return config


def candidate_source_tensor(source, reference, source_names=SOURCE_NAMES):
    """Normalize one immutable trajectory source to a ``[1]`` class tensor."""
    if isinstance(source, str):
        if source not in source_names:
            raise ValueError(f'unknown candidate source: {source!r}')
        source = source_names.index(source)
    value = torch.as_tensor(source, device=reference.device).reshape(-1)
    if (value.shape != (1,) or value.dtype == torch.bool or value.is_complex()
            or not bool(((value >= 0) & (value < len(source_names)) & (value == value.long())).all())):
        raise ValueError('candidate source must be one declared source class')
    return value.long()


def task_source_index(source, task_mode, reference, source_names=CANDIDATE_SOURCE_NAMES):
    """Keep initialization source and requested task independent and explicit."""
    if task_mode is None:
        raise ValueError('candidate model requires explicit task_mode: assemble=0 or refine=1')
    return (candidate_source_tensor(source, reference, source_names)
            + len(source_names)*candidate_source_tensor(task_mode, reference, TASK_NAMES))


def migrate_legacy_candidate_task_embedding(payload):
    """Explicitly expand a source-only candidate checkpoint to source×task rows.

    Legacy candidate checkpoints had one learned embedding per source. Copying
    each source row into both task blocks preserves the old function exactly;
    it does not claim the checkpoint learned distinct assemble/refine behavior.
    """
    config = payload.get('model_config')
    state = payload.get('model_state')
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError('checkpoint must contain model_config and model_state mappings')
    if (not config.get('candidate_driven') or not config.get('source_conditioning')
            or config.get('expert_backend') != 'matcha'):
        raise ValueError('only source-conditioned Matcha candidate checkpoints can be migrated')
    if config.get('candidate_task_contract') not in (None,):
        raise ValueError('checkpoint is not a legacy source-only task-conditioning checkpoint')
    key = 'controller.source.weight'
    source_weight = state.get(key)
    if (not torch.is_tensor(source_weight) or source_weight.ndim != 2
            or source_weight.shape[0] != len(CANDIDATE_SOURCE_NAMES)):
        raise ValueError('legacy source embedding must have exactly one row per candidate source')

    migrated_state = dict(state)
    migrated_state[key] = torch.cat((source_weight, source_weight), dim=0).contiguous()
    migrated_config = dict(config)
    migrated_config['candidate_task_contract'] = CANDIDATE_TASK_CONTRACT
    # Strict construction/load proves this is only a shape-preserving semantic
    # expansion; any unrelated architecture drift must be handled separately.
    migrated_model = TernaryAssembly(**migrated_config)
    migrated_model.load_state_dict(migrated_state, strict=True)

    migrated = dict(payload)
    migrated['model_config'] = migrated_config
    migrated['model_state'] = migrated_state
    migrated['checkpoint_migration'] = {
        'contract': 'legacy_source_embedding_to_source_task_v1',
        'source_names': list(CANDIDATE_SOURCE_NAMES),
        'task_names': list(TASK_NAMES),
        'row_map': {name: {'assemble': index,
                           'refine': index + len(CANDIDATE_SOURCE_NAMES)}
                    for index, name in enumerate(CANDIDATE_SOURCE_NAMES)},
        'source_weight_shape': list(source_weight.shape),
        'migrated_weight_shape': list(migrated_state[key].shape),
        'semantics': 'duplicate_each_source_row_into_both_task_blocks; no task-specific behavior was learned',
    }
    return migrated


def candidate_fullatom_evidence(coords, types, e3, poi, ligand, body):
    """Node-level cross-component heavy-atom geometry, without target access.

    Sidechain atoms remain rigid with each monomer. Ligand nodes therefore
    observe actual sidechain obstacles, not only C-alpha proxy distances.
    Scalars are invariant; smooth local displacement vectors are equivariant.
    """
    from rdkit import Chem

    if any(key not in part for part in (e3, poi) for key in ('steric_atoms', 'steric_radii')):
        raise ValueError('candidate model requires full-atom monomer templates')
    # POI templates are attached to the input monomer, not its current coords.
    # The immutable template center is supplied explicitly by the data loader.
    if 'steric_ca_center' not in poi:
        raise ValueError('candidate POI requires its immutable template center')
    p2 = (poi['steric_atoms'][0]-poi['steric_ca_center']) @ body.T + poi['coords'][0].mean(0)
    mol = ligand['mol']
    mol = Chem.Mol(mol) if isinstance(mol, bytes) else mol
    table = Chem.GetPeriodicTable()
    lr = coords.new_tensor([table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()])
    query_radii = coords.new_full((len(coords),), 1.7)
    query_radii[types == 2] = lr
    components = ((e3['steric_atoms'][0], e3['steric_radii']),
                  (p2, poi['steric_radii']), (ligand['coords'][0], lr))
    scalars, vectors = [], []
    for component, (atoms, radii) in enumerate(components):
        chunks, directions = [], []
        for start in range(0, len(coords), 64):
            delta = atoms[None]-coords[start:start+64, None]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            weights = (-distance.square()/4.).softmax(-1)
            local_distance = (weights*distance).sum(-1)
            limits = .75*(query_radii[start:start+64, None]+radii[None])
            overlap = torch.relu(limits-distance)
            chunks.append(torch.stack((torch.log1p(local_distance/10.),
                torch.log1p(torch.exp(-distance.square()/16.).sum(-1)),
                torch.log1p(overlap.square().sum(-1))), -1))
            directions.append(bounded_vector((weights[..., None]*delta).sum(1)/10.))
        cross = (types != component).to(coords.dtype)[:, None]
        scalars.append(torch.cat(chunks)*cross)
        vectors.append(torch.cat(directions)*cross)
    return torch.cat(scalars, -1), torch.stack(vectors, 1)


def candidate_reference_evidence(coords, types, e3, poi, ligand, *,
                                 contract=CANDIDATE_REFERENCE_CONTRACT):
    """Native-free pose reference held fixed throughout one correction rollout."""
    reference = poi.get('candidate_reference')
    if reference is None:
        raise ValueError('candidate correction requires its original input pose')
    initial_poi, initial_ligand, initial_body = [reference[key].to(coords)
        for key in ('poi', 'ligand', 'body')]
    if (initial_poi.shape != poi['coords'].shape or initial_ligand.shape != ligand['coords'].shape
            or initial_body.shape != (1, 3, 3)
            or any(value.requires_grad for value in (initial_poi, initial_ligand, initial_body))):
        raise ValueError('candidate reference must be immutable and node-aligned')
    initial = torch.cat((e3['coords'][0], initial_poi[0], initial_ligand[0]))
    centers = torch.stack([coords[types == i].mean(0) for i in range(3)])
    initial_centers = torch.stack([initial[types == i].mean(0) for i in range(3)])
    displacement = (initial-coords)/10.
    distance = torch.cdist(coords, centers, compute_mode='donot_use_mm_for_euclid_dist')/10.
    initial_distance = torch.cdist(initial, initial_centers,
        compute_mode='donot_use_mm_for_euclid_dist')/10.
    scalars = torch.cat((torch.log1p(displacement.square().sum(-1, keepdim=True)),
                         torch.log1p(distance)-torch.log1p(initial_distance)), -1)
    if contract == PURE_CANDIDATE_REFERENCE_CONTRACT:
        # Retain three distinct, immutable directions for every node. The old
        # pure model copied one component offset into all three slots, leaving
        # E3 nodes with three zero slots and losing reference-frame geometry.
        directions = (initial_centers[None] - initial[:, None]) / 10.
    elif contract == CANDIDATE_REFERENCE_CONTRACT:
        component_offsets = torch.stack([initial_centers[i]-initial_centers[0]
                                         for i in range(3)])[types]
        directions = component_offsets[:, None].expand(-1, 3, -1)
    else:
        raise ValueError('unknown candidate reference evidence contract')
    vectors = torch.cat((bounded_vector(displacement)[:, None],
                         bounded_vector(directions)), 1)
    return scalars, vectors


_CHEMICAL_PATH_CACHE = OrderedDict()
_CHEMICAL_PATH_CACHE_LIMIT = 32
_CHEMICAL_SIGNATURE_CACHE = OrderedDict()


class ComponentStructureEncoder(nn.Module):
    """Node memory independent of relative poses and ligand folding.

    Proteins read internal distances; ligand reads chemical shortest paths,
    edge attributes and bond lengths, never nonbonded Cartesian distances.
    """

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.geometry = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width + 1))
        self.chemistry = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width + 1))
        self.norm = nn.LayerNorm(width)

    def _protein_messages(self, query, keys, values, query_xyz, xyz):
        distance = torch.linalg.vector_norm(query_xyz[:, None]-xyz[None], dim=-1)/10.
        relation = self.geometry(torch.log1p(distance)[..., None])
        logits = query @ keys.T/math.sqrt(self.width)+relation[..., -1]
        return (logits.softmax(-1)[..., None]*(values[None]+relation[..., :-1])).sum(1)

    def _ligand_messages(self, query, keys, values, chemistry, connected):
        relation = self.chemistry(chemistry)
        logits = (query @ keys.T/math.sqrt(self.width)+relation[..., -1]).masked_fill(~connected, -torch.inf)
        return (logits.softmax(-1)[..., None]*(values[None]+relation[..., :-1])).sum(1)

    @staticmethod
    def chemical_relations(coords, edges, attributes):
        """Exact unweighted graph distance plus direct-edge chemistry."""
        count = len(coords)
        # Content signatures detect in-place graph edits and atom reindexing.
        # Coordinates are deliberately excluded: only topology is cached.
        key = None
        if count <= 512:
            def signature(tensor):
                raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device),
                        hashlib.sha256(raw).digest())
            views = (tensor_view_key(edges), tensor_view_key(attributes))
            view_key = views if all(value is not None for value in views) else None
            record = _CHEMICAL_SIGNATURE_CACHE.get(view_key) if view_key is not None else None
            if record is None:
                signatures = (signature(edges), signature(attributes))
                if view_key is not None:
                    _CHEMICAL_SIGNATURE_CACHE[view_key] = (signatures, edges, attributes)
                    while len(_CHEMICAL_SIGNATURE_CACHE) > _CHEMICAL_PATH_CACHE_LIMIT:
                        _CHEMICAL_SIGNATURE_CACHE.popitem(last=False)
            else:
                signatures = record[0]
                _CHEMICAL_SIGNATURE_CACHE.move_to_end(view_key)
            key = (count, str(coords.device), coords.dtype, *signatures)
        cached = _CHEMICAL_PATH_CACHE.get(key) if key is not None else None
        if cached is None:
            paths = coords.new_full((count, count), float('inf'))
            paths.diagonal().zero_()
            paths[edges[0], edges[1]] = 1.
            # Connectivity is undirected; chemistry retains directed slots.
            paths[edges[1], edges[0]] = 1.
            for middle in range(count):
                paths = torch.minimum(paths, paths[:, middle:middle + 1] + paths[middle:middle + 1])
            connected = torch.isfinite(paths)
            graph_distance = torch.log1p(paths.masked_fill(~connected, 0.))
            if key is not None:
                _CHEMICAL_PATH_CACHE[key] = (graph_distance, connected)
                while len(_CHEMICAL_PATH_CACHE) > _CHEMICAL_PATH_CACHE_LIMIT:
                    _CHEMICAL_PATH_CACHE.popitem(last=False)
        else:
            graph_distance, connected = cached
            _CHEMICAL_PATH_CACHE.move_to_end(key)
        # Never expose mutable cached storage to a caller.
        connected = connected.clone()
        features = coords.new_zeros(count, count, 6)
        features[..., 0] = graph_distance
        features[edges[0], edges[1], 1:5] = attributes
        lengths = torch.linalg.vector_norm(coords[edges[0]] - coords[edges[1]], dim=-1) / 10.
        features[edges[0], edges[1], 5] = torch.log1p(lengths)
        return features, connected

    def forward(self, identity, coords, types, edges, attributes, *,
                component_indices=None, ligand_edges=None):
        encoded = torch.zeros_like(identity)
        for component in range(3):
            ids = (component_indices[component] if component_indices is not None
                   else (types == component).nonzero(as_tuple=False).flatten())
            if not len(ids):
                continue
            h, xyz = identity[ids], coords[ids]
            keys, values = self.key(h), self.value(h)
            if component == 2:
                if ligand_edges is None:
                    local = torch.empty(len(identity), device=ids.device, dtype=torch.long)
                    local[ids] = torch.arange(len(ids), device=ids.device)
                    ligand_edges = local[edges]
                chemistry, connected = self.chemical_relations(xyz, ligand_edges, attributes)
            pieces = []
            chunk = 16 if len(ids) > 512 else 64
            for start in range(0, len(ids), chunk):
                stop = min(start + chunk, len(ids))
                query = self.query(h[start:stop])
                if component == 2:
                    fn = self._ligand_messages
                    args = (query, keys, values, chemistry[start:stop], connected[start:stop])
                else:
                    fn = self._protein_messages
                    args = (query, keys, values, xyz[start:stop], xyz)
                # Save query/key inputs, not every quadratic relation MLP's
                # activations. Bound recomputation independently of the outer
                # rollout checkpoint; do not capture mutable loop variables.
                if self.training and torch.is_grad_enabled():
                    messages = checkpoint(fn, *args, use_reentrant=False)
                else:
                    messages = fn(*args)
                pieces.append(messages)
            encoded[ids] = self.norm(h + torch.cat(pieces))
        return encoded


class EquivariantInteraction(nn.Module):
    """Dense, source-balanced SO(3) attention without a pooled control token."""

    def __init__(self, width, channels):
        super().__init__()
        self.width = width
        self.heads = 4 if width % 4 == 0 and channels % 4 == 0 else 1
        # Runtime batching only: no parameters or checkpoint metadata change.
        self.query_chunk_size = 64
        self._compiled_messages = None
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.geometry = nn.Sequential(nn.Linear(5 + 2 * channels, width), nn.SiLU(),
                                      nn.Linear(width, width + self.heads))
        self.direction = nn.Linear(width, channels)
        self.vector_mix = nn.Linear(3 * channels, channels, bias=False)
        self.scalar_mix = nn.Sequential(nn.Linear(4 * width + channels + 3, width), nn.SiLU(),
                                        nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def _messages(self, q, query_vectors, query_coords, keys, values, vectors,
                  coords, component_indices, bonds):
        # Retain the direct-call API used by historical diagnostics. Production
        # passes precomputed indices, so nonzero stays outside the compiled core.
        if torch.is_tensor(component_indices):
            component_indices = tuple((component_indices == i).nonzero(as_tuple=False).flatten() for i in range(3))
        delta = (coords[None] - query_coords[:, None]) / 10.
        distance = delta.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        direction = delta / (1. + distance)
        # Direction-dependent invariants retain orientation, not just distances.
        left = (query_vectors[:, None] * direction[:, :, None]).sum(-1)
        right = (vectors[None] * direction[:, :, None]).sum(-1)
        geometry = torch.cat((torch.log1p(distance), left, right, bonds), -1)
        query = q.reshape(len(q), self.heads, -1)
        keys = keys.reshape(len(keys), self.heads, -1)
        logits = torch.einsum('ihd,jhd->ijh', query, keys) / math.sqrt(self.width // self.heads)
        relation = self.geometry(geometry)
        logits = logits + relation[..., -self.heads:]
        pair_values = relation[..., :-self.heads].reshape(len(q), len(keys), self.heads, -1)
        scalar_messages, vector_messages, densities = [], [], []
        for component in range(3):
            selected = component_indices[component]
            weights = logits[:, selected].softmax(1)
            messages = (values[selected].reshape(1, -1, self.heads, self.width // self.heads)
                        + pair_values[:, selected])
            scalar_messages.append(torch.einsum('ijh,ijhd->ihd', weights, messages).flatten(1))
            direct = torch.einsum('ijh,jhvc->ihvc', weights,
                vectors[selected].reshape(-1, self.heads, vectors.shape[1] // self.heads, 3))
            # Three-operand einsum invokes opt_einsum, whose path planner
            # converts symbolic dimensions to int and specializes every case.
            weighted_direction = weights[..., None] * direction[:, selected, None, :]
            radial = torch.einsum('ijhc,jhv->ihvc', weighted_direction,
                self.direction(values[selected]).reshape(-1, self.heads, vectors.shape[1] // self.heads))
            handed = torch.cross(vectors[None, selected],
                                 direction[:, selected, None].expand(-1, -1, vectors.shape[1], -1), dim=-1)
            handed = handed.reshape(len(q), -1, self.heads, vectors.shape[1] // self.heads, 3)
            transport = direct + radial + torch.einsum('ijh,ijhvc->ihvc', weights, handed)
            vector_messages.append(transport.flatten(1, 2))
            densities.append(torch.log1p(torch.exp(-distance[:, selected, 0].square()).sum(1, keepdim=True)))
        return torch.cat((*scalar_messages, *densities), -1), torch.cat(vector_messages, 1)

    def set_compile_messages(self, enabled):
        """Compile only the tensor attention core with real Inductor fusion.

        The wrapper stays outside nn.Module registration/state_dict. Failures
        propagate; enabling this never silently falls back to eager execution.
        """
        if enabled:
            # AC(compile(fn)) must select the same graph during recomputation.
            # Dynamo LRU hits can reorder overlapping valid graphs between
            # forward and backward (pytorch/pytorch#166926). This switch is
            # process-wide; do not restore LRU when disabling one layer while
            # another layer may still have outstanding checkpointed forwards.
            torch._C._dynamo.eval_frame._set_lru_cache(False)
        # Independent blocks share this Python code but have distinct parameter,
        # alias and checkpoint/autograd guards. Keep their caches independent.
        # Full/chunk queries, equal/unequal components, and grad/alias contexts
        # need several structural graphs even with genuinely symbolic lengths.
        # This is a bounded per-region budget, not a global fallback policy:
        # fullgraph still raises if the region exceeds it.
        self._compiled_messages = (torch.compile(self._messages, backend='inductor',
            fullgraph=True, dynamic=True, recompile_limit=32,
            isolate_recompiles=True) if enabled else None)

    def _run_compiled_messages(self, *args):
        # Saved/autograd tensors can lose their view base during checkpoint
        # unpacking. AOT may then select a different saved-tensor layout even
        # for equal values/shapes. Give geometric inputs the same non-view,
        # zero-offset contiguous contract in forward and recomputation. Clone
        # keeps gradients; detach would silently break the coordinate flow.
        args = list(args)
        for index in (1, 2, 5, 6):
            args[index] = args[index].clone(memory_format=torch.contiguous_format)
        # Keep fixed feature widths out of symbolic AOT stride algebra. This
        # wrapper also runs during checkpoint recomputation: saved tensors may
        # acquire new Python wrappers and lose their Dynamo dimension hints.
        for tensor, dimensions in ((args[0], (1,)), (args[1], (1, 2)),
                (args[2], (1,)), (args[3], (1,)), (args[4], (1,)),
                (args[5], (1, 2)), (args[6], (1,)), (args[8], (2,))):
            torch._dynamo.mark_static(tensor, dimensions)
        return self._compiled_messages(*args)

    def forward(self, h, vectors, coords, types, edges, attributes, *, component_indices=None):
        if component_indices is None:
            component_indices = tuple((types == i).nonzero(as_tuple=False).flatten() for i in range(3))
        messages_fn = self._run_compiled_messages if self._compiled_messages is not None else self._messages
        keys, values = self.key(h), self.value(h)
        scalar, vector = [], []
        chunk_size = self.query_chunk_size if len(h) < 600 else 64
        for start in range(0, len(h), chunk_size):
            end = min(start + chunk_size, len(h))
            bonds = h.new_zeros(end - start, len(h), 4)
            selected = (edges[0] >= start) & (edges[0] < end)
            bonds[edges[0, selected] - start, edges[1, selected]] = attributes[selected]
            args = (self.query(h[start:end]), vectors[start:end], coords[start:end],
                    keys, values, vectors, coords, component_indices, bonds)
            if self.training and torch.is_grad_enabled():
                messages = checkpoint(messages_fn, *args, use_reentrant=False)
            else:
                messages = messages_fn(*args)
            scalar.append(messages[0])
            vector.append(messages[1])
        vectors = vectors + self.vector_mix(torch.cat(vector).transpose(1, 2)).transpose(1, 2)
        # A smooth invariant rescaling bounds repeated vector transport.
        vectors = vectors / (1. + vectors.square().mean((1, 2), keepdim=True)).sqrt()
        norms = vectors.square().sum(-1).clamp_min(1e-12).sqrt()
        h = self.norm(h + self.scalar_mix(torch.cat((h, torch.cat(scalar), norms), -1)))
        return h, vectors


class LegalPoseDecoder(nn.Module):
    """Decode legal DOFs, with explicit legacy or horizontal-equivalent rotors.

    Reversing a bond and complementing its rotating fragment describes the
    same positive horizontal tangent. V2 symmetrizes every directed read and
    measures shared action support from that tangent, not the raw mask.
    """

    def __init__(self, width, channels, *,
                 torsion_representation_contract=LEGACY_TORSION_REPRESENTATION,
                 identity_initialization=False, torsion_phase_contract=None,
                 target_ligand_frame_contract=None):
        super().__init__()
        if torsion_representation_contract not in (LEGACY_TORSION_REPRESENTATION, TORSION_REPRESENTATION_CONTRACT):
            raise ValueError("unknown torsion representation contract")
        self.equivalent_rotors = torsion_representation_contract == TORSION_REPRESENTATION_CONTRACT
        self.width = width
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.distance = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width + 1))
        self.bond_query = nn.Linear(2 * width + 2 * channels + 2, width)
        self.bond_geometry = nn.Sequential(nn.Linear(2 * channels + 3, width), nn.SiLU(),
                                           nn.Linear(width, width + 1))
        self.fragment = nn.Linear(2 * width, width)
        self.readout = nn.Linear(3 * width, width)
        self.joint = nn.MultiheadAttention(width, 4 if width % 4 == 0 else 1,
                                          dropout=0., batch_first=True)
        self.joint_relation = nn.Linear(5, self.joint.num_heads, bias=False)
        self.norm = nn.LayerNorm(width)
        self.rigid = nn.ModuleDict({branch: nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 2 * channels))
            for branch in BRANCHES if not (target_ligand_frame_contract and branch == 'e3_ligand')})
        self.bond = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.target_phase_head = (nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(),
            nn.Linear(width, 2)) if torsion_phase_contract == TORSION_PHASE_CONTRACT else None)
        if self.target_phase_head is not None:
            nn.init.normal_(self.target_phase_head[-1].weight, std=.01)
            with torch.no_grad():
                self.target_phase_head[-1].bias.copy_(torch.tensor([1., 0.]))
        self.target_frame_head = (nn.Sequential(nn.Linear(4 * width + 6 + 6 * channels, width), nn.SiLU(),
            nn.Linear(width, 9)) if target_ligand_frame_contract == TARGET_LIGAND_FRAME_CONTRACT else None)
        if self.target_frame_head is not None:
            nn.init.normal_(self.target_frame_head[-1].weight, std=.01)
            with torch.no_grad():
                self.target_frame_head[-1].bias.copy_(
                    torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0.]))
        for head in (*self.rigid.values(), self.bond):
            if identity_initialization:
                nn.init.zeros_(head[-1].weight)
            else:
                nn.init.normal_(head[-1].weight, std=.01)
            nn.init.zeros_(head[-1].bias)

    def _relations(self, queries, positions, h, coords):
        distance = torch.linalg.vector_norm(positions[:, None] - coords[None], dim=-1) / 10.
        relation = self.distance(torch.log1p(distance)[..., None])
        logits = queries @ self.key(h).T / math.sqrt(self.width) + relation[..., -1]
        return logits, relation[..., :-1]

    @staticmethod
    def joint_relations(positions, affected):
        """Physical DOF coupling: separation, shared atoms and directed overlap."""
        distance = torch.linalg.vector_norm(positions[:, None] - positions[None], dim=-1) / 10.
        count = affected.sum(-1)
        shared = affected @ affected.T
        union = count[:, None] + count[None] - shared
        return torch.stack((torch.log1p(distance), shared / union.clamp_min(1.),
                            shared / count[:, None].clamp_min(1.)), -1)

    @staticmethod
    def rotor_tangents(xyz, bonds, masks):
        """Unit positive-angle atom derivatives in the executor's horizontal gauge."""
        u, v = bonds
        axis = xyz[u] - xyz[v]
        length = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        if bool((length <= 1e-10).any()):
            raise ValueError('degenerate torsion axis')
        axis = axis / length
        raw = torch.cross(axis[:, None].expand(-1, len(xyz), -1),
                          xyz[None] - xyz[v, None], dim=-1) * masks[..., None]
        spin = horizontal_torsion_spin(xyz[None], raw, .5 * axis).to(xyz.dtype)
        tangent = raw - raw.mean(1, keepdim=True) - torch.cross(
            spin[:, None].expand_as(raw), (xyz - xyz.mean(0))[None].expand_as(raw), dim=-1)
        return axis, tangent

    @staticmethod
    def action_geometry(coords, vectors, ligand, bonds, masks):
        """Invariant geometry with the executor's local horizontal rotor derivative."""
        xyz = coords[ligand]
        u, v = bonds
        axis, horizontal = LegalPoseDecoder.rotor_tangents(xyz, bonds, masks)
        lever = (coords[None] - xyz[v, None]) / 10.
        axial = (lever * axis[:, None]).sum(-1, keepdim=True)
        radial = lever - axial * axis[:, None]
        affected = coords.new_zeros(len(u), len(coords))
        affected[:, ligand] = masks.to(coords.dtype)
        tangent = coords.new_zeros(len(u), len(coords), 3)
        tangent[:, ligand] = horizontal / 10.
        return torch.cat((axial, torch.linalg.vector_norm(radial, dim=-1, keepdim=True),
                          affected[..., None], (vectors[None] * axis[:, None, None]).sum(-1),
                          (vectors[None] * tangent[:, :, None]).sum(-1)), -1)

    def forward(self, h, vectors, coords, types, bonds, masks, state, opinions,
                phase_geometry=None, anchor_role_logits=None):
        time = state[:, -1:]
        if bool(((time < 0) | (time > 1)).any()):
            raise ValueError('remaining flow time must be in [0, 1]')
        # Time conditions the controller upstream; it is not a reciprocal gain.
        # Predict physical velocities, integrated exactly once by the executor.
        components = (2, 1)  # Physical ligand and POI DOFs, not learned global tokens.
        queries = [h[types == component].mean(0) for component in components]
        positions = [coords[types == component].mean(0) for component in components]
        if bonds.shape[1]:
            ligand = types == 2
            lh, lv, xyz = h[ligand], vectors[ligand], coords[ligand]
            u, v = bonds
            axis = xyz[u] - xyz[v]
            length = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
            if bool((length <= 1e-10).any()):
                raise ValueError('degenerate torsion axis')
            axis = axis / length
            features = torch.cat((lh[u], lh[v],
                                  (lv[u] * axis[:, None]).sum(-1),
                                  (lv[v] * axis[:, None]).sum(-1),
                                  torch.stack(opinions, -1).tanh()), -1)
            if self.equivalent_rotors:
                reverse_features = torch.cat((lh[v], lh[u],
                    -(lv[v] * axis[:, None]).sum(-1),
                    -(lv[u] * axis[:, None]).sum(-1),
                    torch.stack(opinions, -1).tanh()), -1)
                features = .5 * (features + reverse_features)
            bond_queries = self.bond_query(features)
            rotor_local = bond_queries
            midpoint = (xyz[u] + xyz[v]) / 2.
            action_features = self.action_geometry(coords, vectors, ligand, bonds, masks)
            geometry = self.bond_geometry(action_features)
            if self.equivalent_rotors:
                # Average nonlinear reads, retaining even directional evidence
                # instead of cancelling signed inputs before the MLP.
                reverse_geometry = self.bond_geometry(self.action_geometry(
                    coords, vectors, ligand, bonds.flip(0), ~masks))
                geometry = .5 * (geometry + reverse_geometry)
            logits, distance_values = self._relations(bond_queries, midpoint, lh, xyz)
            logits = logits + geometry[:, ligand, -1]
            # Keep atom identity and geometry until bond-specific selection;
            # no unconditional fragment average before control.
            fragments = []
            for selected in (masks, ~masks):
                if not bool(selected.any(-1).all()):
                    raise ValueError('each rotating bond requires two nonempty fragments')
                weights = logits.masked_fill(~selected, -torch.inf).softmax(-1)
                fragments.append((weights[..., None] * (
                    self.value(lh)[None] + distance_values + geometry[:, ligand, :-1])).sum(1))
            fragment_features = torch.cat(fragments, -1)
            if self.equivalent_rotors:
                fragment_features = .5 * (fragment_features + torch.cat(fragments[::-1], -1))
            bond_queries = bond_queries + self.fragment(fragment_features)
            queries.extend(bond_queries.unbind(0))
            positions.extend(midpoint.unbind(0))
        queries, positions = torch.stack(queries), torch.stack(positions)
        affected = torch.stack([(types == component).to(h.dtype) for component in components])
        if bonds.shape[1]:
            bond_affected = h.new_zeros(len(masks), len(h))
            if self.equivalent_rotors:
                _, horizontal = self.rotor_tangents(coords[types == 2], bonds, masks)
                magnitude = horizontal.square().sum(-1)
                bond_affected[:, types == 2] = magnitude / magnitude.amax(-1, keepdim=True).clamp_min(1e-12)
            else:
                bond_affected[:, types == 2] = masks.to(h.dtype)
            affected = torch.cat((affected, bond_affected), 0)
        logits, distance_values = self._relations(queries, positions, h, coords)
        if bonds.shape[1]:
            logits = torch.cat((logits[:2], logits[2:] + geometry[..., -1]), 0)
        # Each physical DOF reads each component separately; large proteins
        # cannot swamp ligand evidence merely by having more nodes.
        context = []
        for component in range(3):
            selected = types == component
            values = self.value(h[selected])[None] + distance_values[:, selected]
            if bonds.shape[1]:
                values = torch.cat((values[:2], values[2:] + geometry[:, selected, :-1]), 0)
            context.append((logits[:, selected].softmax(-1)[..., None] * values).sum(1))
        decisions = queries + self.readout(torch.cat(context, -1))
        coupling = h.new_zeros(len(positions), len(positions), 2)
        if bonds.shape[1]:
            axes, tangents = self.rotor_tangents(coords[types == 2], bonds, masks)
            flat = tangents.flatten(1)
            unit = flat / flat.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
            axial_overlap = axes @ axes.T
            coupling[2:, 2:, 0] = axial_overlap.square() if self.equivalent_rotors else axial_overlap
            coupling[2:, 2:, 1] = unit @ unit.T
        relation = self.joint_relation(torch.cat((self.joint_relations(positions, affected),
                                                  coupling), -1)).permute(2, 0, 1)
        coupled, _ = self.joint(decisions[None], decisions[None], decisions[None],
                                attn_mask=relation, need_weights=False)
        decisions = self.norm(decisions + coupled[0])
        motion = {}
        for index, (branch, component, column) in enumerate(zip(BRANCHES, components, (0, 3))):
            if branch not in self.rigid:
                motion[branch] = {'tr': h.new_zeros(1, 3), 'rot': h.new_zeros(1, 3)}
                continue
            selected = types == component
            local = h[selected]
            coefficients = self.rigid[branch](torch.cat((
                local, decisions[index:index + 1].expand(len(local), -1)), -1))
            coefficients = coefficients.reshape(-1, 2, vectors.shape[1])
            logits = local @ decisions[index] / math.sqrt(self.width)
            if self.target_frame_head is not None and branch == 'poi_ligand':
                if anchor_role_logits is None:
                    raise ValueError('POI rigid motion requires predicted protein anchors')
                anchor = anchor_role_logits[branch]['protein'].sigmoid().clamp_min(1e-4)
                spread = (torch.quantile(anchor.detach(), .9)
                          - torch.quantile(anchor.detach(), .1)).square()
                logits = logits + .5 * spread * anchor.log()
            weights = logits.softmax(0)
            displacement, rotation = (
                weights[:, None, None] * (coefficients[..., None]
                                         * vectors[selected, None]).sum(2)).sum(0)
            shift = displacement[None] * 10.
            motion[branch] = {
                'tr': -shift / state[:, column:column + 1].exp(),
                'rot': -rotation[None]}
        torsion = (-self.bond(decisions[2:]).squeeze(-1)[None]
                   if bonds.shape[1] else h.new_empty(1, 0))
        if self.target_phase_head is not None and bonds.shape[1]:
            if phase_geometry is None:
                raise ValueError('target-phase control needs directed rotor quads')
            quads, rotor_indices = phase_geometry
            rotor_decisions = decisions[2:]
            rotor_contrast = torch.nn.functional.layer_norm(
                rotor_decisions - rotor_decisions.mean(0, keepdim=True), (self.width,))
            target_phase = torch.nn.functional.normalize(self.target_phase_head(
                torch.cat((rotor_decisions, rotor_local, rotor_contrast), -1)),
                dim=-1, eps=1e-6)
            if len(quads):
                current_phase = torsion_phase_pairs(coords[types == 2][None], quads)[0]
                predicted = target_phase[rotor_indices]
                dot = (current_phase * predicted).sum(-1)
                cross = current_phase[:, 0] * predicted[:, 1] - current_phase[:, 1] * predicted[:, 0]
                correction = torch.atan2(cross, dot) / math.pi
                torsion = torsion.index_copy(1, rotor_indices, correction[None])
            motion_target_phase = target_phase[None]
        for branch in BRANCHES:
            motion[branch]['tor'] = torsion
        motion.update(torsion=torsion, rigid_output_mode='independent_centroids_v1',
                      torsion_update_mode='continuous_horizontal_v1')
        if self.target_phase_head is not None:
            motion['target_torsion_phase'] = (motion_target_phase if bonds.shape[1]
                                              else h.new_empty(1, 0, 2))
        if self.target_frame_head is not None:
            if anchor_role_logits is None:
                raise ValueError('target ligand frame requires predicted protein anchor roles')
            origin, basis = e3_reference_frame(coords[types == 0][None])
            anchor_h, anchor_local, anchor_vectors = [], [], []
            for branch, component in zip(BRANCHES, (0, 1)):
                selected = types == component
                weights = anchor_role_logits[branch]['protein'].sigmoid()
                weights = weights / weights.sum().clamp_min(1e-6)
                anchor_h.append((weights[:, None] * h[selected]).sum(0))
                point = (weights[:, None] * coords[selected]).sum(0)
                anchor_local.append(((point-origin[0]) @ basis[0]) / 10.)
                direction = (weights[:, None, None] * vectors[selected]).sum(0)
                anchor_vectors.append((direction @ basis[0]).flatten())
            context = torch.cat((*anchor_h, h[types == 2].mean(0), decisions[0],
                                 *anchor_local, *anchor_vectors), -1)
            target = self.target_frame_head(context)
            motion['target_ligand_pose_local'] = torch.cat((
                anchor_local[0] + target[:3], target[3:]), -1)[None]
        return motion



class UnifiedTernaryController(nn.Module):
    """Nodewise scalar/vector interaction and bond-conditioned action readout.

    No native geometry enters this module. All three components remain explicit
    through every interaction layer; only the final action readout aggregates.
    """

    def __init__(self, input_dim, width, scalar_dim, vector_dim, source_conditioning,
                 torsion_representation_contract=LEGACY_TORSION_REPRESENTATION,
                 protein_local_geometry_contract='none', candidate_driven=False,
                 protac_anchor_contract=None, motion_contract=FLOW_MOTION_CONTRACT,
                 contact_guidance_contract=None, torsion_phase_contract=None,
                 target_ligand_frame_contract=None):
        super().__init__()
        channels = max(4, width // 4)
        self.scalar_dim, self.vector_dim = scalar_dim, vector_dim
        self.candidate_driven = candidate_driven
        self.motion_contract = motion_contract
        self.contact_guidance_contract = contact_guidance_contract
        self.torsion_phase_contract = torsion_phase_contract
        self.target_ligand_frame_contract = target_ligand_frame_contract
        if contact_guidance_contract == CONTACT_GUIDANCE_CONTRACT:
            self.contact_query = nn.Linear(width, width, bias=False)
            self.contact_key = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(2)])
            self.contact_distance = nn.Linear(1, 1, bias=False)
            self.contact_vector = nn.Linear(2, channels, bias=False)
            nn.init.zeros_(self.contact_distance.weight)
        elif contact_guidance_contract == ANCHOR_RELATION_CONTRACT:
            self.anchor_query = nn.Linear(width, width, bias=False)
            self.anchor_key = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(2)])
            self.anchor_ligand = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])
            self.anchor_protein = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])
            self.anchor_distance = nn.Linear(1, 1, bias=False)
            self.anchor_relation_bias = nn.Parameter(torch.zeros(2, 3))
            self.anchor_relation_vector = nn.Linear(6, channels, bias=False)
            nn.init.zeros_(self.anchor_distance.weight)
        self.protac_anchor_contract = protac_anchor_contract
        self.protac_anchor_scalar = (nn.Linear(5, width, bias=False)
            if protac_anchor_contract == PROTAC_ANCHOR_CONTRACT else None)
        self.fullatom_scalar = nn.Linear(13, width) if candidate_driven else None
        self.protein = nn.Linear(input_dim, width)
        self.protein_geometry_contract = protein_local_geometry_contract
        self.protein_geometry = (nn.Linear(33, width, bias=False)
                                 if protein_local_geometry_contract != 'none' else None)
        self.ligand = LigandFeatureEncoder(width)
        self.expert = (nn.Linear(2 * scalar_dim + 2, width)
                       if scalar_dim > 0 else None)
        self.types = nn.Embedding(3, width)
        self.clock = nn.Linear(9, width)
        self.structure = ComponentStructureEncoder(width)
        self.structure_read = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(3)])
        self.source = None
        self.source_names = ((PURE_CANDIDATE_SOURCE_NAMES if scalar_dim == 0 else CANDIDATE_SOURCE_NAMES)
                             if candidate_driven else SOURCE_NAMES)
        if source_conditioning:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(0)
                self.source = nn.Embedding(len(self.source_names)*(2 if candidate_driven else 1), width)
        self.quality = (nn.Sequential(nn.Linear(9 * width + 84, width), nn.SiLU(),
            nn.Linear(width, len(QUALITY_COMPONENTS)))
            if candidate_driven and motion_contract != DRIFTING_MOTION_CONTRACT else None)
        self.vector_input = nn.Linear(
            2 * vector_dim + (10 if candidate_driven else 3)
            + (2 if self.protac_anchor_scalar is not None else 0), channels, bias=False)
        self.layers = nn.ModuleList([EquivariantInteraction(width, channels) for _ in range(3)])
        self.assembly = LegalPoseDecoder(width, channels,
            torsion_representation_contract=torsion_representation_contract,
            identity_initialization=candidate_driven and motion_contract != DRIFTING_MOTION_CONTRACT,
            torsion_phase_contract=torsion_phase_contract,
            target_ligand_frame_contract=target_ligand_frame_contract)
        if protac_anchor_contract == PROTAC_ANCHOR_CONTRACT:
            for parameter in self.assembly.rigid.parameters():
                parameter.requires_grad_(False)

    def _anchor_readout(self, h, vectors, coords, types, supplied=None):
        """Predict both sides' roles and pair contacts, then return equivariant guidance."""
        ligand = types == 2
        ligand_query = self.anchor_query(h[ligand])
        contact_logits, anchor_role_logits = {}, {}
        for side, branch in enumerate(BRANCHES):
            receptor = types == side
            ligand_role = self.anchor_ligand[side](h[ligand]).squeeze(-1)
            protein_role = self.anchor_protein[side](h[receptor]).squeeze(-1)
            if supplied is not None:
                roles = supplied[branch]
                for name, predicted in (('ligand', ligand_role), ('protein', protein_role)):
                    observed = torch.as_tensor(roles[name], device=coords.device, dtype=torch.bool)
                    if observed.shape != predicted.shape or not bool(observed.any()):
                        raise ValueError(f'{branch} supplied {name} anchor roles differ from graph nodes')
                    if name == 'ligand':
                        ligand_role = torch.where(observed, predicted.new_tensor(8.), predicted.new_tensor(-8.))
                    else:
                        protein_role = torch.where(observed, predicted.new_tensor(8.), predicted.new_tensor(-8.))
            anchor_role_logits[branch] = {'ligand': ligand_role, 'protein': protein_role}
            delta = coords[receptor][None] - coords[ligand][:, None]
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            a = ligand_role.sigmoid()[:, None]
            b = protein_role.sigmoid()[None, :]
            relation = torch.stack((a*b, a*(1-b)+(1-a)*b, (1-a)*(1-b)), -1)
            contact_logits[branch] = (
                ligand_query @ self.anchor_key[side](h[receptor]).T
                / math.sqrt(ligand_query.shape[-1])
                + self.anchor_distance(torch.log1p(distance / 10.)).squeeze(-1)
                + (relation * self.anchor_relation_bias[side]).sum(-1))
        field = two_sided_anchor_relation_field(contact_logits, anchor_role_logits,
                                                coords, types)
        vectors = vectors + self.anchor_relation_vector(field.transpose(1, 2)).transpose(1, 2)
        return h, vectors, contact_logits, anchor_role_logits

    def forward(self, graph, proposals, state, source, task_mode=None):
        coords, types = graph['node_coords'], graph['node_types']
        scalar, raw_vectors = graph.get('expert_node_features'), graph.get('expert_node_vectors')
        if self.expert is None:
            scalar = coords.new_zeros((len(coords), 0))
            raw_vectors = coords.new_zeros((len(coords), 0, 3))
        elif scalar is None or scalar.shape != (len(coords), 2 * self.scalar_dim + 2):
            raise ValueError('assembly requires node-aligned binary scalar evidence')
        if raw_vectors is None or raw_vectors.shape != (len(coords), 2 * self.vector_dim, 3):
            raise ValueError('assembly requires node-aligned binary vector evidence')
        if state.shape != (1, 9) or not all((types == i).any() for i in range(3)):
            raise ValueError('assembly requires one clock and three nonempty components')
        if not all(torch.isfinite(x).all() for x in (coords, scalar, raw_vectors, state)):
            raise ValueError('assembly inputs must be finite')
        h = self.protein(graph['node_features'])
        if self.protein_geometry is not None:
            geometry = graph.get('protein_local_geometry')
            if geometry is None:
                if self.protein_geometry_contract == 'intrinsic32_required_v1':
                    raise ValueError('controller requires protein local geometry')
                geometry = h.new_zeros(len(h), 33)
            if geometry.shape != (len(h), 33) or not torch.isfinite(geometry).all():
                raise ValueError('controller protein geometry must be node-aligned [N,33]')
            protein = types != 2
            availability = geometry[:, -1:]
            if not ((availability == 0) | (availability == 1)).all():
                raise ValueError('controller geometry availability must be binary')
            if self.protein_geometry_contract == 'intrinsic32_required_v1' and not availability[protein].bool().all():
                raise ValueError('controller requires measured protein geometry')
            geometry = torch.cat((geometry[:, :32] * availability, availability), -1)
            h = h + self.protein_geometry(geometry.to(h)) * protein[:, None]
        ligand = types == 2
        h = h.clone()
        h[ligand] = self.ligand(graph['node_features'][ligand])
        identity = h + self.types(types)
        h = identity + self.clock(state)
        if self.expert is not None:
            h = h + self.expert(scalar)
        anchor_vectors = None
        if self.protac_anchor_scalar is not None:
            anchor_scalar = graph.get('protac_anchor_scalar')
            anchor_vectors = graph.get('protac_anchor_vectors')
            if (anchor_scalar is None or anchor_scalar.shape != (len(h), 5)
                    or anchor_vectors is None or anchor_vectors.shape != (len(h), 2, 3)
                    or not torch.isfinite(anchor_scalar).all()
                    or not torch.isfinite(anchor_vectors).all()):
                raise ValueError('PROTAC model requires node-aligned binary-anchor context')
            h = h + self.protac_anchor_scalar(anchor_scalar.to(h))
        fullatom_vectors = None
        if self.candidate_driven:
            physical = graph.get('fullatom_scalar_features')
            fullatom_vectors = graph.get('fullatom_vector_features')
            if (physical is None or fullatom_vectors is None or physical.shape != (len(h), 9)
                    or fullatom_vectors.shape != (len(h), 3, 3)
                    or not torch.isfinite(physical).all() or not torch.isfinite(fullatom_vectors).all()):
                raise ValueError('candidate controller requires node-aligned all-atom proximity evidence')
            reference_scalar = graph.get('candidate_reference_scalar')
            reference_vectors = graph.get('candidate_reference_vectors')
            if (reference_scalar is None or reference_vectors is None
                    or reference_scalar.shape != (len(h), 4)
                    or reference_vectors.shape != (len(h), 4, 3)
                    or not torch.isfinite(reference_scalar).all()
                    or not torch.isfinite(reference_vectors).all()):
                raise ValueError('candidate correction requires immutable candidate-reference evidence')
            h = h + self.fullatom_scalar(torch.cat((physical, reference_scalar), -1))
            fullatom_vectors = torch.cat((fullatom_vectors, reference_vectors), 1)
        if self.source is not None:
            index = (task_source_index(source, task_mode, h, self.source_names) if self.candidate_driven else
                     candidate_source_tensor(source, h, self.source_names))
            h = h + self.source(index)
        centers = torch.stack([coords[types == i].mean(0) for i in range(3)])
        offsets = bounded_vector((coords[:, None] - centers[None]) / 10.)
        # Conditional binary node evidence informs the joint controller; binary
        # rigid-head outputs are not broadcast as shared physical-body actions.
        vectors = torch.cat((bounded_vector(raw_vectors), offsets), 1)
        if anchor_vectors is not None:
            vectors = torch.cat((vectors, bounded_vector(anchor_vectors.to(vectors))), 1)
        if fullatom_vectors is not None:
            vectors = torch.cat((vectors, fullatom_vectors), 1)
        vectors = self.vector_input(vectors.transpose(1, 2)).transpose(1, 2)
        component_indices = graph.get('component_indices')
        ligand_ids = (component_indices[2] if component_indices is not None
                      else ligand.nonzero(as_tuple=False).flatten())
        local_edges = graph['lig_edge_index']
        edges = graph.get('global_ligand_edges')
        if edges is None:
            edges = ligand_ids[local_edges]
        attributes = graph.get('lig_edge_attr')
        if attributes is None:
            attributes = h.new_zeros(local_edges.shape[1], 4)
        if attributes.shape != (local_edges.shape[1], 4):
            raise ValueError('assembly requires four chemical bond attributes per edge')
        memory = self.structure(identity, coords, types, edges, attributes,
                                component_indices=component_indices, ligand_edges=local_edges)
        for layer, read in zip(self.layers, self.structure_read):
            h = read(torch.cat((h, memory), -1))
            h, vectors = layer(h, vectors, coords, types, edges, attributes,
                               component_indices=component_indices)
        contact_logits = None
        anchor_role_logits = None
        if self.contact_guidance_contract == CONTACT_GUIDANCE_CONTRACT:
            # Predict target pockets from the current ternary graph, then feed
            # their equivariant directions to both movable bodies. No native
            # coordinates or teacher contacts enter this inference path.
            ligand_query = self.contact_query(h[ligand])
            contact_logits = {}
            for side, branch in enumerate(BRANCHES):
                receptor = types == side
                delta = coords[receptor][None] - coords[ligand][:, None]
                distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                logits = (ligand_query @ self.contact_key[side](h[receptor]).T
                          / math.sqrt(ligand_query.shape[-1])
                          + self.contact_distance(torch.log1p(distance / 10.)).squeeze(-1))
                contact_logits[branch] = logits
            field = two_sided_contact_direction_field(contact_logits, coords, types)
            vectors = vectors + self.contact_vector(field.transpose(1, 2)).transpose(1, 2)
        elif self.contact_guidance_contract == ANCHOR_RELATION_CONTRACT:
            h, vectors, contact_logits, anchor_role_logits = self._anchor_readout(
                h, vectors, coords, types, graph.get('supplied_anchor_roles'))
        bonds = local_edges[:, graph['lig_edge_mask'].bool()]
        opinions = ([proposals[branch]['tor'].flatten() for branch in BRANCHES]
                    if proposals else [coords.new_zeros(bonds.shape[1]),
                                       coords.new_zeros(bonds.shape[1])])
        if all(len(x) == 0 for x in opinions):
            bonds = bonds[:, :0]
            masks = torch.zeros(0, len(ligand_ids), device=coords.device, dtype=torch.bool)
        else:
            if any(len(x) != bonds.shape[1] for x in opinions):
                raise ValueError('assembly expert torsions must match canonical rotating bonds')
            masks = torch.as_tensor(graph['lig_mask_rotate'], device=coords.device, dtype=torch.bool)
        if masks.shape != (bonds.shape[1], len(ligand_ids)):
            raise ValueError('assembly torsion masks must match directed rotating bonds')
        phase_geometry = (ligand_rotor_quads(local_edges, graph['lig_edge_mask'])
            if self.torsion_phase_contract else None)
        body_motion = self.assembly(h, vectors, coords, types, bonds, masks, state, opinions,
                                    phase_geometry=phase_geometry,
                                    anchor_role_logits=anchor_role_logits)
        if self.protac_anchor_scalar is not None:
            allowed = graph.get('protac_torsion_mask')
            if allowed is None or allowed.shape != body_motion['torsion'].shape[1:]:
                raise ValueError('PROTAC controller requires the linker-only torsion action mask')
            body_motion['torsion'] = body_motion['torsion'] * allowed[None].to(
                body_motion['torsion'])
            for branch in BRANCHES:
                body_motion[branch]['tr'] = torch.zeros_like(body_motion[branch]['tr'])
                body_motion[branch]['rot'] = torch.zeros_like(body_motion[branch]['rot'])
                body_motion[branch]['tor'] = body_motion['torsion']
        components = torch.stack([h[types == i].mean(0) for i in range(3)])
        motion_key = ('normalized_action' if self.motion_contract == DRIFTING_MOTION_CONTRACT
                      else 'normalized_velocity')
        result = {motion_key: body_motion,
                  'rank_features': components.mean(0, keepdim=True),
                  'residual_penalty': h.new_zeros(())}
        if contact_logits is not None:
            result['contact_logits'] = contact_logits
        if anchor_role_logits is not None:
            result['anchor_role_logits'] = anchor_role_logits
        if self.quality is not None:
            # Read actual contacts, not time/source/reference-conditioned motion
            # activations. Identity stays explicit but identical across a pool.
            def statistics(nodes):
                return torch.cat((nodes.mean(0), nodes.amax(0),
                    nodes.var(0, unbiased=False).add(1e-8).sqrt()))
            quality_features = torch.cat((
                *(statistics(identity[types == i]) for i in range(3)),
                *(statistics(physical[types == i]) for i in range(3)),
                torch.pdist(centers).log1p())).unsqueeze(0).detach()
            errors = torch.nn.functional.softplus(self.quality(quality_features))
            result.update(quality_errors=errors, candidate_score=-errors.mean(-1),
                          candidate_quality_features=quality_features)
            if (self.motion_contract == QUALITY_GATED_REPAIR_MOTION_CONTRACT
                    and candidate_source_tensor(task_mode, h, TASK_NAMES).item() == 1):
                # A native-free predicted error controls edit magnitude. Cap
                # the result again at the initial gate in refine_candidates.
                result['repair_gate'] = (errors.mean(-1, keepdim=True)
                    / QUALITY_GATE_ERROR_SCALE).clamp(min=.05, max=1.)
        return result


class TernaryAssembly(nn.Module):
    """The sole production controller; retired motion architectures are rejected."""

    supports_conditions = False
    direct_ternary_motion = True

    def __init__(self, node_dim=128, input_dim=None, source_conditioning=False, *,
                 direct_ternary_motion=True, model_profile=VECTOR_EXPERT_MODEL_PROFILE,
                 expert_free=None,
                 expert_scalar_dim=58, expert_vector_dim=20, expert_backend='diffdock',
                 conditioning_contract=MATCHA_CONDITIONING_CONTRACT,
                 information_contract=INFORMATION_CONTRACT,
                 rigid_decoder_contract=RIGID_DECODER_CONTRACT,
                 torsion_representation_contract=LEGACY_TORSION_REPRESENTATION,
                 protein_local_geometry_contract='none', candidate_driven=False,
                 protac_anchor_contract=None,
                 candidate_geometry_contract=None, candidate_quality_contract=None,
                 candidate_condition_contract=None, candidate_reference_contract=None,
                 candidate_expert_frame_policy='backbone_rigid_transport_v1', candidate_task_contract=None,
                 motion_contract=FLOW_MOTION_CONTRACT, contact_guidance_contract=None,
                 torsion_phase_contract=None, target_ligand_frame_contract=None):
        # A saved model_config without the new field reconstructs historical
        # behavior. New training recipes use production_model_config's V2.
        super().__init__()
        if candidate_geometry_contract is not None and candidate_geometry_contract != CANDIDATE_GEOMETRY_CONTRACT:
            raise ValueError('retired candidate geometry contract')
        if candidate_quality_contract not in (None, CANDIDATE_QUALITY_CONTRACT):
            raise ValueError('retired candidate quality contract')
        if candidate_condition_contract not in (None, CANDIDATE_CONDITION_CONTRACT):
            raise ValueError('retired candidate conditioning contract')
        expected_reference = (PURE_CANDIDATE_REFERENCE_CONTRACT
            if model_profile == PURE_TERNARY_MODEL_PROFILE else CANDIDATE_REFERENCE_CONTRACT)
        if candidate_reference_contract not in (None, expected_reference):
            raise ValueError('retired candidate reference contract')
        if candidate_task_contract not in (None, CANDIDATE_TASK_CONTRACT):
            raise ValueError('retired candidate task contract')
        if model_profile not in (VECTOR_EXPERT_MODEL_PROFILE, PURE_TERNARY_MODEL_PROFILE) or not direct_ternary_motion:
            raise ValueError('retired architecture: construct a supported direct ternary model')
        expected_expert_free = model_profile == PURE_TERNARY_MODEL_PROFILE
        if expert_free is not None and bool(expert_free) != expected_expert_free:
            raise ValueError('checkpoint expert-free marker disagrees with its model profile')
        self.config = production_model_config(
            node_dim, input_dim or node_dim, source_conditioning,
            model_profile=model_profile, expert_scalar_dim=expert_scalar_dim,
            expert_vector_dim=expert_vector_dim, expert_backend=expert_backend,
            conditioning_contract=conditioning_contract, information_contract=information_contract,
            rigid_decoder_contract=rigid_decoder_contract,
            torsion_representation_contract=torsion_representation_contract,
            protein_local_geometry_contract=protein_local_geometry_contract,
            candidate_driven=candidate_driven, candidate_expert_frame_policy=candidate_expert_frame_policy,
            protac_anchor_contract=protac_anchor_contract, motion_contract=motion_contract,
            contact_guidance_contract=contact_guidance_contract,
            torsion_phase_contract=torsion_phase_contract,
            target_ligand_frame_contract=target_ligand_frame_contract)
        self.controller = UnifiedTernaryController(
            self.config['input_dim'], node_dim, self.config['expert_scalar_dim'],
            self.config['expert_vector_dim'], source_conditioning, torsion_representation_contract,
            protein_local_geometry_contract, candidate_driven, protac_anchor_contract,
            motion_contract, contact_guidance_contract, torsion_phase_contract,
            target_ligand_frame_contract)
        self.condition_protein_geometry = (nn.Linear(33, node_dim, bias=False)
            if expert_backend == "matcha" and protein_local_geometry_contract != "none" else None)
        if expert_backend == 'matcha':
            self.condition_query = nn.Linear(self.config['input_dim'], node_dim)
            self.condition_key = nn.Linear(self.config['input_dim'], node_dim)
            self.condition_value = nn.Linear(self.config['input_dim'], node_dim)
            self.condition_geometry = nn.Linear(3, node_dim, bias=False)
            self.condition_fullatom = nn.Linear(13, node_dim, bias=False) if candidate_driven else None
            self.condition_clock = nn.Linear(9, node_dim) if candidate_driven else None
            self.condition_ligand = LigandFeatureEncoder(node_dim)
            self.condition_chemistry = ComponentStructureEncoder(node_dim)
            self.condition_interface = nn.Linear(2 * node_dim + 3, node_dim)
            nn.init.zeros_(self.condition_geometry.weight)
            self.binary_film = nn.ModuleList([nn.Linear(node_dim, 2 * expert_scalar_dim) for _ in range(3)])
            for film in self.binary_film:
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)

    def binary_condition(self, receptor, opposite, ligand, *, stage_index=0,
                         ablation_mode='all', diffusion_state=None, source=0, task_mode=None):
        """Nodewise opposite-protein context; no target or pooled condition token."""
        if isinstance(stage_index, bool) or not isinstance(stage_index, int) or not 0 <= stage_index < 3:
            raise ValueError('Matcha condition requires an explicit valid stage index')
        if ablation_mode not in ('all', 'off', 'early', 'late'):
            raise ValueError('unknown binary FiLM ablation mode')
        q = self.condition_query(receptor['features'][0])
        k = self.condition_key(opposite['features'][0])
        value = self.condition_value(opposite['features'][0])
        if self.condition_fullatom is not None:
            receiving = receptor.get('candidate_fullatom_features')
            sending = opposite.get('candidate_fullatom_features')
            if (receiving is None or sending is None or receiving.shape != (len(q), 13)
                    or sending.shape != (len(k), 13)
                    or not torch.isfinite(receiving).all() or not torch.isfinite(sending).all()):
                raise ValueError('candidate FiLM requires both proteins\' full-atom interface evidence')
            if (diffusion_state is None or diffusion_state.shape != (1, 9)
                    or not torch.isfinite(diffusion_state).all()):
                raise ValueError('candidate FiLM requires the shared refinement clock')
            receiving = self.condition_fullatom(receiving)
            sending = self.condition_fullatom(sending)
            clock = self.condition_clock(diffusion_state)
            source_feature = self.controller.source(task_source_index(source, task_mode, q))
            q = q + receiving + clock + source_feature
            k = k + sending
            value = value + sending
        if self.condition_protein_geometry is not None:
            required = self.config['protein_local_geometry_contract'] == 'intrinsic32_required_v1'
            receiving_geometry = self.condition_protein_geometry(
                protein_local_geometry_features(receptor, required=required))
            opposite_geometry = self.condition_protein_geometry(
                protein_local_geometry_features(opposite, required=required))
            q = q + receiving_geometry
            k = k + opposite_geometry
            value = value + opposite_geometry
        ligand_identity = self.condition_ligand(ligand['features'][0])
        ligand_identity = self.condition_chemistry(
            ligand_identity, ligand['coords'][0],
            torch.full((len(ligand_identity),), 2, device=q.device, dtype=torch.long),
            ligand['edge_index'], ligand['edge_attr'])
        interface_distance = torch.cdist(opposite['coords'][0], ligand['coords'][0],
                                        compute_mode='donot_use_mm_for_euclid_dist') / 10.
        interface_weights = (-interface_distance).softmax(-1)
        interface_log = torch.log1p(interface_distance)
        interface_geometry = torch.stack(((interface_weights * interface_log).sum(-1),
                                          (interface_weights * interface_log.square()).sum(-1),
                                          torch.log1p(torch.exp(-interface_distance.square()).sum(-1))), -1)
        value = value + self.condition_geometry(interface_geometry)
        distance = torch.cdist(receptor['coords'][0], opposite['coords'][0],
                               compute_mode='donot_use_mm_for_euclid_dist') / 10.
        weights = (q @ k.T / math.sqrt(q.shape[-1]) - torch.log1p(distance)).softmax(-1)
        # Softmax-weighted identity alone erases distance when values coincide.
        # Keep absolute nodewise separation, spread and contact density without
        # materializing an [N_receptor,N_opposite,width] message tensor.
        log_distance = torch.log1p(distance)
        geometry = torch.stack(((weights * log_distance).sum(-1),
                                (weights * log_distance.square()).sum(-1),
                                torch.log1p(torch.exp(-distance.square()).sum(-1))), -1)
        # Keep ligand atom slots until the receiving receptor node selects them.
        # Matrix association avoids allocating receptor x opposite x ligand.
        atom_support = weights @ interface_weights
        receiver_distance = torch.cdist(receptor['coords'][0], ligand['coords'][0],
                                        compute_mode='donot_use_mm_for_euclid_dist') / 10.
        atom_weights = (torch.log(atom_support.clamp_min(1e-30)) - receiver_distance).softmax(-1)
        context = self.condition_interface(torch.cat((weights @ value,
            atom_weights @ ligand_identity, geometry), -1)) + self.condition_geometry(geometry)
        if self.condition_fullatom is not None:
            context = context + receiving + clock + source_feature
        gamma, beta = self.binary_film[stage_index](context).chunk(2, -1)
        enabled = (ablation_mode == 'all' or
                   ablation_mode == 'early' and stage_index == 0 or
                   ablation_mode == 'late' and stage_index in (1, 2))
        if not enabled:
            gamma, beta = torch.zeros_like(gamma), torch.zeros_like(beta)
        return .1 * gamma.tanh().unsqueeze(0), .1 * beta.tanh().unsqueeze(0)

    def forward(self, graph, proposals, diffusion_state, source=0, *, conditions=None, task_mode=None):
        if conditions is not None and self.config.get('protac_anchor_contract') is None:
            if not isinstance(conditions, dict) or conditions.get('contract') != 'external_binary_anchor_roles_v1':
                raise ValueError('unsupported external anchor condition')
        return self.controller(graph, proposals, diffusion_state, source, task_mode)


class TernaryAssemblyModel(nn.Module):
    """Direct ternary model, optionally retaining a legacy frozen-evidence path."""

    architecture = DIRECT_ARCHITECTURE

    def __init__(self, frozen_expert, trifuse, *,
                 expert_input_grad=True, checkpoint_ternary=False, checkpoint_min_nodes=0):
        super().__init__()
        profile = trifuse.config.get('model_profile') if isinstance(trifuse, TernaryAssembly) else None
        if profile not in (VECTOR_EXPERT_MODEL_PROFILE, PURE_TERNARY_MODEL_PROFILE):
            raise ValueError('production requires a supported direct ternary controller')
        expert_free = profile == PURE_TERNARY_MODEL_PROFILE
        if expert_free and frozen_expert is not None:
            raise ValueError('pure ternary construction must not receive a binary expert')
        if not expert_free and frozen_expert is None:
            raise ValueError('legacy evidence model requires its frozen binary expert')
        if frozen_expert is not None and any('lora' in name for name, _ in frozen_expert.named_parameters()):
            raise ValueError('model-v2 Stage 1 requires an unmodified frozen expert')
        self.expert = frozen_expert.requires_grad_(False).eval() if frozen_expert is not None else None
        self.trifuse = trifuse
        self.architecture = (PURE_TERNARY_ARCHITECTURE if expert_free else DIRECT_ARCHITECTURE)
        if frozen_expert is not None:
            backend = getattr(frozen_expert, 'backend', 'diffdock')
            if backend != trifuse.config.get('expert_backend', 'diffdock'):
                raise ValueError('expert backend and controller configuration disagree')
            reader = 'read_pair' if backend == 'matcha' else 'forward_features'
            if (not callable(getattr(frozen_expert, reader, None))
                    or frozen_expert.node_scalar_dim != trifuse.config['expert_scalar_dim']
                    or frozen_expert.node_vector_dim != trifuse.config['expert_vector_dim']):
                raise ValueError('assembly requires matching frozen expert scalar/vector interfaces')
        # Keep pretrained internal edges; only cross edges receive tapering.
        if frozen_expert is not None and getattr(frozen_expert, 'backend', 'diffdock') == 'diffdock':
            frozen_expert.model.smooth_cross_edges = True
        self.expert_input_grad = bool(expert_input_grad) if self.expert is not None else False
        if trifuse.config.get('candidate_driven'):
            if self.expert is not None and not self.expert_input_grad:
                raise ValueError('candidate endpoint training requires complete expert coordinate gradients')
            if (self.expert is not None
                    and getattr(frozen_expert, 'frame_policy', None)
                    != trifuse.config['candidate_expert_frame_policy']):
                raise ValueError('candidate expert frame policy differs from the saved model contract')
        self.checkpoint_ternary = bool(checkpoint_ternary)
        if isinstance(checkpoint_min_nodes, bool) or not isinstance(checkpoint_min_nodes, int) or checkpoint_min_nodes < 0:
            raise ValueError('checkpoint_min_nodes must be a nonnegative integer')
        # Runtime memory policy only; saved model/loss semantics stay unchanged.
        self.checkpoint_min_nodes = checkpoint_min_nodes
        self.checkpoint_memory_reserve_bytes = 0
        self.checkpoint_activation_bytes_per_node_squared = 0
        self.checkpoint_memory_fraction = 1.0
        self._conditioning = SDPIFlow()

    def set_ternary_compile_messages(self, enabled):
        for module in self.trifuse.modules():
            if isinstance(module, EquivariantInteraction):
                module.set_compile_messages(enabled)

    def refine_candidates(self, e3_data, poi_data, ligand_data, candidates, *, steps=5):
        """Jointly refine generated assemblies; score only their actual final states.

        Native targets are not arguments. E3 remains fixed; POI rigid pose and
        ligand rigid/torsion coordinates share the legal geometric executor.
        The caller must supply a trained candidate-framework checkpoint before
        treating the learned scores as inference-time quality predictions.
        """
        from tricoopdiff.runtime import evaluation_context
        from tricoopdiff.initialization.candidates import _valid_candidate

        if not self.trifuse.config.get('candidate_driven') or not candidates:
            raise ValueError('joint refinement requires a candidate model and nonempty pool')
        if self.trifuse.config.get('motion_contract') == DRIFTING_MOTION_CONTRACT:
            if steps != 1:
                raise ValueError('drifting inference requires exactly one network evaluation')
            if any(not _valid_candidate(candidate) for candidate in candidates):
                raise ValueError('drifting pool contains invalid generated geometry')
            with evaluation_context(self):
                return [{**self.generate_candidate(e3_data, poi_data, ligand_data, candidate),
                         'candidate_id': candidate['candidate_id'],
                         'provenance': candidate['provenance']} for candidate in candidates]
        flow = SDPIFlow(inference_steps=steps)
        results = []
        with evaluation_context(self):
            for candidate in candidates:
                if not _valid_candidate(candidate):
                    raise ValueError('candidate pool contains invalid geometry')
                source = candidate.get('provenance', {}).get('candidate_source')
                source_names = (PURE_CANDIDATE_SOURCE_NAMES if self.trifuse.config.get('expert_free')
                                else CANDIDATE_SOURCE_NAMES)
                source_index = candidate_source_tensor(source, e3_data['coords'], source_names)
                ligand, poi, body = [candidate[key].to(e3_data['coords']).clone()
                                     for key in ('ligand', 'poi', 'rotation')]
                if ligand.shape != ligand_data['coords'].shape or poi.shape != poi_data['coords'].shape:
                    raise ValueError('candidate atom counts differ from indexed features')
                candidate_poi_data = {**poi_data, 'candidate_reference': {
                    'ligand': ligand.detach().clone(), 'poi': poi.detach().clone(),
                    'body': body.detach().clone()}, 'task_mode': 'refine'}
                initial_gate = None
                for t, dt in flow.reverse_times(ligand):
                    output = flow.forward_state(self, e3_data, candidate_poi_data, ligand_data,
                                                ligand, poi, body, t, dt, source_index)
                    if 'repair_gate' in output:
                        initial_gate = (output['repair_gate'].detach() if initial_gate is None
                                        else torch.minimum(initial_gate, output['repair_gate'].detach()))
                        output = {**output, 'repair_gate': initial_gate}
                    (ligand, poi, body), _ = flow.apply_prediction(
                        output, t, dt, e3_data['coords'], poi_data['coords'],
                        ligand, poi, body, ligand_data,
                        motion_contract=self.trifuse.config.get(
                            'motion_contract', FLOW_MOTION_CONTRACT))
                    if not all(torch.isfinite(x).all() for x in (ligand, poi, body)):
                        raise FloatingPointError('nonfinite refined candidate')
                zero = ligand.new_zeros(1)
                final = flow.forward_state(self, e3_data, candidate_poi_data, ligand_data,
                                           ligand, poi, body, zero, zero, source_index)
                if not all(torch.isfinite(final[key]).all()
                           for key in ('quality_errors', 'candidate_score')):
                    raise FloatingPointError('nonfinite terminal candidate quality')
                results.append({'ligand': ligand, 'poi': poi, 'rotation': body,
                    'candidate_id': candidate['candidate_id'],
                    'provenance': candidate['provenance'],
                    'quality_errors': final['quality_errors'],
                    'candidate_score': final['candidate_score']})
        return results

    def generate_candidate(self, e3_data, poi_data, ligand_data, candidate, *,
                           return_aux=False, conditions=None):
        """Predict and execute one legal action on the same candidate pose."""
        if self.trifuse.config.get('motion_contract') != DRIFTING_MOTION_CONTRACT:
            raise ValueError('one-pass generation requires the drifting motion contract')
        ligand, poi, body = [candidate[key].to(e3_data['coords'])
                             for key in ('ligand', 'poi', 'rotation')]
        if ligand.shape != ligand_data['coords'].shape or poi.shape != poi_data['coords'].shape:
            raise ValueError('drifting input and decoder template must share indexed atom counts')
        scales = SDPIFlow.scales(e3_data['coords'], poi_data['coords'])
        state = ligand.new_zeros(1, 9)
        state[:, 0], state[:, 3] = scales[0].log(), scales[1].log()
        state[:, 5:] = 1.
        pd = {**poi_data, 'coords': poi, 'candidate_reference': {
            'ligand': ligand.detach(), 'poi': poi.detach(), 'body': body.detach()}}
        output = self(e3_data, pd, {**ligand_data, 'coords': ligand},
                      t_e3l=ligand.new_ones(1), r_e3_to_poi=body,
                      diffusion_state=state, source='hdock_etkdg', task_mode='refine',
                      conditions=conditions)
        action = SDPIFlow().velocities(output['normalized_action'],
                                      e3_data['coords'], poi_data['coords'])
        target_pose = output['normalized_action'].get('target_ligand_pose_local')
        if target_pose is not None:
            action['e3_ligand']['tr'] = torch.zeros_like(action['e3_ligand']['tr'])
            action['e3_ligand']['rot'] = torch.zeros_like(action['e3_ligand']['rot'])
        ligand, poi, body = SDPIFlow.differentiable_step(
            ligand, poi, body, action, ligand.new_ones(1), ligand_data)
        if target_pose is not None:
            ligand = place_ligand_in_target_frame(
                ligand, e3_data['coords'], ligand_data['edge_index'], target_pose)
        result = {'ligand': ligand, 'poi': poi, 'rotation': body}
        if return_aux:
            result['normalized_torsion_action'] = output['normalized_action']['torsion']
            result['normalized_rigid_action'] = {
                branch: {component: output['normalized_action'][branch][component]
                         for component in ('tr', 'rot')}
                for branch in BRANCHES}
            if 'target_torsion_phase' in output['normalized_action']:
                result['predicted_torsion_target_phase'] = output['normalized_action']['target_torsion_phase']
            if target_pose is not None:
                result['predicted_ligand_pose_local'] = target_pose
        if return_aux and 'contact_logits' in output:
            result['contact_logits'] = output['contact_logits']
        if return_aux and 'anchor_role_logits' in output:
            result['anchor_role_logits'] = output['anchor_role_logits']
        return result

    def set_ternary_query_chunk_size(self, chunk_size):
        """Batch equivalent query rows; retain 64-row chunks on large graphs."""
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('ternary query chunk size must be a positive integer')
        for module in self.trifuse.modules():
            if isinstance(module, EquivariantInteraction):
                module.query_chunk_size = chunk_size

    def _should_checkpoint_ternary(self, node_count, device):
        """Select recomputation from size and deterministic runtime headroom.

        Allocator-reserved but unallocated bytes are reusable. The node-squared
        activation budget is a conservative heuristic, not an OOM guarantee.
        Never retry a failed backward or replay synchronized gradients here.
        """
        if not self.checkpoint_ternary:
            return False
        if node_count >= self.checkpoint_min_nodes:
            return True
        reserve = self.checkpoint_memory_reserve_bytes
        if reserve <= 0 or device.type != 'cuda':
            return False
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        available = min(self.checkpoint_memory_fraction * total - allocated,
                        free + max(reserved - allocated, 0))
        budget = reserve + self.checkpoint_activation_bytes_per_node_squared * node_count ** 2
        return available < budget

    def _expert_context(self):
        # Frozen weights do not require detached input Jacobians. Respect the
        # caller's no_grad during inference; never force enable_grad here.
        # Matcha FiLM belongs to the trainable controller, not the frozen prior.
        # Its gradient must survive even when expert coordinate VJPs are disabled.
        if self.expert_input_grad or getattr(self.expert, 'backend', None) == 'matcha':
            return nullcontext()
        return torch.no_grad()

    def _read_expert(self, graph):
        # Recompute frozen activations during backward rather than retaining
        # eleven expert layers at every rollout step. Clone because DiffDock
        # stores intermediate attributes on its private graph.
        def read(private_graph):
            return self.expert.forward_features(private_graph.clone(), include_vectors=True)

        needs_input_grad = (graph['ligand'].pos.requires_grad
                            or graph['receptor'].pos.requires_grad)
        if self.expert_input_grad and torch.is_grad_enabled() and needs_input_grad:
            return checkpoint(read, graph, use_reentrant=False)
        return read(graph)

    @staticmethod
    def _add_protac_anchor_context(graph, condition, ligand_data):
        """Attach target-free roles and equivariant vectors from both warheads."""
        coords, types = graph['node_coords'], graph['node_types']
        component_indices = graph.get('component_indices')
        ligand_ids = (component_indices[2] if component_indices is not None
                      else (types == 2).nonzero(as_tuple=False).flatten())
        first = torch.as_tensor(condition['anchor1_atoms'], device=coords.device,
                                dtype=torch.long).flatten()
        second = torch.as_tensor(condition['anchor2_atoms'], device=coords.device,
                                 dtype=torch.long).flatten()
        n_ligand = len(ligand_ids)
        if (len(first) < 3 or len(second) < 3 or
                set(first.tolist()) & set(second.tolist()) or
                min(int(first.min()), int(second.min())) < 0 or
                max(int(first.max()), int(second.max())) >= n_ligand):
            raise ValueError('PROTAC anchor roles must be disjoint valid ligand atom indices')
        roles = coords.new_zeros(len(coords), 3)
        ligand_offset = int(ligand_ids[0])
        roles[ligand_ids, 2] = 1
        roles[ligand_offset + first, 0] = 1
        roles[ligand_offset + second, 1] = 1
        roles[ligand_offset + first, 2] = 0
        roles[ligand_offset + second, 2] = 0
        centers = torch.stack((coords[ligand_offset + first].mean(0),
                               coords[ligand_offset + second].mean(0)))
        displacement = (centers[None] - coords[:, None]) / 10.
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        graph['protac_anchor_scalar'] = torch.cat((roles, torch.log1p(distances)), -1)
        graph['protac_anchor_vectors'] = displacement
        from tricoopdiff.initialization.protac import linker_torsion_mask
        torsion_mask = linker_torsion_mask(ligand_data, first, second).to(coords.device)
        if not bool(torsion_mask.any()):
            raise ValueError('PROTAC condition has no rotatable bond between its warheads')
        graph['protac_torsion_mask'] = torsion_mask

    def _checked_read(self, outputs, ligand, receptor, branch):
        tr, rot, tor, scalars, vectors = outputs
        count = ligand['coords'].shape[1] + receptor['coords'].shape[1]
        torsions = (0 if getattr(self.expert.model, 'no_torsion', False)
                    else int(ligand['edge_mask'].sum()))
        if (tr.shape != (1, 3) or rot.shape != (1, 3)
                or tor.shape not in ((torsions,), (1, torsions))
                or scalars.shape != (count, self.expert.node_scalar_dim)
                or vectors.shape != (count, self.expert.node_vector_dim, 3)):
            raise ValueError(f'{branch}: frozen evidence shape/rotor contract differs')
        if not all(torch.isfinite(value).all() for value in outputs):
            raise FloatingPointError(f'{branch}: non-finite frozen evidence')
        return tr, rot, tor.reshape(1, -1), scalars, vectors

    def forward(self, e3_data, poi_data, ligand_data, t_e3l, t_pol=None,
                r_e3_to_poi=None, receptor_crop_cutoff=None, diffusion_state=None,
                source=0, *, conditions=None, task_mode=None):
        anchor_contract = self.trifuse.config.get('protac_anchor_contract')
        if conditions is not None and anchor_contract is None:
            if not isinstance(conditions, dict) or conditions.get('contract') != 'external_binary_anchor_roles_v1':
                raise ValueError('unsupported external anchor condition')
        if anchor_contract is not None:
            if not isinstance(conditions, dict) or conditions.get('contract') not in (
                    'external_binary_exact_bond_anchor_v1',
                    'synthetic_independent_binary_frames_v1'):
                raise ValueError('PROTAC model requires an explicit binary-anchor condition')
        if t_e3l.shape != (1,):
            raise ValueError('model-v2 currently requires B=1')
        if t_pol is not None and not torch.equal(t_e3l, t_pol):
            raise ValueError('ternary assembly requires one shared flow clock')
        if diffusion_state is None or diffusion_state.shape != (1, 9):
            raise ValueError('model-v2 requires explicit diffusion_state [1,9]')
        for name, data in (('E3', e3_data), ('POI', poi_data), ('ligand', ligand_data)):
            coords, features = data['coords'], data['features']
            if (coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[2] != 3
                    or coords.shape[1] == 0 or features.ndim != 3
                    or features.shape[:2] != coords.shape[:2]):
                raise ValueError(f'{name}: assembly requires one node-aligned example, not a padded batch')
        if (not torch.isfinite(t_e3l).all() or not torch.isfinite(diffusion_state).all()
                or not torch.equal(t_e3l, diffusion_state[:, -1])):
            raise ValueError('binary expert and ternary controller must share one finite clock')
        dt = diffusion_state[:, 5:8]
        if (bool(((t_e3l < 0) | (t_e3l > 1)).any())
                or not torch.equal(dt, dt[:, :1].expand_as(dt))
                or bool(((dt < 0) | (dt > t_e3l[:, None])).any())
                or bool(((t_e3l > 0) & (dt[:, 0] <= 0)).any())):
            raise ValueError('all motion DOFs require the same valid remaining-time step')
        if self.expert is None:
            if self.trifuse.config.get('candidate_driven') and r_e3_to_poi is None:
                raise ValueError('candidate model requires explicit POI body rotation')
            graph = build_trifuse_graph(
                e3_data['coords'][0], e3_data['features'][0],
                poi_data['coords'][0], poi_data['features'][0],
                ligand_data['coords'][0], ligand_data['features'][0],
                lig_edge_index=ligand_data['edge_index'],
                lig_edge_mask=ligand_data['edge_mask'],
                lig_mask_rotate=ligand_data['mask_rotate'],
                lig_edge_attr=ligand_data['edge_attr'])
            if conditions is not None:
                graph['supplied_anchor_roles'] = conditions['roles']
            if self.trifuse.config.get('candidate_driven'):
                coords, types = graph['node_coords'], graph['node_types']
                frame = (r_e3_to_poi[0] if r_e3_to_poi is not None
                         else torch.eye(3, device=coords.device, dtype=coords.dtype))
                physical, vectors = candidate_fullatom_evidence(
                    coords, types, e3_data, poi_data, ligand_data, frame)
                reference_scalar, reference_vectors = candidate_reference_evidence(
                    coords, types, e3_data, poi_data, ligand_data,
                    contract=self.trifuse.config['candidate_reference_contract'])
                graph['fullatom_scalar_features'] = physical
                graph['fullatom_vector_features'] = vectors
                graph['candidate_reference_scalar'] = reference_scalar
                graph['candidate_reference_vectors'] = reference_vectors
            if self.trifuse.config.get('protein_local_geometry_contract', 'none') != 'none':
                required = self.trifuse.config['protein_local_geometry_contract'] == 'intrinsic32_required_v1'
                graph['protein_local_geometry'] = torch.cat((
                    protein_local_geometry_features(e3_data, required=required),
                    protein_local_geometry_features(poi_data, required=required),
                    ligand_data['coords'].new_zeros(ligand_data['coords'].shape[1], 33)), 0)
            if (self.training and torch.is_grad_enabled()
                    and self._should_checkpoint_ternary(len(graph['node_coords']), graph['node_coords'].device)):
                return checkpoint(self.trifuse, graph, {}, diffusion_state,
                                  source=source,
                                  task_mode=task_mode if task_mode is not None else poi_data.get('task_mode'),
                                  use_reentrant=False)
            return self.trifuse(graph, {}, diffusion_state, source=source,
                                task_mode=task_mode if task_mode is not None else poi_data.get('task_mode'),
                                conditions=conditions)
        if self.trifuse.config.get('candidate_driven') and r_e3_to_poi is None:
            raise ValueError('candidate model requires explicit POI body rotation')
        frame = (r_e3_to_poi[0] if r_e3_to_poi is not None
                 else torch.eye(3, device=t_e3l.device, dtype=ligand_data['coords'].dtype))
        fullatom_geometry = None
        reference_geometry = None
        if self.trifuse.config.get('candidate_driven'):
            parts = (e3_data, poi_data, ligand_data)
            coords = torch.cat([part['coords'][0] for part in parts])
            types = torch.cat([torch.full((part['coords'].shape[1],), index,
                device=coords.device, dtype=torch.long) for index, part in enumerate(parts)])
            fullatom_geometry = candidate_fullatom_evidence(
                coords, types, e3_data, poi_data, ligand_data, frame)
            reference_geometry = candidate_reference_evidence(coords, types, e3_data, poi_data, ligand_data)
            sizes = [part['coords'].shape[1] for part in parts]
            physical = torch.cat((fullatom_geometry[0], reference_geometry[0]), -1).split(sizes)
            e3_data = {**e3_data, 'candidate_fullatom_features': physical[0]}
            poi_data = {**poi_data, 'candidate_fullatom_features': physical[1]}
        raw, node_reads = {}, []
        # Private graph tensors ensure even in-place expert internals cannot
        # mutate the shared state or the other branch's input.
        with self._expert_context():
            for index, (branch, receptor) in enumerate(zip(BRANCHES, (e3_data, poi_data))):
                if getattr(self.expert, 'backend', 'diffdock') == 'matcha':
                    opposite = poi_data if index == 0 else e3_data
                    if self.trifuse.config.get('candidate_driven'):
                        basis = receptor['matcha'].get('reference_basis')
                        if basis is None:
                            raise ValueError('candidate model requires rigid-transport Matcha metadata')
                        receptor = {**receptor, 'matcha': {**receptor['matcha'],
                            'current_basis': frame @ basis if index else basis}}
                    expert_ligand = ligand_data
                    if not self.expert_input_grad:
                        receptor = {**receptor, 'coords': receptor['coords'].detach()}
                        if 'candidate_fullatom_features' in receptor:
                            receptor['candidate_fullatom_features'] = receptor['candidate_fullatom_features'].detach()
                        opposite = {**opposite, 'coords': opposite['coords'].detach()}
                        if 'candidate_fullatom_features' in opposite:
                            opposite['candidate_fullatom_features'] = opposite['candidate_fullatom_features'].detach()
                        expert_ligand = {**ligand_data, 'coords': ligand_data['coords'].detach()}
                    stage_index, _ = matcha_stage_clock(t_e3l)
                    modulation = self.trifuse.binary_condition(receptor, opposite, expert_ligand,
                        stage_index=stage_index, diffusion_state=diffusion_state, source=source, task_mode=task_mode)
                    def read_pair(rec, lig, time, film):
                        return self.expert.read_pair(rec, lig, time, film)
                    if torch.is_grad_enabled() and (self.expert_input_grad
                                                    or any(value.requires_grad for value in modulation)):
                        outputs = checkpoint(read_pair, receptor, expert_ligand, t_e3l,
                                             modulation, use_reentrant=False)
                    else:
                        outputs = read_pair(receptor, expert_ligand, t_e3l, modulation)
                    tr, rot, tor, scalars, vectors = self._checked_read(
                        outputs, expert_ligand, receptor, branch)
                    n_lig = ligand_data['coords'].shape[1]
                    node_reads.append((scalars[:n_lig], scalars[n_lig:],
                                       vectors[:n_lig], vectors[n_lig:]))
                    raw[branch] = {'tr': tr, 'rot': rot, 'tor': tor}
                    continue
                center = receptor['coords'][0].mean(0)
                rotation = frame if index else torch.eye(3, device=frame.device, dtype=frame.dtype)
                rec_pos = (receptor['coords'][0] - center) @ rotation
                lig_pos = (ligand_data['coords'][0] - center) @ rotation
                crop = crop_receptor_to_ligand(
                    rec_pos, receptor['features'][0], lig_pos,
                    rec_edge_index=receptor.get('edge_index'), cutoff=None,
                    min_residues=1, min_residues_with_edges=8, require_internal_edges=True)
                graph = build_binary_graph(
                    crop['coords'], crop['features'], lig_pos, ligand_data['features'][0],
                    rec_edge_index=crop['edge_index'], lig_edge_index=ligand_data['edge_index'],
                    lig_edge_attr=ligand_data['edge_attr'], lig_edge_mask=ligand_data['edge_mask'],
                    lig_mask_rotate=ligand_data['mask_rotate'])
                _set_time_embeddings(graph, t_e3l, self.expert.model)
                outputs = self._read_expert(graph)
                tr, rot, tor, scalars, vectors = self._checked_read(
                    outputs, ligand_data, receptor, branch)
                n_lig = len(lig_pos)
                node_reads.append((scalars[:n_lig], scalars[n_lig:],
                                   vectors[:n_lig] @ rotation.T, vectors[n_lig:] @ rotation.T))
                raw[branch] = {'tr': tr @ rotation.T, 'rot': rot @ rotation.T,
                               'tor': tor.reshape(1, -1)}
            if getattr(self.expert, 'backend', 'diffdock') == 'matcha':
                proposals = {**raw, 'torsion': raw[BRANCHES[0]]['tor']}
            else:
                proposals = self._conditioning.normalized_predictions(
                    {'interface_scores': raw, 'torsion_score': raw[BRANCHES[0]]['tor']}, t_e3l)
        graph = build_trifuse_graph(
            e3_data['coords'][0], e3_data['features'][0],
            poi_data['coords'][0], poi_data['features'][0],
            ligand_data['coords'][0], ligand_data['features'][0],
            lig_edge_index=ligand_data['edge_index'], lig_edge_mask=ligand_data['edge_mask'],
            lig_mask_rotate=ligand_data['mask_rotate'], lig_edge_attr=ligand_data['edge_attr'])
        if anchor_contract is not None:
            self._add_protac_anchor_context(graph, conditions, ligand_data)
        if self.trifuse.config.get('candidate_driven'):
            scalar_geometry, vector_geometry = fullatom_geometry
            graph['fullatom_scalar_features'] = scalar_geometry
            graph['fullatom_vector_features'] = vector_geometry
            graph['candidate_reference_scalar'], graph['candidate_reference_vectors'] = reference_geometry
        if self.trifuse.config.get('protein_local_geometry_contract', 'none') != 'none':
            required = self.trifuse.config['protein_local_geometry_contract'] == 'intrinsic32_required_v1'
            graph['protein_local_geometry'] = torch.cat((
                protein_local_geometry_features(e3_data, required=required),
                protein_local_geometry_features(poi_data, required=required),
                ligand_data['coords'].new_zeros(ligand_data['coords'].shape[1], 33)), 0)
        a, b = node_reads
        s, v = self.expert.node_scalar_dim, self.expert.node_vector_dim
        # Fixed slots distinguish expert identity and explicit availability.
        e3 = torch.cat((a[1], a[1].new_zeros(len(a[1]), s),
                        a[1].new_ones(len(a[1]), 1), a[1].new_zeros(len(a[1]), 1)), -1)
        poi = torch.cat((b[1].new_zeros(len(b[1]), s), b[1],
                         b[1].new_zeros(len(b[1]), 1), b[1].new_ones(len(b[1]), 1)), -1)
        ligand = torch.cat((a[0], b[0], a[0].new_ones(len(a[0]), 2)), -1)
        graph['expert_node_features'] = torch.cat((e3, poi, ligand), 0)
        graph['expert_node_vectors'] = torch.cat((
            torch.cat((a[3], a[3].new_zeros(len(a[3]), v, 3)), 1),
            torch.cat((b[3].new_zeros(len(b[3]), v, 3), b[3]), 1),
            torch.cat((a[2], b[2]), 1)), 0)
        if (self.training and torch.is_grad_enabled()
                and self._should_checkpoint_ternary(len(graph["node_coords"]), graph["node_coords"].device)):
            result = checkpoint(self.trifuse, graph, proposals, diffusion_state,
                                source=source, conditions=conditions,
                                task_mode=task_mode, use_reentrant=False)
        else:
            result = self.trifuse(graph, proposals, diffusion_state, source=source,
                                  conditions=conditions, task_mode=task_mode)
        # Kept explicitly as diagnostic evidence, never consumed as another update.
        result['binary_proposals'] = proposals
        return result
