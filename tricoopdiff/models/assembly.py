"""Frozen binary priors and learned ternary assembly constraints.

Production v14 exchanges nodewise scalar/vector evidence using typed
attention, then predicts legal rigid pose and bond-conditioned torsion increments.
All vectors use the E3 frame; native targets never enter the model.
"""

from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tricoopdiff.inputs.graphs import build_binary_graph, build_trifuse_graph, crop_receptor_to_ligand
from tricoopdiff.models.expert import _set_time_embeddings
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.diffdock.constants import lig_feature_dims


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


ARCHITECTURE = 'residual_evidence_v3'
DIRECT_ARCHITECTURE = 'ternary_topological_joint_pose_v14'
PREVIOUS_DIRECT_ARCHITECTURE = 'ternary_nodefield_independent_v4'
ASSEMBLY_CHECKPOINT_VERSION = 50
EXPERT_CROP_POLICY = 'full_receptor_cross_taper_v1'
LEGACY_RANK_FEATURE_STATE = 'final_pose_t0_dt0_v1'
DIRECT_RANK_FEATURE_STATE = 'final_pose_evidence_t0_dt0_v2'
INTERFACE_OBJECTIVE_CHECKPOINT_VERSION = 22
OBJECTIVE_CONTRACT_CHECKPOINT_VERSION = 21
ENDPOINT_PROTOCOL_CHECKPOINT_VERSION = 20
LEGACY_CONSENSUS_CHECKPOINT_VERSION = 18
PREVIOUS_CONSENSUS_CHECKPOINT_VERSION = 19
SOURCE_NAMES = ('sdpi', 'bitseed')
SOURCE_CONTRACT_VERSION = 'candidate_source_conditioning_v1'
LEGACY_PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v1'
FEATURE_PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v2'
PREVIOUS_PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v3'
PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v4'
VECTOR_EXPERT_MODEL_PROFILE = 'ternary_endpoint_v14'
MATCHA_CONDITIONING_CONTRACT = 'nodewise_distance_values_v2'
INFORMATION_CONTRACT = 'geometric_message_values_v1'
MOTION_CONTRACT = 'node_evidence_direct_velocity_without_time_gain_v1'


def production_model_config(node_dim, input_dim, source_conditioning=False, *,
                            model_profile=VECTOR_EXPERT_MODEL_PROFILE,
                            expert_scalar_dim=58, expert_vector_dim=20,
                            expert_backend='diffdock', conditioning_contract=MATCHA_CONDITIONING_CONTRACT,
                            information_contract=INFORMATION_CONTRACT, motion_contract=MOTION_CONTRACT):
    """One explicit new-model recipe; older weights retain their own config."""
    if model_profile not in (LEGACY_PRODUCTION_MODEL_PROFILE, FEATURE_PRODUCTION_MODEL_PROFILE,
                            PREVIOUS_PRODUCTION_MODEL_PROFILE, PRODUCTION_MODEL_PROFILE,
                            VECTOR_EXPERT_MODEL_PROFILE):
        raise ValueError(f'unknown model profile: {model_profile}')
    if information_contract != INFORMATION_CONTRACT:
        raise ValueError('retired geometric attention: geometry must enter message values')
    if motion_contract != MOTION_CONTRACT:
        raise ValueError('retired broadcast/time-amplified motion contract')
    config = dict(node_dim=node_dim, input_dim=input_dim, information_contract=information_contract,
                source_conditioning=bool(source_conditioning),
                direct_ternary_motion=True, global_node_context=True,
                spatial_geometry=True, motion_attention=True,
                global_node_attention=True, rotation_scale_normalization=True,
                model_profile=model_profile)
    if model_profile in (FEATURE_PRODUCTION_MODEL_PROFILE, PREVIOUS_PRODUCTION_MODEL_PROFILE,
                         PRODUCTION_MODEL_PROFILE, VECTOR_EXPERT_MODEL_PROFILE):
        if isinstance(expert_scalar_dim, bool) or not isinstance(expert_scalar_dim, int) or expert_scalar_dim <= 0:
            raise ValueError('expert scalar dimension must be positive')
        config['expert_scalar_dim'] = expert_scalar_dim
    if model_profile == VECTOR_EXPERT_MODEL_PROFILE:
        config['motion_contract'] = motion_contract
        if isinstance(expert_vector_dim, bool) or not isinstance(expert_vector_dim, int) or expert_vector_dim <= 0:
            raise ValueError('expert vector dimension must be a positive integer')
        config['expert_vector_dim'] = expert_vector_dim
        for obsolete in ('global_node_context', 'spatial_geometry', 'motion_attention',
                         'global_node_attention', 'rotation_scale_normalization'):
            config.pop(obsolete)
    if expert_backend not in ('diffdock', 'matcha', 'fabind'):
        raise ValueError('unknown binary expert backend')
    if expert_backend in ('matcha', 'fabind'):
        if conditioning_contract != MATCHA_CONDITIONING_CONTRACT:
            raise ValueError('retired Matcha conditioning: distance must enter nodewise values')
        config['expert_backend'] = expert_backend
        config['conditioning_contract'] = conditioning_contract
    return config


def candidate_source_tensor(source, reference):
    """Normalize one immutable trajectory source to a ``[1]`` class tensor."""
    if isinstance(source, str):
        if source not in SOURCE_NAMES:
            raise ValueError(f'unknown candidate source: {source!r}')
        source = SOURCE_NAMES.index(source)
    value = torch.as_tensor(source, device=reference.device).reshape(-1)
    if (value.shape != (1,) or value.dtype == torch.bool or value.is_complex()
            or not bool(((value == 0) | (value == 1)).all())):
        raise ValueError('candidate source must be one SDPI/BiTSeed class')
    return value.long()


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

    @staticmethod
    def chemical_relations(coords, edges, attributes):
        """Exact unweighted graph distance plus direct-edge chemistry."""
        count = len(coords)
        paths = coords.new_full((count, count), float('inf'))
        paths.diagonal().zero_()
        paths[edges[0], edges[1]] = 1.
        # Chemical connectivity is undirected even if the input edge list is
        # directed. Values remain in their recorded directed slots.
        paths[edges[1], edges[0]] = 1.
        for middle in range(count):
            paths = torch.minimum(paths, paths[:, middle:middle + 1] + paths[middle:middle + 1])
        connected = torch.isfinite(paths)
        features = coords.new_zeros(count, count, 6)
        features[..., 0] = torch.log1p(paths.masked_fill(~connected, 0.))
        features[edges[0], edges[1], 1:5] = attributes
        lengths = torch.linalg.vector_norm(coords[edges[0]] - coords[edges[1]], dim=-1) / 10.
        features[edges[0], edges[1], 5] = torch.log1p(lengths)
        return features, connected

    def forward(self, identity, coords, types, edges, attributes):
        encoded = torch.zeros_like(identity)
        for component in range(3):
            ids = (types == component).nonzero(as_tuple=False).flatten()
            h, xyz = identity[ids], coords[ids]
            keys, values = self.key(h), self.value(h)
            if component == 2:
                local = torch.empty(len(identity), device=ids.device, dtype=torch.long)
                local[ids] = torch.arange(len(ids), device=ids.device)
                chemistry, connected = self.chemical_relations(xyz, local[edges], attributes)
            pieces = []
            for start in range(0, len(ids), 64):
                stop = min(start + 64, len(ids))
                logits = self.query(h[start:stop]) @ keys.T / math.sqrt(self.width)
                if component == 2:
                    relation = self.chemistry(chemistry[start:stop])
                    logits = (logits + relation[..., -1]).masked_fill(~connected[start:stop], -torch.inf)
                    messages = values[None] + relation[..., :-1]
                else:
                    distance = torch.linalg.vector_norm(xyz[start:stop, None] - xyz[None], dim=-1) / 10.
                    relation = self.geometry(torch.log1p(distance)[..., None])
                    logits = logits + relation[..., -1]
                    messages = values[None] + relation[..., :-1]
                pieces.append((logits.softmax(-1)[..., None] * messages).sum(1))
            encoded[ids] = self.norm(h + torch.cat(pieces))
        return encoded


class EquivariantInteraction(nn.Module):
    """Dense, source-balanced SO(3) attention without a pooled control token."""

    def __init__(self, width, channels):
        super().__init__()
        self.width = width
        self.heads = 4 if width % 4 == 0 and channels % 4 == 0 else 1
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
                  coords, types, bonds):
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
            selected = types == component
            weights = logits[:, selected].softmax(1)
            messages = (values[selected].reshape(1, -1, self.heads, self.width // self.heads)
                        + pair_values[:, selected])
            scalar_messages.append(torch.einsum('ijh,ijhd->ihd', weights, messages).flatten(1))
            direct = torch.einsum('ijh,jhvc->ihvc', weights,
                vectors[selected].reshape(-1, self.heads, vectors.shape[1] // self.heads, 3))
            radial = torch.einsum('ijh,jhv,ijc->ihvc', weights,
                self.direction(values[selected]).reshape(-1, self.heads, vectors.shape[1] // self.heads),
                direction[:, selected])
            handed = torch.cross(vectors[None, selected],
                                 direction[:, selected, None].expand(-1, -1, vectors.shape[1], -1), dim=-1)
            handed = handed.reshape(len(q), -1, self.heads, vectors.shape[1] // self.heads, 3)
            transport = direct + radial + torch.einsum('ijh,ijhvc->ihvc', weights, handed)
            vector_messages.append(transport.flatten(1, 2))
            densities.append(torch.log1p(torch.exp(-distance[:, selected, 0].square()).sum(1, keepdim=True)))
        return torch.cat((*scalar_messages, *densities), -1), torch.cat(vector_messages, 1)

    def forward(self, h, vectors, coords, types, edges, attributes):
        keys, values = self.key(h), self.value(h)
        scalar, vector = [], []
        for start in range(0, len(h), 64):
            end = min(start + 64, len(h))
            bonds = h.new_zeros(end - start, len(h), 4)
            selected = (edges[0] >= start) & (edges[0] < end)
            bonds[edges[0, selected] - start, edges[1, selected]] = attributes[selected]
            args = (self.query(h[start:end]), vectors[start:end], coords[start:end],
                    keys, values, vectors, coords, types, bonds)
            if self.training and torch.is_grad_enabled():
                messages = checkpoint(self._messages, *args, use_reentrant=False)
            else:
                messages = self._messages(*args)
            scalar.append(messages[0])
            vector.append(messages[1])
        vectors = vectors + self.vector_mix(torch.cat(vector).transpose(1, 2)).transpose(1, 2)
        # A smooth invariant rescaling bounds repeated vector transport.
        vectors = vectors / (1. + vectors.square().mean((1, 2), keepdim=True)).sqrt()
        norms = vectors.square().sum(-1).clamp_min(1e-12).sqrt()
        h = self.norm(h + self.scalar_mix(torch.cat((h, torch.cat(scalar), norms), -1)))
        return h, vectors


class LegalPoseDecoder(nn.Module):
    """Jointly decode physical body/bond DOFs, never free atom displacements."""

    def __init__(self, width, channels):
        super().__init__()
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
        self.joint_relation = nn.Linear(3, self.joint.num_heads, bias=False)
        self.norm = nn.LayerNorm(width)
        self.rigid = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                   nn.Linear(width, 2 * channels))
        self.bond = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        for head in (self.rigid, self.bond):
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
    def action_geometry(coords, vectors, ligand, bonds, masks):
        """Invariant bond/atom geometry; tangent is the raw legal rotor action.

        No inverse solve or target enters here. The executor retains its
        existing horizontal gauge; this is not its complete Jacobian.
        """
        xyz = coords[ligand]
        u, v = bonds
        axis = xyz[u] - xyz[v]
        length = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        if bool((length <= 1e-10).any()):
            raise ValueError('degenerate torsion axis')
        axis = axis / length
        lever = (coords[None] - xyz[v, None]) / 10.
        axial = (lever * axis[:, None]).sum(-1, keepdim=True)
        radial = lever - axial * axis[:, None]
        affected = coords.new_zeros(len(u), len(coords))
        affected[:, ligand] = masks.to(coords.dtype)
        tangent = torch.cross(axis[:, None].expand_as(lever), lever, dim=-1) * affected[..., None]
        return torch.cat((axial, torch.linalg.vector_norm(radial, dim=-1, keepdim=True),
                          affected[..., None], (vectors[None] * axis[:, None, None]).sum(-1),
                          (vectors[None] * tangent[:, :, None]).sum(-1)), -1)

    def forward(self, h, vectors, coords, types, bonds, masks, state, opinions):
        time = state[:, -1:]
        if bool(((time < 0) | (time > 1)).any()):
            raise ValueError('remaining flow time must be in [0, 1]')
        # Preserve dt=t semantics even below the historical 20-step grid.
        # A t=0 read is diagnostic only: keep finite outputs without pretending
        # that it has a positive remaining integration budget.
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
            bond_queries = self.bond_query(features)
            midpoint = (xyz[u] + xyz[v]) / 2.
            geometry = self.bond_geometry(self.action_geometry(coords, vectors, ligand, bonds, masks))
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
            bond_queries = bond_queries + self.fragment(torch.cat(fragments, -1))
            queries.extend(bond_queries.unbind(0))
            positions.extend(midpoint.unbind(0))
        queries, positions = torch.stack(queries), torch.stack(positions)
        affected = torch.stack([(types == component).to(h.dtype) for component in components])
        if bonds.shape[1]:
            bond_affected = h.new_zeros(len(masks), len(h))
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
        relation = self.joint_relation(self.joint_relations(positions, affected)).permute(2, 0, 1)
        coupled, _ = self.joint(decisions[None], decisions[None], decisions[None],
                                attn_mask=relation, need_weights=False)
        decisions = self.norm(decisions + coupled[0])
        motion = {}
        for index, (branch, component, column) in enumerate(zip(BRANCHES, components, (0, 3))):
            selected = types == component
            local = h[selected]
            coefficients = self.rigid(torch.cat((
                local, decisions[index:index + 1].expand(len(local), -1)), -1))
            coefficients = coefficients.reshape(-1, 2, vectors.shape[1])
            weights = (local @ decisions[index] / math.sqrt(self.width)).softmax(0)
            displacement, rotation = (
                weights[:, None, None] * (coefficients[..., None]
                                         * vectors[selected, None]).sum(2)).sum(0)
            shift = displacement[None] * 10.
            angular = math.pi * bounded_vector(rotation[None])
            motion[branch] = {
                'tr': -shift / state[:, column:column + 1].exp(),
                'rot': -angular / math.pi}
        torsion = (-self.bond(decisions[2:]).squeeze(-1).tanh()[None]
                   if bonds.shape[1] else h.new_empty(1, 0))
        for branch in BRANCHES:
            motion[branch]['tor'] = torsion
        motion.update(torsion=torsion, rigid_output_mode='independent_centroids_v1',
                      torsion_update_mode='continuous_horizontal_v1')
        return motion



class UnifiedTernaryController(nn.Module):
    """Nodewise scalar/vector interaction and bond-conditioned action readout.

    No native geometry enters this module. All three components remain explicit
    through every interaction layer; only the final action readout aggregates.
    """

    def __init__(self, input_dim, width, scalar_dim, vector_dim, source_conditioning):
        super().__init__()
        channels = max(4, width // 4)
        self.scalar_dim, self.vector_dim = scalar_dim, vector_dim
        self.protein = nn.Linear(input_dim, width)
        self.ligand = LigandFeatureEncoder(width)
        self.expert = nn.Linear(2 * scalar_dim + 2, width)
        self.types = nn.Embedding(3, width)
        self.clock = nn.Linear(9, width)
        self.structure = ComponentStructureEncoder(width)
        self.structure_read = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(3)])
        self.source = None
        if source_conditioning:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(0)
                self.source = nn.Embedding(len(SOURCE_NAMES), width)
        self.vector_input = nn.Linear(2 * vector_dim + 3, channels, bias=False)
        self.layers = nn.ModuleList([EquivariantInteraction(width, channels) for _ in range(3)])
        self.assembly = LegalPoseDecoder(width, channels)

    def forward(self, graph, proposals, state, source):
        coords, types = graph['node_coords'], graph['node_types']
        scalar, raw_vectors = graph.get('expert_node_features'), graph.get('expert_node_vectors')
        if scalar is None or scalar.shape != (len(coords), 2 * self.scalar_dim + 2):
            raise ValueError('assembly requires node-aligned binary scalar evidence')
        if raw_vectors is None or raw_vectors.shape != (len(coords), 2 * self.vector_dim, 3):
            raise ValueError('assembly requires node-aligned binary vector evidence')
        if state.shape != (1, 9) or not all((types == i).any() for i in range(3)):
            raise ValueError('assembly requires one clock and three nonempty components')
        if not all(torch.isfinite(x).all() for x in (coords, scalar, raw_vectors, state)):
            raise ValueError('assembly inputs must be finite')
        h = self.protein(graph['node_features'])
        ligand = types == 2
        h = h.clone()
        h[ligand] = self.ligand(graph['node_features'][ligand])
        identity = h + self.types(types)
        h = identity + self.expert(scalar) + self.clock(state)
        if self.source is not None:
            h = h + self.source(candidate_source_tensor(source, h))
        centers = torch.stack([coords[types == i].mean(0) for i in range(3)])
        offsets = bounded_vector((coords[:, None] - centers[None]) / 10.)
        vectors = torch.cat((bounded_vector(raw_vectors), offsets), 1)
        vectors = self.vector_input(vectors.transpose(1, 2)).transpose(1, 2)
        ligand_ids = ligand.nonzero(as_tuple=False).flatten()
        local_edges = graph['lig_edge_index']
        edges = ligand_ids[local_edges]
        attributes = graph.get('lig_edge_attr')
        if attributes is None:
            attributes = h.new_zeros(local_edges.shape[1], 4)
        if attributes.shape != (local_edges.shape[1], 4):
            raise ValueError('assembly requires four chemical bond attributes per edge')
        memory = self.structure(identity, coords, types, edges, attributes)
        for layer, read in zip(self.layers, self.structure_read):
            h = read(torch.cat((h, memory), -1))
            h, vectors = layer(h, vectors, coords, types, edges, attributes)
        bonds = local_edges[:, graph['lig_edge_mask'].bool()]
        opinions = [proposals[branch]['tor'].flatten() for branch in BRANCHES]
        if all(len(x) == 0 for x in opinions):
            bonds = bonds[:, :0]
            masks = torch.zeros(0, len(ligand_ids), device=coords.device, dtype=torch.bool)
        else:
            if any(len(x) != bonds.shape[1] for x in opinions):
                raise ValueError('assembly expert torsions must match canonical rotating bonds')
            masks = torch.as_tensor(graph['lig_mask_rotate'], device=coords.device, dtype=torch.bool)
        if masks.shape != (bonds.shape[1], len(ligand_ids)):
            raise ValueError('assembly torsion masks must match directed rotating bonds')
        body_motion = self.assembly(h, vectors, coords, types, bonds, masks, state, opinions)
        return {'normalized_velocity': body_motion,
                'rank_features': torch.stack([h[types == i].mean(0) for i in range(3)]).mean(0, keepdim=True),
                'residual_penalty': h.new_zeros(())}


class TernaryAssembly(nn.Module):
    """The sole production controller; retired motion architectures are rejected."""

    supports_conditions = False
    direct_ternary_motion = True

    def migrate_logit_only_geometry(self, weights):
        """Explicit weights-only upgrade; zero new values, preserve old logits.

        Optimizer state must not be restored across this parameter-shape change.
        No unrelated missing keys or shape mismatches are accepted.
        """
        weights = dict(weights)
        template = self.state_dict()
        heads = [('controller.structure.geometry.2', 1),
                 ('controller.assembly.distance.2', 1)]
        heads.extend((f'controller.layers.{i}.geometry.2', layer.heads)
                     for i, layer in enumerate(self.controller.layers))
        for prefix, count in heads:
            for suffix in ('weight', 'bias'):
                key = f'{prefix}.{suffix}'
                old, new = weights[key], torch.zeros_like(template[key])
                if old.shape != (count, *new.shape[1:]):
                    raise ValueError(f'{key}: not a legacy logit-only geometric head')
                new[-count:] = old.to(new)
                weights[key] = new
        return weights

    def __init__(self, node_dim=128, input_dim=None, source_conditioning=False, *,
                 direct_ternary_motion=True, model_profile=VECTOR_EXPERT_MODEL_PROFILE,
                 expert_scalar_dim=58, expert_vector_dim=20, expert_backend='diffdock',
                 conditioning_contract=MATCHA_CONDITIONING_CONTRACT,
                 information_contract=INFORMATION_CONTRACT, motion_contract=MOTION_CONTRACT):
        super().__init__()
        if model_profile != VECTOR_EXPERT_MODEL_PROFILE or not direct_ternary_motion:
            raise ValueError('retired architecture: construct a fresh ternary_endpoint_v14 model')
        self.config = production_model_config(
            node_dim, input_dim or node_dim, source_conditioning,
            model_profile=model_profile, expert_scalar_dim=expert_scalar_dim,
            expert_vector_dim=expert_vector_dim, expert_backend=expert_backend,
            conditioning_contract=conditioning_contract, information_contract=information_contract,
            motion_contract=motion_contract)
        self.controller = UnifiedTernaryController(
            self.config['input_dim'], node_dim, expert_scalar_dim,
            expert_vector_dim, source_conditioning)
        if expert_backend in ('matcha', 'fabind'):
            self.condition_query = nn.Linear(self.config['input_dim'], node_dim)
            self.condition_key = nn.Linear(self.config['input_dim'], node_dim)
            self.condition_value = nn.Linear(self.config['input_dim'], node_dim)
            self.condition_geometry = nn.Linear(3, node_dim, bias=False)
            nn.init.zeros_(self.condition_geometry.weight)
            self.binary_film = nn.Linear(node_dim, 2 * expert_scalar_dim)
            nn.init.zeros_(self.binary_film.weight)
            nn.init.zeros_(self.binary_film.bias)

    def binary_condition(self, receptor, opposite):
        """Nodewise opposite-protein context; no target or pooled condition token."""
        q = self.condition_query(receptor['features'][0])
        k = self.condition_key(opposite['features'][0])
        value = self.condition_value(opposite['features'][0])
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
        context = weights @ value + self.condition_geometry(geometry)
        gamma, beta = self.binary_film(context).chunk(2, -1)
        return .1 * gamma.tanh().unsqueeze(0), .1 * beta.tanh().unsqueeze(0)

    def forward(self, graph, proposals, diffusion_state, source=0, *, conditions=None):
        if conditions is not None:
            raise ValueError('retired condition-token interface: v8 reads actual ternary nodes')
        return self.controller(graph, proposals, diffusion_state, source)


class TernaryAssemblyModel(nn.Module):
    """Frozen binary evidence followed by an independent ternary motion field."""

    architecture = DIRECT_ARCHITECTURE

    def __init__(self, frozen_expert, trifuse, *,
                 expert_input_grad=True, checkpoint_ternary=False):
        super().__init__()
        if not isinstance(trifuse, TernaryAssembly) or trifuse.config.get('model_profile') != VECTOR_EXPERT_MODEL_PROFILE:
            raise ValueError('production requires the unified ternary_endpoint_v14 controller')
        if any('lora' in name for name, _ in frozen_expert.named_parameters()):
            raise ValueError('model-v2 Stage 1 requires an unmodified frozen expert')
        self.expert = frozen_expert.requires_grad_(False).eval()
        self.trifuse = trifuse
        backend = getattr(frozen_expert, 'backend', 'diffdock')
        if backend != trifuse.config.get('expert_backend', 'diffdock'):
            raise ValueError('expert backend and controller configuration disagree')
        reader = 'read_pair' if backend in ('matcha', 'fabind') else 'forward_features'
        if (not callable(getattr(frozen_expert, reader, None))
                or frozen_expert.node_scalar_dim != trifuse.config['expert_scalar_dim']
                or frozen_expert.node_vector_dim != trifuse.config['expert_vector_dim']):
            raise ValueError('assembly requires matching frozen expert scalar/vector interfaces')
        # Keep pretrained internal edges; only cross edges receive tapering.
        if getattr(frozen_expert, 'backend', 'diffdock') == 'diffdock':
            frozen_expert.model.smooth_cross_edges = True
        self.expert_input_grad = bool(expert_input_grad)
        self.checkpoint_ternary = bool(checkpoint_ternary)
        self._conditioning = SDPIFlow()

    def _expert_context(self):
        # Frozen weights do not require detached input Jacobians. Respect the
        # caller's no_grad during inference; never force enable_grad here.
        # Matcha FiLM belongs to the trainable controller, not the frozen prior.
        # Its gradient must survive even when expert coordinate VJPs are disabled.
        if self.expert_input_grad or getattr(self.expert, 'backend', None) in ('matcha', 'fabind'):
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
                source=0, *, conditions=None):
        if conditions is not None:
            raise ValueError('assembly reads actual ternary nodes, not retired condition tokens')
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
        frame = (r_e3_to_poi[0] if r_e3_to_poi is not None
                 else torch.eye(3, device=t_e3l.device, dtype=ligand_data['coords'].dtype))
        raw, node_reads = {}, []
        # Private graph tensors ensure even in-place expert internals cannot
        # mutate the shared state or the other branch's input.
        with self._expert_context():
            for index, (branch, receptor) in enumerate(zip(BRANCHES, (e3_data, poi_data))):
                if getattr(self.expert, 'backend', 'diffdock') in ('matcha', 'fabind'):
                    opposite = poi_data if index == 0 else e3_data
                    expert_ligand = ligand_data
                    if not self.expert_input_grad:
                        receptor = {**receptor, 'coords': receptor['coords'].detach()}
                        opposite = {**opposite, 'coords': opposite['coords'].detach()}
                        expert_ligand = {**ligand_data, 'coords': ligand_data['coords'].detach()}
                    modulation = self.trifuse.binary_condition(receptor, opposite)
                    def read_pair(rec, lig, time, film):
                        return self.expert.read_pair(rec, lig, time, film)
                    if (getattr(self.expert, 'backend', 'diffdock') != 'fabind'
                            and torch.is_grad_enabled() and (self.expert_input_grad
                                                    or any(value.requires_grad for value in modulation))):
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
            if getattr(self.expert, 'backend', 'diffdock') in ('matcha', 'fabind'):
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
        if self.checkpoint_ternary and self.training and torch.is_grad_enabled():
            result = checkpoint(self.trifuse, graph, proposals, diffusion_state,
                                source=source, use_reentrant=False)
        else:
            result = self.trifuse(graph, proposals, diffusion_state, source=source)
        # Kept explicitly as diagnostic evidence, never consumed as another update.
        result['binary_proposals'] = proposals
        return result
