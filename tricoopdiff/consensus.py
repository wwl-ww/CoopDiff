"""Frozen binary reads and trainable ternary motion.

Legacy checkpoints retain the dual-interface residual field. New direct mode
uses node-level ternary geometry to decide independent ligand/POI centroid
motions; expert proposals are context only. All vectors use the E3 frame.
"""

from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .graphs import build_subgraph_e3l, build_trifuse_graph, crop_receptor_to_ligand
from .expert_time import _set_time_embeddings
from .sdpi_flow import SDPIFlow
from .geometry_encoders import BondContextEncoder, TernaryGeometryEncoder
from .consensus_layers import JudgeHead, StateHead, TokenNegotiator, TorsionCoordinator


ARCHITECTURE = 'residual_consensus_v2'
DIRECT_ARCHITECTURE = 'ternary_geometry_independent_v1'
CONSENSUS_CHECKPOINT_VERSION = 23
INTERFACE_OBJECTIVE_CHECKPOINT_VERSION = 22
OBJECTIVE_CONTRACT_CHECKPOINT_VERSION = 21
ENDPOINT_PROTOCOL_CHECKPOINT_VERSION = 20
LEGACY_CONSENSUS_CHECKPOINT_VERSION = 18
PREVIOUS_CONSENSUS_CHECKPOINT_VERSION = 19
BRANCHES = ('e3_ligand', 'poi_ligand')
SOURCE_NAMES = ('sdpi', 'bitseed')
SOURCE_CONTRACT_VERSION = 'candidate_source_conditioning_v1'
PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v1'


def production_model_config(node_dim, input_dim, num_transformer_layers=3,
                            source_conditioning=False):
    """One explicit new-model recipe; older weights retain their own config."""
    return dict(node_dim=node_dim, input_dim=input_dim,
                num_transformer_layers=num_transformer_layers, dropout=0.,
                source_conditioning=bool(source_conditioning),
                direct_ternary_motion=True, global_node_context=True,
                spatial_geometry=True, motion_attention=True,
                global_node_attention=True, rotation_scale_normalization=True,
                model_profile=PRODUCTION_MODEL_PROFILE)


def rotation_levers(coords, normalize_body_scale=False):
    """Dimensionless centered levers; clamp squared radius before sqrt."""
    centered = coords - coords.mean(0)
    scale = (centered.square().sum(-1).mean().clamp_min(1.).sqrt()
             if normalize_body_scale else 10.)
    return bounded_vector(centered / scale)


def candidate_source_tensor(source, reference):
    """Normalize one immutable trajectory source to a ``[1]`` class tensor."""
    if isinstance(source, str):
        if source not in SOURCE_NAMES:
            raise ValueError(f'unknown candidate source: {source!r}')
        source = SOURCE_NAMES.index(source)
    value = torch.as_tensor(source, device=reference.device, dtype=torch.long).reshape(-1)
    if value.shape != (1,) or not bool(((value >= 0) & (value < len(SOURCE_NAMES))).all()):
        raise ValueError('candidate source must be one SDPI/BiTSeed class')
    return value


def bounded_vector(value):
    """Smooth isotropic compression; unlike elementwise tanh, rotation-safe."""
    return value / (1 + torch.linalg.vector_norm(value, dim=-1, keepdim=True))


def attentive_interface_motion(weights, direction, torque, sender_types):
    """Separate interface/channel attention; large partners cannot win by count."""
    fields = torch.stack((direction, torque), dim=1)
    interfaces = []
    for component in sender_types.unique():
        selected = sender_types == component
        attention = weights[selected, 2:4].softmax(dim=0)
        interfaces.append((attention[..., None] * weights[selected, :2, None]
                           * fields[selected]).sum(0))
    motion = torch.stack(interfaces).mean(0)
    return motion[:1], motion[1:]


class ConsensusFuse(nn.Module):
    """Decide motion from ternary geometry, with an explicit legacy mode."""

    supports_conditions = False

    def __init__(self, node_dim=128, input_dim=None, num_transformer_layers=3,
                 dropout=0.1, source_conditioning=False,
                 protein_pair_motion=False,
                 direct_ternary_motion=False,
                 global_node_context=False,
                 spatial_geometry=False,
                 motion_attention=False,
                 global_node_attention=False,
                 rotation_scale_normalization=False,
                 model_profile=None,
                 **unused_legacy_dimensions):
        super().__init__()
        self.config = dict(node_dim=node_dim, input_dim=input_dim or node_dim,
                           num_transformer_layers=num_transformer_layers, dropout=dropout,
                           source_conditioning=bool(source_conditioning))
        self.protein_pair_motion = bool(protein_pair_motion)
        self.direct_ternary_motion = bool(direct_ternary_motion)
        if global_node_context and not self.direct_ternary_motion:
            raise ValueError('global node context requires direct ternary motion')
        if global_node_context:
            self.config['global_node_context'] = True
        if spatial_geometry and not self.direct_ternary_motion:
            raise ValueError('spatial geometry requires direct ternary motion')
        if spatial_geometry:
            self.config['spatial_geometry'] = True
        self.motion_attention = bool(motion_attention)
        if self.motion_attention and not self.direct_ternary_motion:
            raise ValueError('motion attention requires direct ternary motion')
        if self.motion_attention:
            self.config['motion_attention'] = True
        if global_node_attention and not self.direct_ternary_motion:
            raise ValueError('global node attention requires direct ternary motion')
        if global_node_attention:
            self.config['global_node_attention'] = True
        self.rotation_scale_normalization = bool(rotation_scale_normalization)
        if self.rotation_scale_normalization and not self.direct_ternary_motion:
            raise ValueError('rotation scale normalization requires direct ternary motion')
        if self.rotation_scale_normalization:
            self.config['rotation_scale_normalization'] = True
        if self.direct_ternary_motion and self.protein_pair_motion:
            raise ValueError('direct ternary motion replaces the protein-pair residual path')
        if self.direct_ternary_motion:
            self.config['direct_ternary_motion'] = True
        if self.protein_pair_motion:
            self.config['protein_pair_motion'] = True
        if model_profile is not None:
            if model_profile != PRODUCTION_MODEL_PROFILE:
                raise ValueError(f'unknown model profile: {model_profile}')
            self.config['model_profile'] = model_profile
            expected = production_model_config(node_dim, self.config['input_dim'],
                                               num_transformer_layers, source_conditioning)
            if self.config != expected:
                raise ValueError('production model profile cannot silently change its architecture or dropout')
        self.pooling = TernaryGeometryEncoder(self.config['input_dim'], node_dim,
                                             joint_geometry=self.direct_ternary_motion,
                                             global_node_context=global_node_context,
                                             spatial_geometry=spatial_geometry,
                                             global_node_attention=global_node_attention)
        self.transformer = TokenNegotiator(node_dim, num_transformer_layers, dropout)
        self.evidence_encoder = nn.Sequential(
            nn.Linear(9 + 8 * 8, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))
        if source_conditioning:
            # Source-only parameters must not perturb the RNG sequence used by
            # shared B/C parameters; otherwise the factorial comparison starts
            # from different common weights.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                self.source_embedding = nn.Embedding(len(SOURCE_NAMES), node_dim)
                self.source_gate_head = nn.Linear(node_dim, 4, bias=False)
                nn.init.normal_(self.source_gate_head.weight, std=0.01)
        else:
            self.source_embedding = None
            self.source_gate_head = None
        # Four independent interface/component trust gates, not calibrated confidence.
        self.gate_head = nn.Linear(node_dim, 4)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 2.0)
        self.residual_head = nn.Linear(node_dim, 4 * 8)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.bond_context_encoder = BondContextEncoder(node_dim)
        self.torsion_coordinator = TorsionCoordinator(node_dim)
        self.judge_head = JudgeHead(node_dim)
        self.state_head = StateHead(node_dim)
        for module in (self.judge_head, self.state_head):
            module.requires_grad_(False)
        if self.protein_pair_motion:
            # Keep individual labelled contacts until the equivariant motion head.
            self.protein_pair_head = nn.Sequential(
                nn.Linear(2 * node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, 4))
            nn.init.zeros_(self.protein_pair_head[-1].weight)
            nn.init.zeros_(self.protein_pair_head[-1].bias)
        if self.direct_ternary_motion:
            self.motion_head = nn.Sequential(
                nn.Linear(3 * node_dim + self.pooling.edge_geometry_dim, node_dim),
                nn.SiLU(), nn.Linear(node_dim, 4 if self.motion_attention else 2))
            self.direct_torsion_head = nn.Sequential(
                nn.Linear(node_dim + 2, node_dim), nn.SiLU(), nn.Linear(node_dim, 1))
            for head in (self.motion_head, self.direct_torsion_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
            # Retain old checkpoint keys, but never optimize inactive heads.
            for module in (self.gate_head, self.residual_head, self.torsion_coordinator,
                           self.source_gate_head):
                if module is not None:
                    module.requires_grad_(False)

    def _direct_motion(self, pooled, graph, proposals, context):
        """Experts supply context only; all executable motion is learned here.

        Historical branch slots now denote ligand and POI centroid velocities.
        The explicit output-mode tag changes the decoder and teacher labels;
        it must survive scaling and sampling, never be inferred from slot names.
        """
        from .sdpi_flow import INDEPENDENT_RIGID_MODE

        joint = pooled['joint_geometry']
        receiver, sender = joint['receiver'], joint['sender']
        coords, types = graph['node_coords'], graph['node_types']
        weights = self.motion_head(torch.cat((
            joint['features'][receiver], joint['features'][sender],
            context.expand(len(receiver), -1), joint['geometry']), -1))
        predictions = {'rigid_output_mode': INDEPENDENT_RIGID_MODE}
        fields = []
        for branch, component in zip(BRANCHES, (2, 1)):
            selected = types[receiver] == component
            if self.motion_attention:
                # Internal edges describe body shape, not an external interface
                # driving rigid motion. Keep them in messages, not this readout.
                selected = selected & (types[sender] != component)
            direction = joint['direction'][selected]
            component_mask = types == component
            component_levers = rotation_levers(coords[component_mask], self.rotation_scale_normalization)
            all_levers = torch.zeros_like(coords).index_copy(0, component_mask.nonzero().flatten(), component_levers)
            lever = all_levers[receiver[selected]]
            torque = torch.cross(lever, direction, dim=-1)
            if self.motion_attention:
                tr, rot = attentive_interface_motion(
                    weights[selected], direction, torque, types[sender[selected]])
            else:
                tr = (weights[selected, 0:1] * direction).mean(0, keepdim=True)
                rot = (weights[selected, 1:2] * torque).mean(0, keepdim=True)
            predictions[branch] = {'tr': tr, 'rot': rot, 'tor': proposals[branch]['tor']}
            fields.extend((tr, rot))
        bonds = self.bond_context_encoder(
            pooled['ligand_tokens'][0], coords[types == 2], graph['lig_edge_index'],
            graph['lig_edge_mask'], graph['lig_mask_rotate'], context)
        tor_a, tor_b = (proposals[b]['tor'] for b in BRANCHES)
        opinions = torch.stack((tor_a.tanh(), tor_b.tanh()), -1)
        predictions['torsion'] = self.direct_torsion_head(torch.cat((bonds, opinions), -1)).squeeze(-1)
        return {'normalized_velocity': predictions, 'rank_features': context,
                'residual_penalty': context.new_zeros(()),
                'consensus_diagnostics': {
                    'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                    'rigid_motion_rms': torch.stack(fields).detach().square().mean().sqrt()},
                'confidence': self.judge_head(context), 'state': self.state_head(context)}

    def prepare_conditions(self, graph, conditions):
        raise ValueError('legacy ConsensusFuse does not accept PROTAC conditions')

    def forward(self, graph, proposals, diffusion_state, source=0, *, conditions=None):
        condition_context, condition_gate_bias = None, None
        if conditions is not None:
            graph, condition_context, condition_gate_bias = self.prepare_conditions(graph, conditions)
        coords, types = graph['node_coords'], graph['node_types']
        pooling_kwargs = {'node_residual': graph['condition_node_residual']} if conditions is not None else {}
        if self.protein_pair_motion:
            pooling_kwargs['return_protein_pair_contacts'] = True
        pooled = self.pooling(graph['node_features'], coords, types,
                              lig_edge_index=graph['lig_edge_index'], **pooling_kwargs)
        centers = [coords[types == i].mean(0, keepdim=True) for i in range(3)]
        directions = [bounded_vector((centers[i] - centers[2]) / 10) for i in (0, 1)]
        raw = [proposals[b][c] for b in BRANCHES for c in ('tr', 'rot')]
        # Cross products provide an independent geometry direction and axial channel.
        geometry_cross = torch.cross(directions[0], directions[1], dim=-1)
        proposal_cross = torch.cross(bounded_vector(raw[0]), bounded_vector(raw[2]), dim=-1)
        basis = torch.stack([bounded_vector(v) for v in raw] + directions
                            + [geometry_cross, proposal_cross], dim=1)
        gram = basis @ basis.transpose(-1, -2)
        evidence = self.evidence_encoder(torch.cat((diffusion_state, gram.flatten(1)), -1))
        if condition_context is not None:
            evidence = evidence + condition_context
        source_index = candidate_source_tensor(source, evidence)
        source_context = None
        if self.source_embedding is not None:
            source_context = self.source_embedding(source_index)
            evidence = evidence + source_context
        tokens = torch.stack((pooled['global'], pooled['e3l'], pooled['pol'],
                              pooled['ligand_tokens'][0].mean(0, keepdim=True), evidence), dim=1)
        negotiated = self.transformer(tokens)
        context = negotiated.mean(1)
        if self.direct_ternary_motion:
            return self._direct_motion(pooled, graph, proposals, context)
        gate_logits = self.gate_head(context)
        if condition_gate_bias is not None:
            gate_logits = gate_logits + condition_gate_bias
        if source_context is not None:
            # This source-owned nonzero path avoids the zero-initialized common
            # heads blocking embedding gradients on the first training batch.
            gate_logits = gate_logits + self.source_gate_head(source_context)
        gates = torch.sigmoid(gate_logits)
        coefficients = self.residual_head(context).reshape(-1, 4, 8)
        residual = torch.einsum('bik,bkj->bij', coefficients, basis)
        pair_motion = torch.zeros_like(residual)
        if self.protein_pair_motion:
            contacts = pooled['protein_pair_contacts'][0]
            weights = self.protein_pair_head(torch.cat((
                contacts['tokens'], context.expand(len(contacts['tokens']), -1)), dim=-1))
            direction = contacts['direction']
            torque = torch.cross(bounded_vector(contacts['lever']), direction, dim=-1)
            pair_motion = torch.stack([
                (weights[:, i:i+1] * (direction if i % 2 == 0 else torque)).mean(0)
                for i in range(4)], dim=0).unsqueeze(0)
            residual = residual + pair_motion
        prior = gates.unsqueeze(-1) * torch.stack(raw, 1)
        field = prior + residual
        bond_context = self.bond_context_encoder(
            pooled['ligand_tokens'][0], coords[types == 2], graph['lig_edge_index'],
            graph['lig_edge_mask'], graph['lig_mask_rotate'], context)
        tor_a, tor_b = (proposals[b]['tor'] for b in BRANCHES)
        torsion, tor_gate, tor_residual = self.torsion_coordinator(bond_context, tor_a, tor_b)
        predictions = {
            branch: {'tr': field[:, 2 * i], 'rot': field[:, 2 * i + 1],
                     'tor': proposals[branch]['tor']}
            for i, branch in enumerate(BRANCHES)
        }
        predictions['torsion'] = torsion
        residual_penalty = residual.square().mean()
        if tor_residual.numel():
            residual_penalty = residual_penalty + tor_residual.square().mean()
        conflicts = torch.stack([
            1 - torch.nn.functional.cosine_similarity(raw[i], raw[i + 2], dim=-1)
            for i in (0, 1)], dim=-1)
        return {
            'normalized_velocity': predictions,
            # Invariant negotiated context reserved for a separately supervised
            # pose ranker. It is not consumed by the diffusion loss.
            'rank_features': context,
            'residual_penalty': residual_penalty,
            'consensus_diagnostics': {
                'rigid_gates': gates.detach(), 'direction_disagreement': conflicts.detach(),
                'rigid_prior_rms': prior.detach().square().mean().sqrt(),
                'rigid_residual_rms': residual.detach().square().mean().sqrt(),
                'protein_pair_motion_rms': pair_motion.detach().square().mean().sqrt(),
                'torsion_gate': tor_gate.detach(),
                'torsion_disagreement': (tor_a - tor_b).detach().abs(),
            },
            'torsion_coordination': {'e3l_weight': tor_gate, 'residual': tor_residual},
            'confidence': self.judge_head(context), 'state': self.state_head(context),
        }


class ConsensusCoopDiff(nn.Module):
    """Frozen independent DiffDock reads followed by a ternary motion field."""

    architecture = ARCHITECTURE

    def __init__(self, frozen_expert, trifuse, num_conv_layers=None, use_film=False,
                 expert_input_grad=True, checkpoint_ternary=False):
        super().__init__()
        if getattr(trifuse, 'supports_conditions', False) and self.architecture == ARCHITECTURE:
            raise ValueError('conditional fuse requires its separate model architecture')
        if use_film:
            raise ValueError('model-v2 does not allow FiLM inside binary experts')
        if any('lora' in name for name, _ in frozen_expert.named_parameters()):
            raise ValueError('model-v2 Stage 1 requires an unmodified frozen expert')
        self.expert = frozen_expert.requires_grad_(False).eval()
        self.trifuse = trifuse
        if getattr(trifuse, 'direct_ternary_motion', False):
            self.architecture = DIRECT_ARCHITECTURE
        self.num_conv_layers = num_conv_layers or frozen_expert.num_conv_layers
        self.use_film = False
        self.expert_input_grad = bool(expert_input_grad)
        self.checkpoint_ternary = bool(checkpoint_ternary)
        self._conditioning = SDPIFlow()

    def _expert_context(self):
        # Frozen weights do not require detached input Jacobians. Respect the
        # caller's no_grad during inference; never force enable_grad here.
        return nullcontext() if self.expert_input_grad else torch.no_grad()

    def _select_expert_branch(self, index):
        """Frozen production experts have no branch-specific parameters."""
        return None

    def _read_expert(self, graph):
        # Recompute frozen activations during backward rather than retaining
        # eleven expert layers at every rollout step. Clone because DiffDock
        # stores intermediate attributes on its private graph.
        def read(private_graph):
            return tuple(self.expert(private_graph.clone())[:3])

        needs_input_grad = (graph['ligand'].pos.requires_grad
                            or graph['receptor'].pos.requires_grad)
        if self.expert_input_grad and torch.is_grad_enabled() and needs_input_grad:
            return checkpoint(read, graph, use_reentrant=False)
        return read(graph)

    def forward(self, e3_data, poi_data, ligand_data, t_e3l, t_pol=None,
                r_e3_to_poi=None, receptor_crop_cutoff=None, diffusion_state=None,
                source=0, *, conditions=None):
        if conditions is not None and not self.trifuse.supports_conditions:
            raise ValueError('legacy model cannot consume PROTAC conditions')
        if t_e3l.shape != (1,):
            raise ValueError('model-v2 currently requires B=1')
        if t_pol is not None and not torch.equal(t_e3l, t_pol):
            raise ValueError('consensus requires one shared flow clock')
        if diffusion_state is None or diffusion_state.shape != (1, 9):
            raise ValueError('model-v2 requires explicit diffusion_state [1,9]')
        frame = (r_e3_to_poi[0] if r_e3_to_poi is not None
                 else torch.eye(3, device=t_e3l.device, dtype=ligand_data['coords'].dtype))
        raw = {}
        # Private graph tensors ensure even in-place expert internals cannot
        # mutate the shared state or the other branch's input.
        with self._expert_context():
            for index, (branch, receptor) in enumerate(zip(BRANCHES, (e3_data, poi_data))):
                self._select_expert_branch(index)
                center = receptor['coords'][0].mean(0)
                rotation = frame if index else torch.eye(3, device=frame.device, dtype=frame.dtype)
                rec_pos = (receptor['coords'][0] - center) @ rotation
                lig_pos = (ligand_data['coords'][0] - center) @ rotation
                cutoff = (receptor_crop_cutoff[0] if receptor_crop_cutoff is not None
                          and self.expert.model.dynamic_max_cross else None)
                crop = crop_receptor_to_ligand(
                    rec_pos, receptor['features'][0], lig_pos,
                    rec_edge_index=receptor.get('edge_index'), cutoff=cutoff,
                    min_residues=1, min_residues_with_edges=8, require_internal_edges=True)
                graph = build_subgraph_e3l(
                    crop['coords'], crop['features'], lig_pos, ligand_data['features'][0],
                    rec_edge_index=crop['edge_index'], lig_edge_index=ligand_data['edge_index'],
                    lig_edge_attr=ligand_data['edge_attr'], lig_edge_mask=ligand_data['edge_mask'],
                    lig_mask_rotate=ligand_data['mask_rotate']).clone()
                _set_time_embeddings(graph, t_e3l, self.expert.model)
                tr, rot, tor = self._read_expert(graph)
                if not all(torch.isfinite(v).all() for v in (tr, rot, tor)):
                    raise FloatingPointError(f'{branch}: non-finite frozen proposal')
                raw[branch] = {'tr': tr @ rotation.T, 'rot': rot @ rotation.T,
                               'tor': tor.reshape(1, -1)}
            proposals = self._conditioning.normalized_predictions(
                {'interface_scores': raw, 'torsion_score': raw[BRANCHES[0]]['tor']}, t_e3l)
        graph = build_trifuse_graph(
            e3_data['coords'][0], e3_data['features'][0],
            poi_data['coords'][0], poi_data['features'][0],
            ligand_data['coords'][0], ligand_data['features'][0],
            lig_edge_index=ligand_data['edge_index'], lig_edge_mask=ligand_data['edge_mask'],
            lig_mask_rotate=ligand_data['mask_rotate'])
        condition_kwargs = {'conditions': conditions} if conditions is not None else {}
        if self.checkpoint_ternary and self.training and torch.is_grad_enabled():
            result = checkpoint(self.trifuse, graph, proposals, diffusion_state,
                                source=source, use_reentrant=False, **condition_kwargs)
        else:
            result = self.trifuse(graph, proposals, diffusion_state, source=source, **condition_kwargs)
        # Kept explicitly as diagnostic evidence, never consumed as another update.
        result['binary_proposals'] = proposals
        return result
