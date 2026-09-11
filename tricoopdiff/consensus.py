"""Model v2: immutable binary proposals and a single residual consensus field.

The executable state remains two relative SE(3) poses plus shared torsions.
Rigid proposals constrain different interface DOFs, so they are not averaged.
All vector channels use the common E3 frame; learned scalar coefficients make
the residual SE(3)-equivariant. Experts receive no trainable modulation.
"""

import torch
from torch import nn

from .graphs import build_subgraph_e3l, build_trifuse_graph, crop_receptor_to_ligand
from .expert_time import _set_time_embeddings
from .sdpi_flow import SDPIFlow
from .geometry_encoders import BondContextEncoder, TernaryGeometryEncoder
from .consensus_layers import JudgeHead, StateHead, TokenNegotiator, TorsionCoordinator


ARCHITECTURE = 'residual_consensus_v2'
CONSENSUS_CHECKPOINT_VERSION = 18
BRANCHES = ('e3_ligand', 'poi_ligand')


def bounded_vector(value):
    """Smooth isotropic compression; unlike elementwise tanh, rotation-safe."""
    return value / (1 + torch.linalg.vector_norm(value, dim=-1, keepdim=True))


class ConsensusFuse(nn.Module):
    """Read proposals after expert inference; decide all executable DOFs."""

    def __init__(self, node_dim=128, input_dim=None, num_transformer_layers=3,
                 dropout=0.1, **unused_legacy_dimensions):
        super().__init__()
        self.config = dict(node_dim=node_dim, input_dim=input_dim or node_dim,
                           num_transformer_layers=num_transformer_layers, dropout=dropout)
        self.pooling = TernaryGeometryEncoder(self.config['input_dim'], node_dim)
        self.transformer = TokenNegotiator(node_dim, num_transformer_layers, dropout)
        self.evidence_encoder = nn.Sequential(
            nn.Linear(9 + 8 * 8, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))
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

    def forward(self, graph, proposals, diffusion_state):
        coords, types = graph['node_coords'], graph['node_types']
        pooled = self.pooling(graph['node_features'], coords, types,
                              lig_edge_index=graph['lig_edge_index'])
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
        tokens = torch.stack((pooled['global'], pooled['e3l'], pooled['pol'],
                              pooled['ligand_tokens'][0].mean(0, keepdim=True), evidence), dim=1)
        negotiated = self.transformer(tokens)
        context = negotiated.mean(1)
        gates = torch.sigmoid(self.gate_head(context))
        coefficients = self.residual_head(context).reshape(-1, 4, 8)
        residual = torch.einsum('bik,bkj->bij', coefficients, basis)
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
                'torsion_gate': tor_gate.detach(),
                'torsion_disagreement': (tor_a - tor_b).detach().abs(),
            },
            'torsion_coordination': {'e3l_weight': tor_gate, 'residual': tor_residual},
            'confidence': self.judge_head(context), 'state': self.state_head(context),
        }


class ConsensusCoopDiff(nn.Module):
    """Frozen independent DiffDock reads followed by residual global decision."""

    architecture = ARCHITECTURE

    def __init__(self, frozen_expert, trifuse, num_conv_layers=None, use_film=False):
        super().__init__()
        if use_film:
            raise ValueError('model-v2 does not allow FiLM inside binary experts')
        if any('lora' in name for name, _ in frozen_expert.named_parameters()):
            raise ValueError('model-v2 Stage 1 requires an unmodified frozen expert')
        self.expert = frozen_expert.requires_grad_(False).eval()
        self.trifuse = trifuse
        self.num_conv_layers = num_conv_layers or frozen_expert.num_conv_layers
        self.use_film = False
        self._conditioning = SDPIFlow()

    def _expert_context(self):
        return torch.no_grad()

    def _select_expert_branch(self, index):
        """Frozen production experts have no branch-specific parameters."""
        return None

    def forward(self, e3_data, poi_data, ligand_data, t_e3l, t_pol=None,
                r_e3_to_poi=None, receptor_crop_cutoff=None, diffusion_state=None):
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
                tr, rot, tor = self.expert(graph)[:3]
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
        result = self.trifuse(graph, proposals, diffusion_state)
        # Kept explicitly as diagnostic evidence, never consumed as another update.
        result['binary_proposals'] = proposals
        return result
