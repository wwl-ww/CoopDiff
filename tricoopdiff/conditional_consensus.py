"""Experimental condition-aware consensus; not a production C40 checkpoint format."""

import torch
from torch import nn

from .condition_features import ProtacConditionEncoder, condition_feature_shapes
from .consensus import ConsensusCoopDiff, ConsensusFuse
from .binary_geometry import BinaryGeometryEncoder, binary_torsion_directions
from .interface_geometry import InterfaceGeometryEncoder


ARCHITECTURE = 'conditioned_residual_consensus_v8_side_aware_linker'
CONDITION_CONTRACT = 'protac_condition_features_v5_complete_binary_maps'


class SideAwareConditionFusion(nn.Module):
    """Keep E3 and POI evidence distinct before modelling linker compatibility."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.side = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * hidden_dim + 1, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(2)
        ])
        self.linker = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.joint = nn.Sequential(nn.Linear(5 * hidden_dim + 2, 2 * hidden_dim), nn.SiLU(),
                                   nn.Linear(2 * hidden_dim, hidden_dim))

    @staticmethod
    def _masked_mean(nodes, mask):
        weight = mask.to(nodes).unsqueeze(-1)
        return (nodes * weight).sum(0) / weight.sum().clamp_min(1.)

    def forward(self, encoded, features):
        ligand = encoded['ligand']
        side_contexts, available = [], []
        for index, protein_name in enumerate(('protein1', 'protein2')):
            side_available = (features['sides'][index, :3].sum() > 0).to(ligand)
            role = features['ligand'][:, index] > 0
            ligand_end = self._masked_mean(ligand, role)
            protein = encoded[protein_name].mean(0)
            inputs = torch.cat((encoded['sides'][index], protein, ligand_end,
                                side_available.reshape(1)))
            side_contexts.append(self.side[index](inputs) * side_available)
            available.append(side_available)
        linker_mask = features['ligand'][:, 2] > 0
        linker_available = linker_mask.any().to(ligand)
        linker = self.linker(torch.cat((self._masked_mean(ligand, linker_mask),
                                        linker_available.reshape(1)))) * linker_available
        left, right = side_contexts
        availability = torch.stack(available)
        joint = self.joint(torch.cat((left, right, linker, left * right,
                                      (left - right).abs(), availability)))
        # No condition must remain an exact zero path after migration.
        joint = joint * availability.max()
        return torch.stack(side_contexts), linker, joint


class ConditionedConsensusFuse(ConsensusFuse):
    """Condition residuals affect consensus only, never frozen expert proposals."""

    supports_conditions = True

    def __init__(self, **base_config):
        super().__init__(**base_config)
        # Keep common initialization and the caller RNG stream identical to v2.
        with torch.random.fork_rng(devices=[]):
            # Only CPU parameters are initialized here. manual_seed() would also
            # reset CUDA streams that this CPU-only fork does not restore.
            torch.random.default_generator.manual_seed(1701)
            self.condition_encoder = ProtacConditionEncoder(self.config['node_dim'])
            self.condition_node_projection = nn.Linear(self.config['node_dim'], self.config['node_dim'], bias=False)
            self.condition_gate_head = nn.Linear(self.config['node_dim'], 4, bias=False)
            self.condition_geometry = BinaryGeometryEncoder(self.config['node_dim'])
            self.condition_interface = InterfaceGeometryEncoder(self.config['node_dim'])
            self.condition_side_fusion = SideAwareConditionFusion(self.config['node_dim'])
            self.condition_vector_head = nn.Linear(self.config['node_dim'], 32, bias=False)
            self.condition_torsion_head = nn.Linear(self.config['node_dim'], 2, bias=False)
            # Exact function-preserving migration matters more than avoiding a
            # one-update delay in the new encoders. Each injection path first
            # learns whether to open, then propagates gradients into its body.
            self.condition_scale = nn.Parameter(torch.zeros(()))
            nn.init.normal_(self.condition_node_projection.weight, std=.01)
            nn.init.normal_(self.condition_gate_head.weight, std=.01)

    @classmethod
    def from_base(cls, base):
        if type(base) is not ConsensusFuse:
            raise ValueError('migration requires an unmodified base ConsensusFuse')
        with torch.random.fork_rng(devices=[]):
            model = cls(**base.config).to(next(base.parameters()).device)
        state = model.state_dict()
        state.update(base.state_dict())
        model.load_state_dict(state, strict=True)
        model.train(base.training)
        return model

    def prepare_conditions(self, graph, conditions):
        reference, types = graph['node_features'], graph['node_types']
        shapes = condition_feature_shapes(
            [int((types == 0).sum()), int((types == 1).sum())], int((types == 2).sum()))
        if set(conditions) != set(shapes):
            raise ValueError('condition feature fields do not match the versioned contract')
        values = {}
        for name, shape in shapes.items():
            value = torch.as_tensor(conditions[name], device=reference.device, dtype=reference.dtype).detach()
            if name in ('binary1', 'binary2'):
                if value.ndim != 4 or not 1 <= value.shape[0] <= 4096:
                    raise ValueError('binary geometry requires 1..4096 complete atom maps')
                shape = (value.shape[0], *shape[1:])
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f'condition/node alignment mismatch: {name}')
            values[name] = value
        encoded = self.condition_encoder(values)
        geometry_nodes, geometry_basis, forces = self.condition_geometry(graph, values)
        chemical = reference.new_zeros((len(reference), self.config['node_dim']))
        chemical[types != 2] = self.pooling.protein_feature_proj(reference[types != 2])
        chemical[types == 2] = self.pooling.ligand_feature_encoder(reference[types == 2])
        interface_nodes, interface_basis, interface_forces = self.condition_interface(graph, chemical)
        geometry_nodes = geometry_nodes + interface_nodes
        geometry_basis = torch.cat((geometry_basis, interface_basis), dim=1)
        forces = forces + interface_forces
        delta = reference.new_zeros(reference.shape[0], self.config['node_dim'])
        side_contexts, linker_context, joint_context = self.condition_side_fusion(encoded, values)
        for index, name in enumerate(('protein1', 'protein2', 'ligand')):
            if encoded[name].shape[0] == 0:
                raise ValueError('conditioned consensus requires nonempty node groups')
            delta[types == index] = self.condition_node_projection(encoded[name]) * self.condition_scale
            delta[types == index] += geometry_nodes[types == index] * self.condition_scale
        for side in (0, 1):
            delta[types == side] += side_contexts[side] * self.condition_scale
            ligand_role = values['ligand'][:, side:side + 1]
            delta[types == 2] += ligand_role * side_contexts[side] * self.condition_scale
        delta[types == 2] += values['ligand'][:, 2:3] * linker_context * self.condition_scale
        raw_context = (joint_context + geometry_nodes.mean(0)).unsqueeze(0)
        context = raw_context * self.condition_scale
        gate_bias = self.condition_gate_head(raw_context) * self.condition_scale
        coefficients = self.condition_vector_head(raw_context).reshape(1, 4, 8)
        vector_residual = torch.einsum('bik,bkj->bij', coefficients, geometry_basis)
        torsion_directions = binary_torsion_directions(graph, forces)
        bonds = graph['lig_edge_index'][:, graph['lig_edge_mask']].T
        ligand_nodes = geometry_nodes[types == 2]
        bond_context = (ligand_nodes[bonds[:, 0]] + ligand_nodes[bonds[:, 1]]) / 2 + raw_context
        torsion_residual = (self.condition_torsion_head(bond_context)
                            * torsion_directions).sum(-1).unsqueeze(0)
        return {**graph, 'condition_node_residual': delta,
                'condition_velocity_residual': vector_residual * self.condition_scale,
                'condition_torsion_residual': torsion_residual * self.condition_scale}, context, gate_bias

    def prototype_checkpoint(self, parent_sha256):
        if len(parent_sha256) != 64 or any(c not in '0123456789abcdef' for c in parent_sha256):
            raise ValueError('record the parent checkpoint SHA256')
        return {'format': 'conditioned_fuse_prototype_v1', 'architecture': ARCHITECTURE,
                'condition_contract': CONDITION_CONTRACT, 'base_model_config': dict(self.config),
                'parent_checkpoint_sha256': parent_sha256,
                'fuse_state_dict': {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                'scope': 'experimental fusion module only; not a trained production refinement checkpoint'}


def load_prototype_checkpoint(payload):
    if (payload.get('format') != 'conditioned_fuse_prototype_v1'
            or payload.get('architecture') != ARCHITECTURE
            or payload.get('condition_contract') != CONDITION_CONTRACT):
        raise ValueError('unsupported condition prototype checkpoint')
    parent = payload.get('parent_checkpoint_sha256', '')
    if len(parent) != 64 or any(c not in '0123456789abcdef' for c in parent):
        raise ValueError('missing parent checkpoint identity')
    model = ConditionedConsensusFuse(**payload['base_model_config'])
    model.load_state_dict(payload['fuse_state_dict'], strict=True)
    return model


class ConditionedConsensusCoopDiff(ConsensusCoopDiff):
    """Explicit experimental wrapper; legacy evaluators cannot silently enable it."""

    architecture = ARCHITECTURE

    def __init__(self, frozen_expert, trifuse, **kwargs):
        if not isinstance(trifuse, ConditionedConsensusFuse):
            raise ValueError('conditional model requires ConditionedConsensusFuse')
        super().__init__(frozen_expert, trifuse, **kwargs)
