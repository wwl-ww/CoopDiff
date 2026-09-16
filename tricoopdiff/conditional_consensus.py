"""Experimental condition-aware consensus; not a production C40 checkpoint format."""

import torch
from torch import nn

from .condition_features import ProtacConditionEncoder, condition_feature_shapes
from .consensus import ConsensusCoopDiff, ConsensusFuse


ARCHITECTURE = 'conditioned_residual_consensus_v3_prototype'
CONDITION_CONTRACT = 'protac_condition_features_v2_no_learned_provenance'


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
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f'condition/node alignment mismatch: {name}')
            values[name] = value
        encoded = self.condition_encoder(values)
        delta = reference.new_zeros(reference.shape[0], self.config['node_dim'])
        summaries = [encoded['sides'].mean(0, keepdim=True)]
        for index, name in enumerate(('protein1', 'protein2', 'ligand')):
            if encoded[name].shape[0] == 0:
                raise ValueError('conditioned consensus requires nonempty node groups')
            delta[types == index] = self.condition_node_projection(encoded[name]) * self.condition_scale
            summaries.append(encoded[name].mean(0, keepdim=True))
        raw_context = torch.stack(summaries).mean(0)
        context = raw_context * self.condition_scale
        gate_bias = self.condition_gate_head(raw_context) * self.condition_scale
        return {**graph, 'condition_node_residual': delta}, context, gate_bias

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
