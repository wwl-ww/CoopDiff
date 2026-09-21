"""Initialization identity, independent of torsion sampling and motion bounds.

Legacy anchored checkpoints used row zero. Explicit warm-start migration copies
that trained row into BiAnchor, preserving behavior before further training.

Training CLI: --conditioning-task dual_binary --init-source bianchor
--warm-start <legacy-or-bianchor-checkpoint>. This starts a fresh optimizer;
--resume instead inherits the saved source contract and cannot change it.
New dual_binary runs default to BiAnchor; --init-source legacy_zero explicitly
reproduces historical runs. Sampling reads the contract from the checkpoint.
"""

import torch
from torch import nn


SOURCE_NAMES = ('sdpi', 'bitseed', 'bianchor')
SOURCE_POLICY = 'bianchor_initialization_source_v1'
FORMAT = 'anchored_consensus_training_v11'


def uses_bianchor(config):
    names = tuple(config.get('source_names', SOURCE_NAMES[:2]))
    if names not in (SOURCE_NAMES[:2], SOURCE_NAMES):
        raise ValueError('unsupported initialization source vocabulary')
    return names == SOURCE_NAMES


def anchored_source(model):
    """Use checkpoint-declared semantics, never reinterpret legacy row zero."""
    return 'bianchor' if uses_bianchor(model.trifuse.config) else 0


def install_bianchor_source(model):
    """Expand the existing embedding before constructing the optimizer/EMA."""
    fuse = model.trifuse
    if uses_bianchor(fuse.config):
        return
    old = fuse.source_embedding
    if old is None or old.num_embeddings != 2:
        raise ValueError('BiAnchor migration requires the legacy two-source embedding')
    weights = torch.cat((old.weight.detach(), old.weight[:1].detach()), dim=0).clone()
    fuse.source_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
    fuse.source_names = SOURCE_NAMES
    fuse.config['source_names'] = list(SOURCE_NAMES)
