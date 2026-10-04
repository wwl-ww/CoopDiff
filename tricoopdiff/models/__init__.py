"""Production model API, isolated from historical and experimental models."""

__all__ = ['ConsensusCoopDiff', 'ConsensusFuse', 'production_model_config',
           'FrozenDiffDock', 'build_diffdock_expert']


def __getattr__(name):
    """Load the requested API without making flow imports load a model."""
    if name in ('ConsensusCoopDiff', 'ConsensusFuse', 'production_model_config'):
        from . import consensus
        return getattr(consensus, name)
    if name in ('FrozenDiffDock', 'build_diffdock_expert'):
        from . import expert
        return getattr(expert, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
