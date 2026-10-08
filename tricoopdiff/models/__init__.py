"""Production model API, isolated from historical and experimental models."""

__all__ = ['TernaryAssemblyModel', 'TernaryAssembly', 'production_model_config',
           'FrozenDiffDock', 'build_diffdock_expert']


def __getattr__(name):
    """Load the requested API without making flow imports load a model."""
    if name in ('TernaryAssemblyModel', 'TernaryAssembly', 'production_model_config'):
        from . import assembly
        return getattr(assembly, name)
    if name in ('FrozenDiffDock', 'build_diffdock_expert'):
        from . import expert
        return getattr(expert, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
