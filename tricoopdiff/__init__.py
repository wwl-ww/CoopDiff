"""TriCoopDiff: PROTAC Ternary Complex Cooperative Diffusion Model."""

# Apply torch_cluster / torch_scatter pure-PyTorch fallbacks BEFORE any submodule
# binds those symbols. expert.py imports torch_scatter, and the vendored cg_model does
# `from torch_cluster import ...`; if the patch runs after those bindings, they
# keep the (broken-on-GPU) native kernels.
from . import diffdock_compat  # noqa: F401

from tricoopdiff.inputs.graphs import build_subgraph_e3l, build_subgraph_pol, build_trifuse_graph

__all__ = [
    'TriCoopDiff', 'build_subgraph_e3l', 'build_subgraph_pol',
    'build_trifuse_graph', 'SDPIFlow', 'InterfaceDiffusion',
]


def __getattr__(name):
    """Keep the historical public model import without eagerly loading FiLM."""
    if name == 'TriCoopDiff':
        from tricoopdiff.legacy.model import TriCoopDiff
        return TriCoopDiff
    if name == 'SDPIFlow':
        from tricoopdiff.sdpi_flow import SDPIFlow
        return SDPIFlow
    if name == 'InterfaceDiffusion':
        from tricoopdiff.legacy.diffusion import InterfaceDiffusion
        return InterfaceDiffusion
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
