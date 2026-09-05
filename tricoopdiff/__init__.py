"""TriCoopDiff: PROTAC Ternary Complex Cooperative Diffusion Model."""

# Apply torch_cluster / torch_scatter pure-PyTorch fallbacks BEFORE any submodule
# binds those symbols. trifuse.py and expert.py both do
# `from torch_scatter import ...` at import time, and the vendored cg_model does
# `from torch_cluster import ...`; if the patch runs after those bindings, they
# keep the (broken-on-GPU) native kernels.
from . import diffdock_compat  # noqa: F401

from .model import TriCoopDiff
from .graphs import build_subgraph_e3l, build_subgraph_pol, build_trifuse_graph
from .diffusion import InterfaceDiffusion
