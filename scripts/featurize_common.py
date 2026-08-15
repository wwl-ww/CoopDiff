#!/usr/bin/env python
"""
Shared featurization utilities for TriCoopDiff training + smoke test.

Extracts the featurization path that ``scripts/mvp_smoke_test.py`` validated
(receptor/ligand subgraph featurization via DiffDock's ``InferenceDataset`` +
on-the-fly ESM2-650M embeddings) into a single reusable module.

These functions depend on the sibling ``../DiffDock`` checkout for *data
featurization only* (``InferenceDataset`` / ``get_sequences``), which is why they
live under ``scripts/`` rather than the self-contained ``tricoopdiff/`` package.
The model itself runs entirely on the vendored ``tricoopdiff.diffdock.*``.
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DIFFDOCK_DIR = _PROJECT_ROOT.parent / "DiffDock"

# Make the project root importable so `import tricoopdiff.*` works regardless of
# how this module is invoked (it is not pip-installed).
sys.path.insert(0, str(_PROJECT_ROOT))

# (1) Import diffdock_compat FIRST — its import-time side effect patches
# torch_cluster / torch_scatter with pure-PyTorch fallbacks before the vendored
# cg_model and ../DiffDock's process_mols bind those symbols at import time.
import tricoopdiff.diffdock_compat  # noqa: F401

# (2) Add ../DiffDock to sys.path for data featurization ONLY.
if not _DIFFDOCK_DIR.exists():
    raise RuntimeError(f"DiffDock checkout not found at {_DIFFDOCK_DIR}")
sys.path.insert(0, str(_DIFFDOCK_DIR))

import torch
from torch_geometric.loader import DataLoader

from utils.inference_utils import InferenceDataset, get_sequences

# DiffDock's SO(3)/torus cache lookup resolves relative to CWD; the cache files
# live in the project root. Chdir there so score lookups find them (SLURM also
# does this, but interactive runs may not).
os.chdir(_PROJECT_ROOT)


def generate_esm_embeddings(protein_paths, device):
    """Generate ESM2-650M embeddings for protein PDBs (mirrors experiment1).

    Returns: dict {protein_path: [per-chain torch.Tensor (n_res, 1280)]}.
    """
    from tricoopdiff.esm_embedder import ESMEmbedder

    print(f"Loading ESM2-650M on {device} via HF mirror...", flush=True)
    embedder = ESMEmbedder(device=device)

    print(f"Extracting sequences from {len(protein_paths)} proteins...", flush=True)
    seqs_list = get_sequences(protein_paths, [None] * len(protein_paths))

    per_protein_info = []
    all_chain_seqs = []
    for path, seq_str in zip(protein_paths, seqs_list):
        chain_seqs = seq_str.split(":")
        per_protein_info.append((path, chain_seqs))
        all_chain_seqs.extend(chain_seqs)

    unique_seqs = list(dict.fromkeys(all_chain_seqs))
    print(f"  {len(all_chain_seqs)} chains, {len(unique_seqs)} unique", flush=True)

    unique_embeddings = embedder.embed_from_fasta(unique_seqs, batch_size=4)
    seq_to_emb = dict(zip(unique_seqs, unique_embeddings))

    return {
        path: [seq_to_emb[s].cpu() for s in chain_seqs]
        for path, chain_seqs in per_protein_info
    }


def featurize_subgraph(case_label, protein_path, ligand_path, chain_embs,
                       score_model_args, out_root=None):
    """Featurize one two-body subgraph and return its Batch graph."""
    if out_root is None:
        out_root = _PROJECT_ROOT / "results" / "smoke_tmp"
    precomputed = [[t for t in chain_embs]]  # keep on CPU
    dataset = InferenceDataset(
        out_dir=str(out_root / case_label.replace("/", "_")),
        complex_names=[case_label],
        protein_files=[protein_path],
        ligand_descriptions=[ligand_path],
        protein_sequences=[None],
        lm_embeddings=True,
        precomputed_lm_embeddings=precomputed,
        receptor_radius=score_model_args.receptor_radius,
        remove_hs=score_model_args.remove_hs,
        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
        all_atoms=score_model_args.all_atoms,
        atom_radius=score_model_args.atom_radius,
        atom_max_neighbors=score_model_args.atom_max_neighbors,
        knn_only_graph=(
            False if not hasattr(score_model_args, "not_knn_only_graph")
            else not score_model_args.not_knn_only_graph
        ),
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return next(iter(loader))


def extract_receptor(batch_graph, device):
    """Receptor tensors → the dict format TriCoopDiff.forward expects."""
    return {
        'coords': batch_graph['receptor'].pos.unsqueeze(0).to(device),
        'features': batch_graph['receptor'].x.unsqueeze(0).to(device),
        'edge_index': batch_graph['receptor', 'receptor'].edge_index.to(device),
    }


def extract_ligand(batch_graph, device):
    """Ligand tensors → the dict format TriCoopDiff.forward expects."""
    # mask_rotate (per-rotatable-bond atom mask, [n_rot, n_atoms] bool) is stored
    # as a numpy array on the ligand node; torch_geometric collate wraps per-ligand
    # arrays in a list (Batch) even at batch_size=1. Normalize to the single array.
    mask_rotate = batch_graph['ligand'].mask_rotate
    if isinstance(mask_rotate, (list, tuple)):
        mask_rotate = mask_rotate[0]
    return {
        'coords': batch_graph['ligand'].pos.unsqueeze(0).to(device),
        'features': batch_graph['ligand'].x.unsqueeze(0).to(device),
        'edge_index': batch_graph['ligand', 'ligand'].edge_index.to(device),
        'edge_attr': batch_graph['ligand', 'ligand'].edge_attr.to(device),
        'edge_mask': batch_graph['ligand'].edge_mask.to(device),
        'mask_rotate': mask_rotate,
    }


def _strip_batch_and_cpu(data):
    """Convert a batched device dict → unbatched CPU dict (on-disk cache format)."""
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            v = v.detach().cpu()
        if k in ("coords", "features"):
            v = v[0]  # drop the leading B=1 dim
        out[k] = v
    return out


def _to_device_and_batch(data, device):
    """Reverse of ``_strip_batch_and_cpu``: CPU cache dict → batched device dict."""
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            v = v.to(device)
        if k in ("coords", "features"):
            v = v.unsqueeze(0)  # re-add the leading B=1 dim
        out[k] = v
    return out


def featurize_complex(case_label, protein1, protein2, ligand_path,
                      score_model_args, device, cache_dir=None):
    """Featurize one ternary complex into the three B=1 data dicts.

    Args:
        case_label: unique case name (used for cache key + ESM dedup).
        protein1: E3 protein PDB path.
        protein2: POI protein PDB path.
        ligand_path: PROTAC ligand path (SDF or PDB).
        score_model_args: argparse.Namespace from model_parameters.yml.
        device: 'cuda' or 'cpu'.
        cache_dir: optional dir; caches a CPU copy at ``cache_dir/<case>.pt`` so
            subsequent epochs skip the (expensive) ESM featurization.

    Returns:
        (e3_data, poi_data, ligand_data, receptor_feat_dim) — each data dict is
        in the exact format ``TriCoopDiff.forward`` / ``compute_loss`` expect
        (B=1, on ``device``); ``receptor_feat_dim`` is the receptor feature dim
        (1281 for DiffDock-L).
    """
    for p in (protein1, protein2, ligand_path):
        if p is not None and not Path(p).is_absolute():
            raise ValueError(f"expected absolute path, got: {p}")

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{case_label.replace('/', '_')}.pt"
        if cache_path.exists():
            # weights_only=False: the cache holds numpy arrays (mask_rotate),
            # which PyTorch 2.6's default weights_only=True refuses to unpickle.
            # The file is written by us (trusted), so full unpickling is safe.
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            e3 = _to_device_and_batch(cached['e3_data'], device)
            poi = _to_device_and_batch(cached['poi_data'], device)
            lig = _to_device_and_batch(cached['ligand_data'], device)
            return e3, poi, lig, cached['receptor_feat_dim']

    embeddings_map = generate_esm_embeddings([protein1, protein2], device)

    e3_graph = featurize_subgraph(
        f"{case_label}/e3l", protein1, ligand_path,
        embeddings_map[protein1], score_model_args,
    )
    poi_graph = featurize_subgraph(
        f"{case_label}/poil", protein2, ligand_path,
        embeddings_map[protein2], score_model_args,
    )

    if not e3_graph.success[0] or not poi_graph.success[0]:
        raise RuntimeError(f"preprocessing failed for case {case_label}")

    e3_data = extract_receptor(e3_graph, device)
    poi_data = extract_receptor(poi_graph, device)
    ligand_data = extract_ligand(e3_graph, device)  # ligand identical in both
    receptor_feat_dim = e3_data['features'].shape[-1]

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({
            'e3_data': _strip_batch_and_cpu(e3_data),
            'poi_data': _strip_batch_and_cpu(poi_data),
            'ligand_data': _strip_batch_and_cpu(ligand_data),
            'receptor_feat_dim': receptor_feat_dim,
        }, cache_path)

    return e3_data, poi_data, ligand_data, receptor_feat_dim
