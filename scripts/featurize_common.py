#!/usr/bin/env python
"""
Shared featurization utilities for TriCoopDiff training + smoke test.

Extracts the featurization path that ``scripts/mvp_smoke_test.py`` validated
(receptor/ligand subgraph featurization via DiffDock's ``InferenceDataset`` +
on-the-fly ESM2-650M embeddings) into a single reusable module.

These functions depend on the DiffDock checkout (git-ignored; in-repo
``./DiffDock`` or sibling ``../DiffDock``) for *data featurization only*
(``InferenceDataset`` / ``get_sequences``), which is why they live under
``scripts/`` rather than the self-contained ``tricoopdiff/`` package. The model
itself runs entirely on the vendored ``tricoopdiff.diffdock.*``.
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_diffdock_dir():
    """Locate the DiffDock checkout used for *data featurization only*.

    DiffDock is git-ignored and may be cloned either in-repo (``./DiffDock``,
    the canonical ``.gitignore`` location) or as a sibling (``../DiffDock``).
    Prefer the in-repo clone and fall back to the sibling; both must contain the
    featurization modules ``utils/inference_utils.py`` and
    ``datasets/process_mols.py`` (where ``InferenceDataset`` / ``get_sequences``
    live).
    """
    candidates = (_PROJECT_ROOT / "DiffDock", _PROJECT_ROOT.parent / "DiffDock")
    for d in candidates:
        if (d / "utils" / "inference_utils.py").exists() and \
                (d / "datasets" / "process_mols.py").exists():
            return d
    raise RuntimeError(
        "DiffDock checkout not found. Expected at either "
        f"{candidates[0]} or {candidates[1]} (with utils/inference_utils.py and "
        "datasets/process_mols.py). Clone DiffDock into one of those paths."
    )


_DIFFDOCK_DIR = _resolve_diffdock_dir()

# Make the project root importable so `import tricoopdiff.*` works regardless of
# how this module is invoked (it is not pip-installed).
sys.path.insert(0, str(_PROJECT_ROOT))

# (1) Import diffdock_compat FIRST — its import-time side effect patches
# torch_cluster / torch_scatter with pure-PyTorch fallbacks before the vendored
# cg_model and the DiffDock checkout's process_mols bind those symbols at import
# time.
import tricoopdiff.diffdock_compat  # noqa: F401

# (2) Add the DiffDock checkout to sys.path for data featurization ONLY.
sys.path.insert(0, str(_DIFFDOCK_DIR))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.inference_utils import InferenceDataset, get_sequences

# DiffDock's SO(3)/torus cache lookup resolves relative to CWD; the cache files
# live in the project root. Chdir there so score lookups find them (SLURM also
# does this, but interactive runs may not).
os.chdir(_PROJECT_ROOT)

# Featurization cache schema version. Bump whenever the geometry/frame handling
# of the produced data changes, so stale on-disk caches (written before the
# shared-reference-frame fix) are invalidated instead of silently reused.
_FEATURIZE_CACHE_VERSION = 2


def _protein_ca_centroid(pdb_path):
    """Cα centroid of a protein PDB in its crystal frame.

    Mirrors DiffDock's ``moad_extract_receptor_structure``: the receptor coords
    are all Cα positions (column 1 of ``datasets.parse_chi.get_coords``), and the
    centroid that ``inference_utils`` subtracts is their mean. Returns float32 [3].
    """
    import prody as pr
    from datasets.parse_chi import get_coords

    pdb = pr.parsePDB(pdb_path)
    coords = get_coords(pdb)          # [n_res, 14, 3]; column 1 = Cα
    ca = coords[:, 1, :]              # [n_res, 3]
    center = np.nanmean(ca, axis=0)
    if not np.all(np.isfinite(center)):
        raise RuntimeError(f"could not compute Cα centroid for {pdb_path}")
    return center.astype(np.float32)


def _ligand_heavy_centroid(ligand_path):
    """Heavy-atom centroid of the ligand conformer in its crystal frame.

    The featurizer runs with ``remove_hs=True``, so hydrogens are dropped before
    taking the mean to match the featurized (H-free) ligand. Returns float32 [3].
    """
    from rdkit import Chem

    ext = Path(ligand_path).suffix.lower()
    if ext == ".sdf":
        mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=False)[0]
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(ligand_path, removeHs=False, sanitize=False)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(ligand_path, removeHs=False, sanitize=False)
    else:
        mol = Chem.MolFromMolFile(ligand_path, removeHs=False, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError(f"could not read ligand conformer from {ligand_path}")
    conf = mol.GetConformer()
    # Filter heavy atoms directly (atomic number > 1) to match remove_hs=True,
    # without calling RemoveHs (which can trip on unsanitized PROTACs).
    pos = np.asarray([
        list(conf.GetAtomPosition(a.GetIdx()))
        for a in mol.GetAtoms() if a.GetAtomicNum() > 1
    ], dtype=np.float32)
    if pos.shape[0] == 0:
        raise RuntimeError(f"no heavy atoms found in ligand {ligand_path}")
    return pos.mean(axis=0)


def _true_relative_offsets(protein1, protein2, ligand_path):
    """True inter-molecule translations in the shared crystal frame.

    Returns ``(t_e3_to_poi, lig_offset)``, both float32 [3]:
      * ``t_e3_to_poi`` = POI Cα centroid − E3 Cα centroid  (E3→POI translation)
      * ``lig_offset``   = PROTAC heavy centroid − E3 Cα centroid  (PROTAC in E3 frame)
    """
    e3_center = _protein_ca_centroid(protein1)
    poi_center = _protein_ca_centroid(protein2)
    lig_center = _ligand_heavy_centroid(ligand_path)
    return poi_center - e3_center, lig_center - e3_center


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

        Frame convention (BUG-18): E3 stays at its featurized coords (Cα centroid
        at the origin); POI and PROTAC are re-expressed in that same E3 frame via
        the true crystal-frame translations, so ``model.forward``'s on-the-fly
        ``t_e3_to_poi = poi_center - e3_center`` recovers the real E3→POI
        translation instead of the 0 that ``InferenceDataset``'s per-molecule
        centering would otherwise produce.
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
            if cached.get('cache_version') == _FEATURIZE_CACHE_VERSION:
                e3 = _to_device_and_batch(cached['e3_data'], device)
                poi = _to_device_and_batch(cached['poi_data'], device)
                lig = _to_device_and_batch(cached['ligand_data'], device)
                return e3, poi, lig, cached['receptor_feat_dim']
            # else: stale cache written before the shared-reference-frame fix
            # (BUG-18) — fall through and re-featurize so the corrected geometry
            # is never silently replaced by the old per-molecule-centered one.

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

    # BUG-18 fix: ``InferenceDataset`` independently centers each molecule at its
    # own centroid (E3, POI and PROTAC each end up at the origin), destroying the
    # inter-molecule relative translations — t_e3_to_poi collapses to 0 and the
    # model would learn to denoise everything onto the origin. Recover the true
    # centroids from the raw files and re-express POI + PROTAC in the shared E3
    # reference frame (E3 itself stays at its featurized, centroid-at-origin
    # coords). After this, model.forward's ``t_e3_to_poi = poi_center - e3_center``
    # recovers the true E3→POI translation instead of 0.
    t_e3_to_poi, lig_offset = _true_relative_offsets(protein1, protein2, ligand_path)
    poi_data['coords'] = poi_data['coords'] + torch.tensor(
        t_e3_to_poi, dtype=poi_data['coords'].dtype, device=poi_data['coords'].device
    ).view(1, 1, 3)
    ligand_data['coords'] = ligand_data['coords'] + torch.tensor(
        lig_offset, dtype=ligand_data['coords'].dtype, device=ligand_data['coords'].device
    ).view(1, 1, 3)
    print(f"[data] {case_label}: t_e3_to_poi={t_e3_to_poi.tolist()} "
          f"(|.|={float(np.linalg.norm(t_e3_to_poi)):.2f} Å)", flush=True)

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({
            'e3_data': _strip_batch_and_cpu(e3_data),
            'poi_data': _strip_batch_and_cpu(poi_data),
            'ligand_data': _strip_batch_and_cpu(ligand_data),
            'receptor_feat_dim': receptor_feat_dim,
            't_e3_to_poi': torch.tensor(t_e3_to_poi, dtype=torch.float32),
            'cache_version': _FEATURIZE_CACHE_VERSION,
        }, cache_path)

    return e3_data, poi_data, ligand_data, receptor_feat_dim
