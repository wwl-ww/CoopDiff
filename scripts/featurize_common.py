#!/usr/bin/env python
"""
Shared production featurization for TriCoopDiff training and evaluation.

Builds receptor/ligand subgraphs through DiffDock's ``InferenceDataset`` and
generates ESM2-650M embeddings on demand.

The featurization modules (``InferenceDataset`` / ``get_sequences`` /
``datasets.process_mols``) are vendored under ``tricoopdiff.diffdock``
(``.datasets`` + ``.utils.inference_utils``), so these functions import them via
``tricoopdiff.diffdock.*`` and no longer need the external DiffDock checkout.
They still live under ``scripts/`` rather than ``tricoopdiff/`` because they mix
project data paths (``DeepTernary/output/...``) with the vendored featurization.
"""

import hashlib
import json
import os
import sys
import gc
import time
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Make the project root importable so `import tricoopdiff.*` works regardless of
# how this module is invoked (it is not pip-installed).
sys.path.insert(0, str(_PROJECT_ROOT))

# Import diffdock_compat FIRST — its import-time side effect patches
# torch_cluster / torch_scatter with pure-PyTorch fallbacks before the vendored
# cg_model and the vendored process_mols bind those symbols at import time.
import tricoopdiff.diffdock_compat  # noqa: F401
from tricoopdiff.runtime import atomic_torch_save

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tricoopdiff.diffdock.utils.inference_utils import InferenceDataset, get_sequences

# DiffDock's SO(3)/torus cache lookup resolves relative to CWD; the cache files
# live in the project root. Chdir there so score lookups find them (SLURM also
# does this, but interactive runs may not).
os.chdir(_PROJECT_ROOT)

# Featurization cache schema version. Bump whenever the geometry/frame handling
# of the produced data changes, so stale on-disk caches (written before the
# shared-reference-frame fix) are invalidated instead of silently reused.
# v6 fixes the directed k-NN convention and small-receptor self loops.
# v7 also avoids distance cancellation for translated PDB coordinates.
# v8 centers the exact retained ligand atoms, including isotope hydrogens.
# v9 adds intrinsic monomer backbone/sidechain geometry (v8 can be enriched).
# v10 rebuilds legal torsion masks from a bond-order-bearing molecular graph.
# PDB v9 is a tagged historical compatibility path, never a new cache writer.
# v5 caches contain reversed directed receptor edges and cannot be reused.
_FEATURIZE_CACHE_VERSION = 10
# PDB v9 caches are read only for explicitly identified historical experiments.
# New chemistry must use a complete bond-order-bearing SDF/MOL source.
def _source_cache_version(ligand_path):
    return 9 if Path(ligand_path).suffix.lower() == ".pdb" else _FEATURIZE_CACHE_VERSION


LOCAL_GEOMETRY_DIM = 32


def _encoder_file_digest(path):
    """Hash contents directly: filesystem timestamps may have coarse resolution."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def esm_encoder_identity():
    """Resolve only local assets; never instantiate or download an encoder."""
    model = os.environ.get('TRICOOPDIFF_ESM_MODEL') or 'facebook/esm2_t33_650M_UR50D'
    directory = Path(model).expanduser()
    if not directory.is_dir():
        if directory.is_absolute():
            directory = None
        else:
            from huggingface_hub import try_to_load_from_cache
            from huggingface_hub.errors import HFValidationError
            try:
                config = try_to_load_from_cache(model, 'config.json')
            except (HFValidationError, OSError):
                config = None
            directory = Path(config).parent if isinstance(config, str) else None
    files = {}
    if directory is not None and directory.is_dir():
        for path in sorted(directory.rglob('*')):
            if path.is_file() and (path.suffix in ('.json', '.txt', '.model', '.bin', '.safetensors')
                                   or path.name in ('merges.txt', 'vocab.txt')):
                files[str(path.relative_to(directory))] = _encoder_file_digest(str(path.resolve()))
    weights = any(name.endswith('.safetensors') or Path(name).name.startswith('pytorch_model')
                  and name.endswith('.bin') for name in files)
    tokenizer = any(Path(name).name in ('tokenizer.json', 'vocab.txt', 'tokenizer.model') for name in files)
    shards_complete = True
    if directory is not None:
        for name in files:
            if name.endswith('.index.json'):
                try:
                    index = json.loads((directory / name).read_text())
                    shard_names = set(index['weight_map'].values())
                    shards_complete = shards_complete and bool(shard_names) and all(
                        shard in files for shard in shard_names)
                except (KeyError, TypeError, ValueError, OSError):
                    shards_complete = False
    complete = 'config.json' in files and weights and tokenizer and shards_complete
    return {'model': model, 'status': 'local_assets_hashed' if complete else 'unresolved_local_assets',
            'files': files}


def _esm_identity_verified(identity):
    return isinstance(identity, dict) and identity.get('status') == 'local_assets_hashed'


def _accept_legacy_esm_cache(cached, expected_signature):
    """Keep old research assets usable, explicitly exposing unverified ESM origin."""
    if cached.get('esm_encoder_identity') is not None:
        return False
    if cached.get('source_signature') != expected_signature:
        return False
    if os.environ.get('TRICOOPDIFF_ALLOW_LEGACY_ESM_CACHE', '1') != '1':
        return False
    warnings.warn('Using legacy feature cache: ESM encoder identity is unverified. '
                  'Set TRICOOPDIFF_ALLOW_LEGACY_ESM_CACHE=0 to require verified assets.',
                  RuntimeWarning, stacklevel=2)
    cached['esm_cache_status'] = 'legacy_encoder_unverified'
    return True


def _featurize_signature(protein1, protein2, ligand_path, score_model_args,
                         cache_version='auto', include_esm=True, encoder_identity=None):
    """Fingerprint source structures and model preprocessing arguments."""
    if cache_version == 'auto':
        cache_version = _source_cache_version(ligand_path)
    digest = hashlib.sha256()
    for path in (protein1, protein2, ligand_path):
        resolved = Path(path).resolve()
        digest.update(str(resolved).encode())
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(
        json.dumps(vars(score_model_args), sort_keys=True, default=str).encode()
    )
    if cache_version is not None:
        digest.update(f'featurize_schema_v{cache_version}'.encode())
    if include_esm:
        digest.update(json.dumps(esm_encoder_identity() if encoder_identity is None else encoder_identity, sort_keys=True).encode())
    return digest.hexdigest()


def protein_local_geometry(pdb_path, receptor):
    """Intrinsic monomer scalars [1,N,32], in exactly DiffDock's residue order.

    Ten summaries (N/CA, C/CA, O/CA distances, N-CA-C cosine,
    sidechain count/mean/max radius and centroid radius/N/C cosines), their
    validity masks, then four chi sines, cosines and masks. No partner pose is
    used. Missing atoms are masked, never confused with measured zero angles.
    """
    import prody as pr
    from tricoopdiff.diffdock.datasets.parse_chi import get_coords, get_chi_angles

    pdb = pr.parsePDB(str(pdb_path))
    atoms = get_coords(pdb)
    ca = atoms[:, 1]
    reference = receptor['coords'].detach().cpu().numpy().reshape(-1, 3)
    if (ca.shape != reference.shape or not np.isfinite(ca).all()
            or not np.allclose(ca - ca.mean(0), reference - reference.mean(0), atol=.01, rtol=1e-4)):
        raise ValueError(f'{pdb_path}: local geometry CA ordering does not match receptor graph')
    relative = atoms - ca[:, None]
    valid = np.isfinite(relative).all(-1)
    relative = np.nan_to_num(relative, nan=0., posinf=0., neginf=0.)
    radius = np.linalg.norm(relative, axis=-1)
    unit = relative / np.maximum(radius[..., None], 1e-12)
    side_mask = valid[:, 4:]
    count = side_mask.sum(1)
    centroid = (relative[:, 4:] * side_mask[..., None]).sum(1) / np.maximum(count[:, None], 1)
    centroid_radius = np.linalg.norm(centroid, axis=-1)
    centroid_unit = centroid / np.maximum(centroid_radius[:, None], 1e-12)
    values = np.stack((radius[:, 0] / 2, radius[:, 2] / 2, radius[:, 3] / 3,
        (unit[:, 0] * unit[:, 2]).sum(-1), count / 10,
        (radius[:, 4:] * side_mask).sum(1) / np.maximum(count, 1) / 5,
        (radius[:, 4:] * side_mask).max(1) / 10, centroid_radius / 5,
        (centroid_unit * unit[:, 0]).sum(-1), (centroid_unit * unit[:, 2]).sum(-1)), axis=-1)
    present = valid & (radius > 1e-10)
    side_present = count > 0
    masks = np.stack((present[:, 0], present[:, 2], present[:, 3],
        present[:, 0] & present[:, 2], np.ones(len(ca), dtype=bool), side_present,
        side_present, side_present, side_present & present[:, 0] & (centroid_radius > 1e-10),
        side_present & present[:, 2] & (centroid_radius > 1e-10)), axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        angles = np.deg2rad(get_chi_angles(atoms, pdb.ca.getSequence()))
    angle_mask = np.isfinite(angles)
    angles = np.nan_to_num(angles)
    features = np.concatenate((values * masks, masks, np.sin(angles) * angle_mask,
                               np.cos(angles) * angle_mask, angle_mask), axis=-1)
    if features.shape != (len(ca), LOCAL_GEOMETRY_DIM) or not np.isfinite(features).all():
        raise ValueError(f'{pdb_path}: invalid intrinsic protein geometry')
    return torch.as_tensor(features, device=receptor['coords'].device,
                           dtype=receptor['coords'].dtype).unsqueeze(0)


def _protein_ca_centroid(pdb_path):
    """Cα centroid of a protein PDB in its crystal frame.

    Mirrors DiffDock's ``moad_extract_receptor_structure``: the receptor coords
    are all Cα positions (column 1 of ``datasets.parse_chi.get_coords``), and the
    centroid that ``inference_utils`` subtracts is their mean. Returns float32 [3].
    """
    import prody as pr
    from tricoopdiff.diffdock.datasets.parse_chi import get_coords

    pdb = pr.parsePDB(pdb_path)
    coords = get_coords(pdb)          # [n_res, 14, 3]; column 1 = Cα
    ca = coords[:, 1, :]              # [n_res, 3]
    center = np.nanmean(ca, axis=0)
    if not np.all(np.isfinite(center)):
        raise RuntimeError(f"could not compute Cα centroid for {pdb_path}")
    return center.astype(np.float32)


def _ligand_centroid(ligand_path, remove_hs=True):
    """Centroid of exactly the atoms retained by graph construction, float32 [3]."""
    from tricoopdiff.diffdock.datasets.process_mols import read_molecule

    mol = read_molecule(str(ligand_path), sanitize=True, remove_hs=remove_hs)
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError(f"could not read ligand conformer from {ligand_path}")
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    if pos.shape[0] == 0 or not np.isfinite(pos).all():
        raise RuntimeError(f"invalid ligand coordinates in {ligand_path}")
    return pos.mean(axis=0)


def _true_relative_offsets(protein1, protein2, ligand_path, remove_hs=True):
    """True inter-molecule translations in the shared crystal frame.

    Returns ``(t_e3_to_poi, lig_offset)``, both float32 [3]:
      * ``t_e3_to_poi`` = POI Cα centroid − E3 Cα centroid  (E3→POI translation)
      * ``lig_offset``   = retained ligand centroid − E3 Cα centroid
    """
    e3_center = _protein_ca_centroid(protein1)
    poi_center = _protein_ca_centroid(protein2)
    lig_center = _ligand_centroid(ligand_path, remove_hs)
    return poi_center - e3_center, lig_center - e3_center


def generate_esm_embeddings(protein_paths, device):
    """Generate ESM2-650M embeddings for protein PDBs.

    Returns: dict {protein_path: [per-chain torch.Tensor (n_res, 1280)]}.
    """
    from tricoopdiff.inputs.esm_embedder import ESMEmbedder

    print(f"Loading ESM2-650M on {device} via HF mirror...", flush=True)
    local_model = os.environ.get("TRICOOPDIFF_ESM_MODEL")
    embedder = (
        ESMEmbedder(model_name=local_model, device=device)
        if local_model else ESMEmbedder(device=device)
    )

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

    result = {
        path: [seq_to_emb[s].cpu() for s in chain_seqs]
        for path, chain_seqs in per_protein_info
    }
    # This helper is called lazily once per uncached training case. ESM2-650M is
    # much larger than the score model, so retaining one allocator block per
    # call eventually exhausts a single GPU even with B=1.
    del embedder, unique_embeddings, seq_to_emb
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


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
        # These are crystallographic training/evaluation SDFs. The generic
        # DiffDock inference path regenerates file conformers by default, which
        # destroys the ternary pose even when its centroid is restored later.
        preserve_ligand_conformer=True,
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
    # Rebuild axes and masks together from chemical bonds; never reuse the
    # vendored topology-only torsion map.
    from rdkit import Chem

    mol = batch_graph.mol[0] if isinstance(batch_graph.mol, (list, tuple)) else batch_graph.mol
    if mol.GetNumAtoms() != batch_graph['ligand'].pos.shape[0]:
        mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() != batch_graph['ligand'].pos.shape[0]:
        raise ValueError("RDKit atom order/count does not match ligand graph")
    from tricoopdiff.inputs.ligand_chemistry import CHEMISTRY_CONTRACT, legal_torsion_masks
    edge_index = batch_graph['ligand', 'ligand'].edge_index
    edge_mask, mask_rotate = legal_torsion_masks(mol, edge_index)
    mol.SetProp('tricoopdiff_chemistry_contract', CHEMISTRY_CONTRACT)
    return {
        'chemistry_contract': CHEMISTRY_CONTRACT,
        'mol': mol.ToBinary(Chem.PropertyPickleOptions.AllProps),
        'coords': batch_graph['ligand'].pos.unsqueeze(0).to(device),
        'features': batch_graph['ligand'].x.unsqueeze(0).to(device),
        'edge_index': batch_graph['ligand', 'ligand'].edge_index.to(device),
        'edge_attr': batch_graph['ligand', 'ligand'].edge_attr.to(device),
        'edge_mask': torch.as_tensor(edge_mask, device=device),
        'mask_rotate': mask_rotate,
    }


def _strip_batch_and_cpu(data):
    """Convert a batched device dict → unbatched CPU dict (on-disk cache format)."""
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            v = v.detach().cpu()
        if k in ("coords", "features", "local_geometry"):
            v = v[0]  # drop the leading B=1 dim
        out[k] = v
    return out


def _to_device_and_batch(data, device):
    """Reverse of ``_strip_batch_and_cpu``: CPU cache dict → batched device dict."""
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            v = v.to(device)
        if k in ("coords", "features", "local_geometry"):
            v = v.unsqueeze(0)  # re-add the leading B=1 dim
        out[k] = v
    return out


def featurize_complex(case_label, protein1, protein2, ligand_path,
                      score_model_args, device, cache_dir=None,
                      cache_read_only=False, cache_wait_seconds=3600., cache_only=False,
                      expected_esm_encoder_identity=None):
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
    # Read-only runs may pin a content-hashed encoder identity at startup.
    # These workers never load the encoder; each cache still proves its source
    # and encoder identity against the run's immutable expected provenance.
    if expected_esm_encoder_identity is not None:
        if not (cache_read_only or cache_only):
            raise ValueError('pinned encoder identity requires read-only feature loading')
        if not _esm_identity_verified(expected_esm_encoder_identity):
            raise ValueError('pinned encoder identity must be verified from local assets')
    for p in (protein1, protein2, ligand_path):
        if p is not None and not Path(p).is_absolute():
            raise ValueError(f"expected absolute path, got: {p}")

    cache_version = _source_cache_version(ligand_path)
    cache_path = None
    legacy_cache_signature = _featurize_signature(
        protein1, protein2, ligand_path, score_model_args, include_esm=False
    )
    cache_signature = legacy_cache_signature
    if cache_wait_seconds < 0:
        raise ValueError('cache_wait_seconds must be nonnegative')
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{case_label.replace('/', '_')}.pt"
        deadline = time.monotonic() + cache_wait_seconds
        while True:
            if cache_path.exists():
                # weights_only=False: the cache holds numpy arrays (mask_rotate),
                # which PyTorch 2.6's default weights_only=True refuses to unpickle.
                # The file is written by us (trusted), so full unpickling is safe.
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                cache_signature = legacy_cache_signature
                runtime_identity = None
                if cached.get('esm_encoder_identity') is not None:
                    runtime_identity = (expected_esm_encoder_identity
                                        if expected_esm_encoder_identity is not None
                                        else esm_encoder_identity())
                    cache_signature = _featurize_signature(
                        protein1, protein2, ligand_path, score_model_args,
                        encoder_identity=runtime_identity)
                # v8 already has correct atom order/frames/ESM. Enrich it from
                # source monomers without rebuilding embeddings or graphs.
                if (expected_esm_encoder_identity is None
                        and cache_version == 9
                        and cached.get('cache_version') in (7, 8)
                        and os.environ.get('TRICOOPDIFF_ALLOW_LEGACY_ESM_CACHE', '1') == '1'
                        and cached.get('esm_encoder_identity') is None
                        and cached.get('source_signature') ==
                        _featurize_signature(protein1, protein2, ligand_path, score_model_args,
                                             cache_version=None, include_esm=False)):
                    if cached.get('esm_encoder_identity') is None:
                        if not _accept_legacy_esm_cache(cached, cached['source_signature']):
                            raise ValueError(f'{case_label}: legacy ESM cache disabled by strict policy')
                    if cached['cache_version'] == 7:
                        # v7 differed only in retained/isotope-H ligand centering.
                        # Correct that translation from source, retaining ESM and topology.
                        position = cached['ligand_data']['coords']
                        center = _ligand_centroid(ligand_path, remove_hs=score_model_args.remove_hs) - _protein_ca_centroid(protein1)
                        cached['ligand_data']['coords'] = position - position.mean(0) + position.new_tensor(center)
                    for key, path in (('e3_data', protein1), ('poi_data', protein2)):
                        cached[key]['local_geometry'] = protein_local_geometry(path, cached[key])[0].cpu()
                    cached.update(cache_version=cache_version, source_signature=(
                        _featurize_signature(protein1, protein2, ligand_path, score_model_args, include_esm=False)
                        if cached.get('esm_cache_status') == 'legacy_encoder_unverified' else cache_signature))
                    if not cache_read_only:
                        atomic_torch_save(cached, cache_path)
                if (
                    cached.get('cache_version') == cache_version
                    and ((_esm_identity_verified(runtime_identity) and
                          cached.get('esm_encoder_identity') == runtime_identity and
                          cached.get('source_signature') == cache_signature) or
                         (expected_esm_encoder_identity is None and
                          _accept_legacy_esm_cache(cached, cache_signature)))
                ):
                    for key in ('e3_data', 'poi_data'):
                        geometry = cached[key].get('local_geometry')
                        if (not torch.is_tensor(geometry) or geometry.shape !=
                                (cached[key]['coords'].shape[0], LOCAL_GEOMETRY_DIM)
                                or not torch.isfinite(geometry).all()):
                            raise ValueError(f'{case_label}: cache lacks valid {key} local_geometry')
                    if cache_version == 9:
                        from rdkit import Chem
                        from tricoopdiff.inputs.ligand_chemistry import LEGACY_CHEMISTRY_CONTRACT
                        cached['ligand_data'] = dict(cached['ligand_data'])
                        mol = Chem.Mol(cached['ligand_data']['mol'])
                        mol.SetProp('tricoopdiff_chemistry_contract', LEGACY_CHEMISTRY_CONTRACT)
                        cached['ligand_data']['mol'] = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
                        cached['ligand_data']['chemistry_contract'] = LEGACY_CHEMISTRY_CONTRACT
                    elif cached['ligand_data'].get('chemistry_contract') != 'bond_order_legal_torsions_v1':
                        raise ValueError(f'{case_label}: corrected cache lacks chemistry contract')
                    e3 = _to_device_and_batch(cached['e3_data'], device)
                    poi = _to_device_and_batch(cached['poi_data'], device)
                    lig = _to_device_and_batch(cached['ligand_data'], device)
                    return e3, poi, lig, cached['receptor_feat_dim']
            if cache_only:
                raise FileNotFoundError(f'{case_label}: no compatible feature cache; ESM rebuilding disabled')
            if not cache_read_only:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f'{case_label}: read-only featurization follower timed out waiting for writer')
            time.sleep(min(.1, max(0., deadline - time.monotonic())))
            # A stale cache can be atomically replaced by the writer while the
            # follower waits.  It must never featurize or publish independently.

    if cache_only:
        raise FileNotFoundError(f'{case_label}: cache-only mode requires a compatible cache directory')
    if cache_version == 9:
        raise ValueError('PDB ligand chemistry is ambiguous: prepare a CCD-corrected SDF index; '
                         'legacy v9 caches may only be reused, never rebuilt')
    encoder_identity = esm_encoder_identity()
    if not _esm_identity_verified(encoder_identity):
        raise ValueError('ESM local encoder assets are incomplete; cannot record verified cache identity')
    cache_signature = _featurize_signature(
        protein1, protein2, ligand_path, score_model_args, encoder_identity=encoder_identity)
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
    e3_data['local_geometry'] = protein_local_geometry(protein1, e3_data)
    poi_data['local_geometry'] = protein_local_geometry(protein2, poi_data)
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
    t_e3_to_poi, lig_offset = _true_relative_offsets(
        protein1, protein2, ligand_path, remove_hs=score_model_args.remove_hs)
    poi_data['coords'] = poi_data['coords'] + torch.tensor(
        t_e3_to_poi, dtype=poi_data['coords'].dtype, device=poi_data['coords'].device
    ).view(1, 1, 3)
    ligand_data['coords'] = ligand_data['coords'] + torch.tensor(
        lig_offset, dtype=ligand_data['coords'].dtype, device=ligand_data['coords'].device
    ).view(1, 1, 3)
    print(f"[data] {case_label}: t_e3_to_poi={t_e3_to_poi.tolist()} "
          f"(|.|={float(np.linalg.norm(t_e3_to_poi)):.2f} Å)", flush=True)

    if cache_path is not None:
        payload = {
            'e3_data': _strip_batch_and_cpu(e3_data),
            'poi_data': _strip_batch_and_cpu(poi_data),
            'ligand_data': _strip_batch_and_cpu(ligand_data),
            'receptor_feat_dim': receptor_feat_dim,
            't_e3_to_poi': torch.tensor(t_e3_to_poi, dtype=torch.float32),
            'cache_version': cache_version,
            'source_signature': cache_signature,
            'esm_encoder_identity': encoder_identity,
            'esm_cache_status': 'encoder_identity_recorded',
        }
        atomic_torch_save(payload, cache_path)

    return e3_data, poi_data, ligand_data, receptor_feat_dim
