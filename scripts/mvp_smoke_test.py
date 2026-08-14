#!/usr/bin/env python
"""
MVP integration smoke test for TriCoopDiff.

Runs the full pipeline ONCE on a single real ternary complex:

  1. Featurize the E3-L and POI-L two-body subgraphs (DiffDock's
     ``InferenceDataset`` + on-the-fly ESM2-650M embeddings).
  2. Build the frozen DiffDock-L expert + TriBrain + TriCoopDiff +
     MultiRateDiffusion + TriCoopDiffTrainer.
  3. **GATE 1** — zero-FiLM transparency: ``zero_film_verification`` asserts the
     zero-initialized FiLM path reproduces the native expert output.
  4. **GATE 2** — gradient flow: one ``compute_loss`` + ``backward()`` must
     produce finite, non-None gradients on the FiLM heads and the SE3 head
     (while the frozen expert stays grad-free).

Exit code 0 = both gates pass; 1 = any failure.

The model + sampling run entirely on the vendored ``tricoopdiff.diffdock.*``.
``../DiffDock`` is only imported for *data featurization*
(``InferenceDataset`` / ``get_sequences``), which stays out of the vendor scope.
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DIFFDOCK_DIR = _PROJECT_ROOT.parent / "DiffDock"

# Make the project root importable so `import tricoopdiff.*` works whether this
# script is run as `python scripts/mvp_smoke_test.py` (sys.path[0] = scripts/) or
# from the root. `tricoopdiff` is not pip-installed (no setup.py / pyproject).
sys.path.insert(0, str(_PROJECT_ROOT))

# (1) Import diffdock_compat FIRST — its import-time side effect patches
# torch_cluster / torch_scatter with pure-PyTorch fallbacks before the vendored
# cg_model and ../DiffDock's process_mols bind those symbols at import time.
# NOTE: importing any `tricoopdiff.*` submodule runs tricoopdiff/__init__.py,
# which itself imports diffdock_compat before model/graphs/diffusion, so the
# patch is guaranteed to be in place for tribrain.py / expert.py too.
import tricoopdiff.diffdock_compat  # noqa: F401

# (2) Add ../DiffDock to sys.path for data featurization ONLY.
if not _DIFFDOCK_DIR.exists():
    print(f"FATAL: DiffDock checkout not found at {_DIFFDOCK_DIR}", flush=True)
    sys.exit(1)
sys.path.insert(0, str(_DIFFDOCK_DIR))

import torch
import yaml
from argparse import Namespace
from torch_geometric.loader import DataLoader

from utils.inference_utils import InferenceDataset, get_sequences

from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.tribrain import TriBrain
from tricoopdiff.model import TriCoopDiff, zero_film_verification
from tricoopdiff.train import TriCoopDiffTrainer
from tricoopdiff.diffusion import MultiRateDiffusion
from tricoopdiff.graphs import build_subgraph_e3l, build_tribrain_graph

# DiffDock's SO(3) cache lookup resolves relative to CWD; the cache files live
# in the project root. Chdir there so `so3_score_norm`/`so3_score_vec` find them
# (SLURM also does this, but interactive runs may not).
os.chdir(_PROJECT_ROOT)


def build_parser():
    p = argparse.ArgumentParser(description="TriCoopDiff MVP integration smoke test")
    p.add_argument("--protac-dir", default="DeepTernary/output/protac22")
    p.add_argument("--model-dir", default="checkpoints/score_model")
    p.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt")
    p.add_argument("--case", default="5T35_D_A_759",
                   help="Ternary complex case name (default: first available)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-5,
                   help="Absolute tolerance for zero-FiLM transparency gate")
    return p


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


def featurize_subgraph(case_label, protein_path, ligand_path, chain_embs, score_model_args):
    """Featurize one two-body subgraph and return its Batch graph."""
    precomputed = [[t for t in chain_embs]]  # keep on CPU
    dataset = InferenceDataset(
        out_dir=str(_PROJECT_ROOT / "results" / "smoke_tmp" / case_label.replace("/", "_")),
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
    return {
        'coords': batch_graph['ligand'].pos.unsqueeze(0).to(device),
        'features': batch_graph['ligand'].x.unsqueeze(0).to(device),
        'edge_index': batch_graph['ligand', 'ligand'].edge_index.to(device),
        'edge_attr': batch_graph['ligand', 'ligand'].edge_attr.to(device),
        'edge_mask': batch_graph['ligand'].edge_mask.to(device),
    }


def _grad_ok(param):
    """A parameter is 'grad-ok' iff it has a non-None, finite gradient."""
    if param.grad is None:
        return False
    return bool(torch.isfinite(param.grad).all().item())


def main():
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda requested but CUDA is unavailable", flush=True)
        sys.exit(1)
    device = args.device

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Resolve the case directory.
    protac_dir = _PROJECT_ROOT / args.protac_dir
    case_dir = protac_dir / args.case
    if not case_dir.exists():
        print(f"FATAL: case directory not found: {case_dir}", flush=True)
        sys.exit(1)

    protein1 = str(case_dir / "protein1.pdb")   # E3
    protein2 = str(case_dir / "protein2.pdb")   # POI
    _sdf = case_dir / "ligand.sdf"
    _pdb = case_dir / "ligand.pdb"
    ligand_path = str(_sdf if _sdf.exists() else _pdb)

    print("=" * 72, flush=True)
    print(f"TriCoopDiff MVP smoke test — case {args.case}", flush=True)
    print(f"  device={device}  model-dir={args.model_dir}", flush=True)
    print(f"  protein1(E3)={protein1}", flush=True)
    print(f"  protein2(POI)={protein2}", flush=True)
    print(f"  ligand={ligand_path}", flush=True)
    print("=" * 72, flush=True)

    # --- Load score-model args (needed for InferenceDataset hyperparams) ------
    with open(_PROJECT_ROOT / args.model_dir / "model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))
    print(f"[cfg] ns={score_model_args.ns} nv={score_model_args.nv} "
          f"num_conv_layers={score_model_args.num_conv_layers}", flush=True)

    # --- Build the frozen expert (vendored core, strict weight load) ---------
    print("[expert] building frozen DiffDock-L expert...", flush=True)
    expert = build_diffdock_expert(
        model_dir=str(_PROJECT_ROOT / args.model_dir),
        device=device,
        ckpt=args.ckpt,
        num_conv_layers=score_model_args.num_conv_layers,
    )
    num_conv_layers = expert.num_conv_layers
    print(f"[expert] loaded (num_conv_layers={num_conv_layers}, "
          f"ns={expert.ns}) — all params frozen", flush=True)

    # --- Featurize both subgraphs (on-the-fly ESM2) --------------------------
    print("[data] generating ESM embeddings for E3 + POI...", flush=True)
    embeddings_map = generate_esm_embeddings(
        [protein1, protein2], device
    )

    print("[data] featurizing E3-L subgraph...", flush=True)
    e3_graph = featurize_subgraph(
        f"{args.case}/e3l", protein1, ligand_path,
        embeddings_map[protein1], score_model_args,
    )
    print("[data] featurizing POI-L subgraph...", flush=True)
    poi_graph = featurize_subgraph(
        f"{args.case}/poil", protein2, ligand_path,
        embeddings_map[protein2], score_model_args,
    )

    if not e3_graph.success[0] or not poi_graph.success[0]:
        print("FATAL: preprocessing failed for one of the subgraphs", flush=True)
        sys.exit(1)

    e3_data = extract_receptor(e3_graph, device)
    poi_data = extract_receptor(poi_graph, device)
    ligand_data = extract_ligand(e3_graph, device)  # ligand is identical in both

    n_e3, d_rec = e3_data['features'].shape[1:]
    n_poi = poi_data['features'].shape[1]
    n_lig, d_lig = ligand_data['features'].shape[1:]
    n_rot = int(ligand_data['edge_mask'].sum().item())
    print(f"[data] E3 residues={n_e3}  POI residues={n_poi}  "
          f"PROTAC atoms={n_lig}  rotatable bonds={n_rot}", flush=True)
    print(f"[data] receptor feat dim={d_rec}  ligand feat dim={d_lig}", flush=True)

    # --- Build TriBrain + TriCoopDiff + diffusion + trainer ------------------
    # TriBrain input_dim must match the (padded) ternary node feature dim, which
    # is the receptor feature dim (1318 = 38 residue + 1280 ESM2 for DiffDock-L).
    tribrain = TriBrain(
        node_dim=128,
        input_dim=d_rec,
        num_conv_layers_per_expert=num_conv_layers,
        film_dim=expert.ns,  # 48 — must equal the expert's scalar channel count
    )
    model = TriCoopDiff(
        frozen_expert=expert,
        tribrain=tribrain,
        num_conv_layers=num_conv_layers,
        use_film=True,
        use_learned_merge=False,
    )
    diffusion = MultiRateDiffusion()
    trainer = TriCoopDiffTrainer(model=model, diffusion=diffusion, device=device)
    n_params = sum(p.numel() for p in tribrain.parameters())
    print(f"[model] TriBrain trainable params = {n_params:,}", flush=True)

    # ======================================================================
    # GATE 1 — zero-FiLM transparency
    # ======================================================================
    print("\n" + "=" * 72, flush=True)
    print("GATE 1: zero-FiLM transparency", flush=True)
    print("=" * 72, flush=True)

    sub1 = build_subgraph_e3l(
        e3_coords=e3_data['coords'][0],
        e3_features=e3_data['features'][0],
        ligand_coords=ligand_data['coords'][0],
        ligand_features=ligand_data['features'][0],
        rec_edge_index=e3_data['edge_index'],
        lig_edge_index=ligand_data['edge_index'],
        lig_edge_attr=ligand_data['edge_attr'],
        lig_edge_mask=ligand_data['edge_mask'],
        lig_mask_rotate=None,
    )
    tri_graph = build_tribrain_graph(
        e3_coords=e3_data['coords'][0],
        e3_features=e3_data['features'][0],
        poi_coords=poi_data['coords'][0],
        poi_features=poi_data['features'][0],
        ligand_coords=ligand_data['coords'][0],
        ligand_features=ligand_data['features'][0],
    )

    gate1_passed, gate1_maxdiff = zero_film_verification(
        expert, sub1, tribrain, tri_graph,
        t=torch.tensor([0.5], device=device), atol=args.atol,
    )
    if gate1_passed:
        print(f"GATE 1 PASS (max_abs_diff={gate1_maxdiff:.3e} < atol={args.atol})",
              flush=True)
    else:
        print(f"GATE 1 FAIL (max_abs_diff={gate1_maxdiff:.3e} >= atol={args.atol})",
              flush=True)

    # ======================================================================
    # GATE 2 — gradient flow to TriBrain (FiLM + SE3 heads), expert frozen
    # ======================================================================
    print("\n" + "=" * 72, flush=True)
    print("GATE 2: finite gradients to TriBrain", flush=True)
    print("=" * 72, flush=True)

    loss_dict = trainer.compute_loss(e3_data, poi_data, ligand_data, None)
    print(f"[loss] total={loss_dict['loss_total'].item():.4f}  "
          f"score={loss_dict['loss_score']:.4f}  "
          f"se3={loss_dict['loss_se3']:.4f}  "
          f"phys={loss_dict['loss_phys']:.4f}", flush=True)

    loss_dict['loss_total'].backward()

    film_grad_ok = _grad_ok(tribrain.film_heads[0].net.weight)
    se3_grad_ok = _grad_ok(tribrain.se3_head.net[-1].weight)

    # Frozen expert must stay grad-free.
    expert_has_grad = any(p.grad is not None for p in expert.model.parameters())

    print(f"[grad] FiLM head[0] grad: "
          f"{'finite' if film_grad_ok else 'MISSING/NON-FINITE'}", flush=True)
    print(f"[grad] SE3 head grad:   "
          f"{'finite' if se3_grad_ok else 'MISSING/NON-FINITE'}", flush=True)
    print(f"[grad] frozen expert has grad: {expert_has_grad} (must be False)",
          flush=True)

    gate2_passed = film_grad_ok and se3_grad_ok and not expert_has_grad
    if gate2_passed:
        print("GATE 2 PASS", flush=True)
    else:
        print("GATE 2 FAIL", flush=True)

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 72, flush=True)
    overall = gate1_passed and gate2_passed
    print(f"SMOKE TEST: {'PASS' if overall else 'FAIL'}", flush=True)
    print("=" * 72, flush=True)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
