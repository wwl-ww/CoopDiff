#!/usr/bin/env python3
"""
Experiment 3: Zero-OOD Verification

Verify that FrozenDiffDock produces numerically identical output to native
DiffDock CGModel on the same input.

Usage:
    python scripts/experiment3_ood_verify.py --case <CASE_DIR> [--device cuda]
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from argparse import Namespace

# Add DiffDock to path (before importing its modules)
_DIFFDOCK = str(Path(__file__).parent.parent.parent / "DiffDock")
if _DIFFDOCK not in sys.path:
    sys.path.insert(0, _DIFFDOCK)

# Add CoopDiff to path
_COOPDIFF = str(Path(__file__).parent.parent)
if _COOPDIFF not in sys.path:
    sys.path.insert(0, _COOPDIFF)

from torch_geometric.loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Zero-OOD Verification")
    parser.add_argument("--case", type=str, default=None,
                        help="Path to PROTAC benchmark case")
    parser.add_argument("--model-dir", type=str,
                        default="checkpoints/score_model")
    parser.add_argument("--ckpt", type=str,
                        default="best_ema_inference_epoch_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str,
                        default="results/ood_verification.txt")
    args = parser.parse_args()

    # Resolve case directory
    if args.case:
        case_dir = Path(args.case)
        protein1_path = str(case_dir / "protein1.pdb")
        ligand_path = str(case_dir / "ligand.sdf")
    else:
        case_dir = Path("DeepTernary/output/protac22/5T35_D_A_759")
        protein1_path = str(case_dir / "protein1.pdb")
        ligand_path = str(case_dir / "ligand.sdf")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Protein: {protein1_path}", flush=True)
    print(f"Ligand:  {ligand_path}", flush=True)

    # ================================================================
    # Step 1: Load model via DiffDock's standard pipeline
    # ================================================================
    print("=== Step 1: Loading DiffDock-L model ===", flush=True)
    model_dir = args.model_dir
    with open(f"{model_dir}/model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))

    print(f"  ns={score_model_args.ns}, nv={score_model_args.nv}, "
          f"num_conv_layers={score_model_args.num_conv_layers}, "
          f"sh_lmax={score_model_args.sh_lmax}", flush=True)

    from utils.utils import get_model
    from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl

    t_to_sigma = lambda *x: t_to_sigma_compl(*x, args=score_model_args)
    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma,
                      no_parallel=True, old=False)
    state_dict = torch.load(f"{model_dir}/{args.ckpt}",
                            map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully.", flush=True)

    # ================================================================
    # Step 2: Preprocess test complex (use lm_embeddings=False to
    #         avoid ESMFold model download on restricted network)
    # ================================================================
    print("=== Step 2: Preprocessing test complex ===", flush=True)
    from utils.inference_utils import InferenceDataset

    test_dataset = InferenceDataset(
        out_dir="/tmp/ood_test",
        complex_names=["test_complex"],
        protein_files=[protein1_path],
        ligand_descriptions=[ligand_path],
        protein_sequences=[None],
        lm_embeddings=False,  # skip ESMFold download
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
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    print("  Dataset created, loading first batch...", flush=True)
    native_graph = next(iter(test_loader)).to(device)

    if not native_graph.success[0]:
        print("  ERROR: Failed to preprocess complex!", flush=True)
        sys.exit(1)

    n_rec = native_graph["receptor"].x.shape[0]
    n_lig = native_graph["ligand"].x.shape[0]
    print(f"  Receptor atoms: {n_rec}", flush=True)
    print(f"  Ligand atoms:   {n_lig}", flush=True)

    # Add complex_t (normally set during diffusion sampling). Use
    # t=0.5 for all noise types — any value works for the comparison.
    native_graph.complex_t = {"tr": 0.5, "rot": 0.5, "tor": 0.5}

    # Pad receptor features with 1280 zeros for missing ESM2 embeddings.
    # Model was trained with lm_embedding_type='precomputed' (1280-dim
    # ESM2-650M), but we use lm_embeddings=False to avoid network calls.
    # Both native and FrozenDiffDock see the same padded input, so the
    # equivalence verification remains valid.
    ESM2_DIM = 1280
    rec_x_padded = torch.cat([
        native_graph["receptor"].x,
        torch.zeros(n_rec, ESM2_DIM, device=device)
    ], dim=1)
    native_graph["receptor"].x = rec_x_padded
    print(f"  Receptor x padded: {n_rec} × {ESM2_DIM} zeros appended "
          f"(shape={rec_x_padded.shape})", flush=True)

    # ================================================================
    # Step 3: Run native CGModel forward
    # ================================================================
    print("=== Step 3: Running native CGModel.forward() ===", flush=True)
    with torch.no_grad():
        native_out = model(native_graph)
    tr_native, rot_native, tor_native, _ = native_out
    print(f"  tr_score:  shape={tr_native.shape}, "
          f"norm={tr_native.norm().item():.4f}", flush=True)
    print(f"  rot_score: shape={rot_native.shape}, "
          f"norm={rot_native.norm().item():.4f}", flush=True)
    print(f"  tor_score: shape={tor_native.shape}", flush=True)
    if tor_native.numel() > 0:
        print(f"             norm={tor_native.norm().item():.4f}", flush=True)

    # ================================================================
    # Step 4: Run FrozenDiffDock wrapper on SAME graph
    # ================================================================
    print("=== Step 4: Running FrozenDiffDock.forward() on same graph ===",
          flush=True)
    from tricoopdiff.expert import FrozenDiffDock

    frozen = FrozenDiffDock(model,
                            num_conv_layers=score_model_args.num_conv_layers)
    with torch.no_grad():
        frozen_out = frozen(native_graph)
    tr_frozen, rot_frozen, tor_frozen, _ = frozen_out
    print(f"  tr_score norm:  {tr_frozen.norm().item():.4f}", flush=True)
    print(f"  rot_score norm: {rot_frozen.norm().item():.4f}", flush=True)
    if tor_frozen.numel() > 0:
        print(f"  tor_score norm: {tor_frozen.norm().item():.4f}", flush=True)

    # ================================================================
    # Step 5: Compare outputs
    # ================================================================
    print("=== Step 5: Comparing outputs ===", flush=True)
    tr_match = torch.allclose(tr_native, tr_frozen, atol=1e-5)
    rot_match = torch.allclose(rot_native, rot_frozen, atol=1e-5)
    tor_match = True
    if tor_native.numel() > 0 and tor_frozen.numel() > 0:
        tor_match = torch.allclose(tor_native, tor_frozen, atol=1e-5)

    print(f"  tr_score  allclose: {tr_match}  "
          f"(max_diff={((tr_native - tr_frozen).abs().max().item()):.2e})",
          flush=True)
    print(f"  rot_score allclose: {rot_match}  "
          f"(max_diff={((rot_native - rot_frozen).abs().max().item()):.2e})",
          flush=True)
    if tor_native.numel() > 0:
        print(f"  tor_score allclose: {tor_match}  "
              f"(max_diff={((tor_native - tor_frozen).abs().max().item()):.2e})",
              flush=True)

    # ================================================================
    # Step 6: Test build_subgraph_e3l() compatibility
    # ================================================================
    print("=== Step 6: Testing build_subgraph_e3l() compatibility ===",
          flush=True)
    from tricoopdiff.graphs import build_subgraph_e3l

    # Extract features from native graph
    rec_pos = native_graph["receptor"].pos.clone()
    rec_x = native_graph["receptor"].x.clone()
    lig_pos = native_graph["ligand"].pos.clone()
    lig_x = native_graph["ligand"].x.clone()

    edge_types = native_graph.edge_types

    subgraph = build_subgraph_e3l(
        e3_coords=rec_pos,
        e3_features=rec_x,
        ligand_coords=lig_pos,
        ligand_features=lig_x,
        rec_edge_index=(
            native_graph["receptor", "receptor"].edge_index.clone()
            if ("receptor", "receptor") in edge_types else None),
        lig_edge_index=(
            native_graph["ligand", "ligand"].edge_index.clone()
            if ("ligand", "ligand") in edge_types else None),
        lig_edge_attr=(
            native_graph["ligand", "ligand"].edge_attr.clone()
            if ("ligand", "ligand") in edge_types else None),
        lig_edge_mask=(
            native_graph["ligand"].edge_mask.clone()
            if hasattr(native_graph["ligand"], "edge_mask") else None),
        lig_mask_rotate=None,  # will be set from native graph
    )

    # Copy metadata that build_subgraph_e3l doesn't set
    subgraph.complex_t = native_graph.complex_t
    subgraph.num_graphs = native_graph.num_graphs
    subgraph["receptor"].batch = native_graph["receptor"].batch
    subgraph["ligand"].batch = native_graph["ligand"].batch
    if hasattr(native_graph["receptor"], "sequence"):
        subgraph["receptor"].sequence = native_graph["receptor"].sequence

    # Copy precomputed embedding fields (set by embedding() call)
    for field in ["rec_node_attr", "node_sigma_emb"]:
        if hasattr(native_graph["receptor"], field):
            setattr(subgraph["receptor"], field,
                    getattr(native_graph["receptor"], field))
    if hasattr(native_graph["ligand"], "node_t"):
        subgraph["ligand"].node_t = native_graph["ligand"].node_t
    for edge_key in [("receptor", "receptor"), ("ligand", "ligand")]:
        if edge_key in edge_types:
            for field in ["rec_edge_attr", "edge_sh", "edge_weight",
                          "rec_edge_attr", "rec_edge_index"]:
                if hasattr(native_graph[edge_key], field):
                    subgraph[edge_key][field] = (
                        getattr(native_graph[edge_key], field))

    subgraph = subgraph.to(device)

    with torch.no_grad():
        subgraph_out = frozen(subgraph)
    tr_sub, rot_sub, tor_sub, _ = subgraph_out

    tr_sub_match = torch.allclose(tr_native, tr_sub, atol=1e-5)
    rot_sub_match = torch.allclose(rot_native, rot_sub, atol=1e-5)
    tor_sub_match = True
    if tor_native.numel() > 0 and tor_sub.numel() > 0:
        tor_sub_match = torch.allclose(tor_native, tor_sub, atol=1e-5)

    print(f"  tr_score  allclose: {tr_sub_match}  "
          f"(max_diff={((tr_native - tr_sub).abs().max().item()):.2e})",
          flush=True)
    print(f"  rot_score allclose: {rot_sub_match}  "
          f"(max_diff={((rot_native - rot_sub).abs().max().item()):.2e})",
          flush=True)
    if tor_native.numel() > 0:
        print(f"  tor_score allclose: {tor_sub_match}  "
              f"(max_diff={((tor_native - tor_sub).abs().max().item()):.2e})",
              flush=True)

    # ================================================================
    # Summary
    # ================================================================
    all_pass = all([tr_match, rot_match, tor_match,
                    tr_sub_match, rot_sub_match, tor_sub_match])

    print("=" * 60, flush=True)
    if all_pass:
        print("PASS: Zero OOD verified", flush=True)
        print("  - FrozenDiffDock == native CGModel (same graph)", flush=True)
        print("  - build_subgraph_e3l() produces compatible graphs", flush=True)
    else:
        print("FAIL: Mismatch detected!", flush=True)
        for name, ok in [("tr (same)", tr_match), ("rot (same)", rot_match),
                         ("tor (same)", tor_match),
                         ("tr (sub)", tr_sub_match), ("rot (sub)", rot_sub_match),
                         ("tor (sub)", tor_sub_match)]:
            if not ok:
                print(f"  - {name} MISMATCH", flush=True)

    # Save result
    os.makedirs("results", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"Zero-OOD Verification: {'PASS' if all_pass else 'FAIL'}\n")
        from datetime import datetime
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"ns={score_model_args.ns}, nv={score_model_args.nv}, "
                f"num_conv_layers={score_model_args.num_conv_layers}, "
                f"sh_lmax={score_model_args.sh_lmax}\n")
        f.write(f"Test: {protein1_path} + {ligand_path}\n")
        f.write(f"\nSame graph:\n")
        f.write(f"  tr:  {tr_match}\n")
        f.write(f"  rot: {rot_match}\n")
        f.write(f"  tor: {tor_match}\n")
        f.write(f"\nSubgraph (build_subgraph_e3l):\n")
        f.write(f"  tr:  {tr_sub_match}\n")
        f.write(f"  rot: {rot_sub_match}\n")
        f.write(f"  tor: {tor_sub_match}\n")

    print(f"Results: {args.output}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
