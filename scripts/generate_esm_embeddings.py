#!/usr/bin/env python3
"""
Precompute ESM2-650M per-residue embeddings for PROTAC benchmark proteins.

Uses tricoopdiff.esm_embedder (HuggingFace transformers via HF mirror).
Output format: {protein_path: [per-chain tensor, ...]} stored as .pt file.

Usage:
    python scripts/generate_esm_embeddings.py
    python scripts/generate_esm_embeddings.py --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import torch

_COOPDIFF = str(Path(__file__).parent.parent)
if _COOPDIFF not in sys.path:
    sys.path.insert(0, _COOPDIFF)

from tricoopdiff.diffdock.utils.inference_utils import get_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM2 embeddings for PROTAC benchmark proteins")
    parser.add_argument("--protac-dir", type=str,
                        default="DeepTernary/output/protac22")
    parser.add_argument("--output", type=str,
                        default="results/esm_embeddings.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    protac_dir = Path(args.protac_dir)
    cases = sorted(d for d in os.listdir(protac_dir)
                   if os.path.isdir(protac_dir / d))
    print(f"Found {len(cases)} cases")

    # Collect unique protein files
    protein_paths = set()
    for case in cases:
        for pname in ["protein1.pdb", "protein2.pdb"]:
            p = str(protac_dir / case / pname)
            if os.path.exists(p):
                protein_paths.add(p)
    protein_paths = sorted(protein_paths)
    print(f"Unique protein files: {len(protein_paths)}")

    from tricoopdiff.esm_embedder import ESMEmbedder

    print(f"Loading ESM2-650M on {args.device}...")
    embedder = ESMEmbedder(device=args.device)

    embeddings_map = {}
    for p in protein_paths:
        seqs = get_sequences([p], [None])
        chain_seqs = seqs[0].split(":")
        print(f"  {Path(p).parent.name}/{Path(p).name}: "
              f"{len(chain_seqs)} chain(s), "
              f"lengths={[len(s) for s in chain_seqs]}", flush=True)

        chain_embeddings = embedder.embed_from_fasta(
            chain_seqs, batch_size=len(chain_seqs))
        embeddings_map[p] = [emb.cpu() for emb in chain_embeddings]

        for i, (seq, emb) in enumerate(zip(chain_seqs, chain_embeddings)):
            assert len(seq) == emb.shape[0], \
                f"Length mismatch: seq {len(seq)} vs emb {emb.shape[0]}"
            assert emb.shape[1] == 1280, \
                f"Wrong dim: {emb.shape[1]}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_map, out_path)
    total_res = sum(sum(e.shape[0] for e in v) for v in embeddings_map.values())
    print(f"Saved {len(embeddings_map)} proteins ({total_res} residues) "
          f"to {out_path}")


if __name__ == "__main__":
    main()
