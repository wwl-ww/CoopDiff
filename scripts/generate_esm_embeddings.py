#!/usr/bin/env python3
"""
Batch ESM2 embedding generation for protein structures.

Walk a directory of PDB/mmCIF files, extract protein sequences, deduplicate,
and generate per-residue ESM2 embeddings. Output cached as .pt tensors for
fast loading during training/inference.

Usage:
    # Single case
    python scripts/generate_esm_embeddings.py \
        --input data/pdb_mmcif/ \
        --output data/esm_embeddings/ \
        --device cuda

    # All cases with SLURM
    sbatch scripts/submit_esm_embeddings.slurm

    # Resume interrupted run
    python scripts/generate_esm_embeddings.py \
        --input data/pdb_mmcif/ \
        --output data/esm_embeddings/ \
        --resume

Output structure:
    data/esm_embeddings/
    ├── index.json           # {pdb_id: {chain_id: embedding_path}}
    ├── <pdb_id>/
    │   ├── <chain_id>.pt    # per-residue embedding tensor (L, 1280)
    │   └── sequence.txt     # amino acid sequence for reference
    └── ...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Set, Dict, Optional
from hashlib import md5

# Use HF mirror for model download (huggingface.co blocked from cluster)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tricoopdiff.esm_embedder import ESMEmbedder


def find_pdb_files(input_dir: Path) -> list:
    """Find all PDB/mmCIF files in a directory tree."""
    extensions = {".pdb", ".cif", ".mmcif", ".ent", ".pdb.gz"}
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(set(files), key=lambda p: p.name)


def extract_sequences_from_pdb(
    pdb_path: Path,
    standard_aa: Set[str] = None,
) -> Dict[str, str]:
    """Extract protein sequences from a PDB file, keyed by chain ID."""
    from Bio.PDB import PDBParser, MMCIFParser

    if standard_aa is None:
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")

    aa3to1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        "MSE": "M", "HIE": "H", "HID": "H", "HIP": "H",
        "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E",
        "LYN": "K", "ARN": "R",
    }

    suffix = pdb_path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    chain_seqs: Dict[str, str] = {}
    for model in structure:
        for chain in model:
            cid = chain.get_id()
            residues = []
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                one = aa3to1.get(residue.get_resname().strip(), "X")
                residues.append(one if one in standard_aa else "X")
            if residues:
                chain_seqs[cid] = "".join(residues)

    return chain_seqs


def seq_hash(sequence: str) -> str:
    """Short hash of a sequence for deduplication."""
    return md5(sequence.encode()).hexdigest()[:12]


def load_index(output_dir: Path) -> dict:
    """Load existing embedding index for resume support."""
    index_path = output_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return {}


def save_index(output_dir: Path, index: dict):
    """Save embedding index to disk."""
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description="Batch ESM2 embedding generation for protein structures"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory of PDB/mmCIF files"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--device", "-d", default="cuda",
        help="Device for inference (cuda or cpu, default: cuda)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Sequences per batch (default: 4, reduce for GPU OOM)"
    )
    parser.add_argument(
        "--model", default="facebook/esm2_t33_650M_UR50D",
        help="ESM2 model name on HuggingFace"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-computed embeddings"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only discover files and sequences, do not compute"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N PDB files (for testing)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Discover and deduplicate sequences
    # ------------------------------------------------------------------
    pdb_files = find_pdb_files(input_dir)
    print(f"Found {len(pdb_files)} PDB/mmCIF files in {input_dir}")

    if args.limit:
        pdb_files = pdb_files[: args.limit]
        print(f"Limited to first {args.limit} files")

    # Extract sequences and build dedup map
    # seq_hash → {first_pdb, first_chain, sequence}
    dedup: Dict[str, dict] = {}
    # pdb_file → {chain_id → seq_hash}
    file_map: Dict[str, Dict[str, str]] = {}

    for pdb_path in pdb_files:
        try:
            chain_seqs = extract_sequences_from_pdb(pdb_path)
        except Exception as e:
            print(f"  SKIP {pdb_path.name}: parse error ({e})")
            continue

        pdb_id = pdb_path.stem
        file_map[pdb_id] = {}
        for chain_id, seq in chain_seqs.items():
            h = seq_hash(seq)
            file_map[pdb_id][chain_id] = h
            if h not in dedup:
                dedup[h] = {
                    "sequence": seq,
                    "first_pdb": pdb_id,
                    "first_chain": chain_id,
                    "length": len(seq),
                }

    total_chains = sum(len(v) for v in file_map.values())
    unique_seqs = len(dedup)
    print(f"Total chains: {total_chains}, Unique sequences: {unique_seqs}")
    print(f"Effective computation: {unique_seqs} sequences "
          f"(saving {total_chains - unique_seqs} dedup hits)")

    if args.dry_run:
        # Print summary and exit
        len_dist = {}
        for info in dedup.values():
            l = info["length"]
            bucket = (l // 100) * 100
            len_dist[bucket] = len_dist.get(bucket, 0) + 1
        print("\nSequence length distribution:")
        for bucket in sorted(len_dist):
            print(f"  {bucket}-{bucket+99}: {len_dist[bucket]} sequences")
        print("\nSample sequences:")
        for i, info in enumerate(list(dedup.values())[:5]):
            seq_preview = info["sequence"][:60] + ("..." if len(info["sequence"]) > 60 else "")
            print(f"  [{info['first_pdb']}:{info['first_chain']}] "
                  f"L={info['length']} {seq_preview}")
        return

    # ------------------------------------------------------------------
    # Phase 2: Load existing index for resume
    # ------------------------------------------------------------------
    index = load_index(output_dir) if args.resume else {}
    completed_hashes: Set[str] = set()
    if args.resume:
        for pdb_entry in index.values():
            for chain_entry in pdb_entry.values():
                if isinstance(chain_entry, dict):
                    completed_hashes.add(chain_entry.get("hash", ""))
        print(f"Resume: {len(index)} PDBs already indexed, "
              f"{len(completed_hashes)} unique sequences already embedded")

    # Filter to only new sequences
    pending_hashes = [h for h in dedup if h not in completed_hashes]
    pending_sequences = [dedup[h]["sequence"] for h in pending_hashes]
    print(f"Pending: {len(pending_hashes)} sequences to embed")

    if not pending_hashes:
        print("All sequences already embedded, nothing to do.")
        return

    # ------------------------------------------------------------------
    # Phase 3: Run ESM2
    # ------------------------------------------------------------------
    print(f"\nLoading ESM2 model: {args.model}")
    t0 = time.time()
    embedder = ESMEmbedder(
        model_name=args.model,
        device=args.device,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s, "
          f"embed_dim={embedder.embed_dim}")

    # Embed in batches
    print(f"Generating embeddings (batch_size={args.batch_size})...")
    all_embeddings = embedder.embed_from_fasta(
        pending_sequences,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Phase 4: Save to disk
    # ------------------------------------------------------------------
    # Map pending hash → embedding tensor
    hash_to_embedding = dict(zip(pending_hashes, all_embeddings))

    saved_count = 0
    for pdb_id, chains in file_map.items():
        pdb_output_dir = output_dir / pdb_id
        pdb_output_dir.mkdir(parents=True, exist_ok=True)

        if pdb_id not in index:
            index[pdb_id] = {}

        for chain_id, h in chains.items():
            if h not in hash_to_embedding and h not in completed_hashes:
                continue

            emb = hash_to_embedding.get(h)
            if emb is not None:
                # Save embedding
                emb_path = pdb_output_dir / f"{chain_id}.pt"
                torch.save(emb, emb_path)

                # Save sequence reference
                seq_path = pdb_output_dir / f"{chain_id}.txt"
                with open(seq_path, "w") as f:
                    f.write(dedup[h]["sequence"])

                index[pdb_id][chain_id] = {
                    "path": str(emb_path.relative_to(output_dir)),
                    "hash": h,
                    "source_pdb": dedup[h]["first_pdb"],
                    "source_chain": dedup[h]["first_chain"],
                    "length": len(dedup[h]["sequence"]),
                }
                saved_count += 1

    # Save index
    save_index(output_dir, index)

    elapsed = time.time() - t0
    print(f"\nDone. Saved {saved_count} new embeddings in {elapsed:.1f}s "
          f"({elapsed/len(pending_hashes):.1f}s/sequence)")


if __name__ == "__main__":
    main()
