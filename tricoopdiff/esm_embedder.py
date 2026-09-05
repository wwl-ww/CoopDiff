"""
ESM2 protein residue embedding extractor.

Uses Meta's ESM2-650M (esm2_t33_650M_UR50D) via HuggingFace transformers
to generate per-residue embeddings for protein sequences. Aligned with
DiffDock-L's protein feature pipeline.

Embeddings are 1280-dimensional per residue. These can be used directly
as e3_features/poi_features inputs to TriCoopDiff's subgraph builders,
or cached to disk for reuse.

Usage:
    embedder = ESMEmbedder(device='cuda')
    embeddings = embedder.embed_from_fasta(['MALWMR...', 'KPVT...'])
    for seq, emb in zip(sequences, embeddings):
        print(emb.shape)  # (seq_len, 1280)
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Union

# Use HF mirror for model download (huggingface.co blocked from cluster)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModel


class ESMEmbedder(nn.Module):
    """
    Wrapper around ESM2 for per-residue embedding extraction.

    Loads facebook/esm2_t33_650M_UR50D by default (DiffDock-L standard).
    Outputs per-residue embeddings at 1280 dimensions.

    Model variants and their output dims:
      esm2_t30_150M_UR50D  → 640
      esm2_t33_650M_UR50D  → 1280  (default)
      esm2_t36_3B_UR50D    → 2560
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cuda",
        repr_layer: int = 33,  # last layer
    ):
        """
        Args:
            model_name: HuggingFace model ID for ESM2
            device: 'cuda', 'cpu', or specific device
            repr_layer: which transformer layer to extract (default: last)
        """
        super().__init__()
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        self.model.eval()

        self.embed_dim = self.model.config.hidden_size
        self.repr_layer = repr_layer

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def embed_from_fasta(
        self,
        sequences: List[str],
        batch_size: int = 8,
    ) -> List[torch.Tensor]:
        """
        Extract per-residue embeddings for a list of protein sequences.

        Sequences use standard 20 amino acid single-letter codes.
        Non-standard residues will be tokenized as unknown/ignored.

        Args:
            sequences: list of amino acid strings, e.g. ['MALWMR...', 'KPVT...']
            batch_size: max sequences per batch (adjust for GPU memory)

        Returns:
            list of tensors, each shape (seq_len_i, embed_dim), on CPU.
            Order matches input sequences.
        """
        all_embeddings: List[torch.Tensor] = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch_tokens = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_tokens["input_ids"],
                    attention_mask=batch_tokens["attention_mask"],
                    output_hidden_states=True,
                )

            # Extract the specified layer's hidden states
            hidden = outputs.hidden_states[self.repr_layer]  # (B, T, D)

            # For each sequence, extract per-residue embeddings.
            # ESM token sequence: <cls> R1 R2 ... RL <eos> <pad> ...
            # Per-residue embeddings are tokens 1:L (strip <cls> and trailing specials)
            for j in range(len(batch_seqs)):
                seq_len = len(batch_seqs[j])
                # Take positions 1 to 1+seq_len (skip <cls>, take real residues)
                res_emb = hidden[j, 1 : 1 + seq_len, :].cpu()
                all_embeddings.append(res_emb)

        return all_embeddings

    def embed_from_pdb(
        self,
        pdb_path: Union[str, Path],
        chain_id: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract per-residue embeddings from a PDB file.

        Uses BioPython to parse the PDB structure and extract amino acid
        sequences. Returns embeddings keyed by chain ID.

        Args:
            pdb_path: path to PDB or mmCIF file
            chain_id: optional specific chain to embed (if None, all chains)

        Returns:
            dict mapping chain_id → per-residue embedding tensor (L_chain, 1280)
        """
        from Bio.PDB import PDBParser, MMCIFParser

        pdb_path = Path(pdb_path)

        # Select parser based on file type
        if pdb_path.suffix.lower() in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        structure = parser.get_structure(pdb_path.stem, str(pdb_path))

        # Extract sequences per chain
        chain_sequences: Dict[str, str] = {}
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")

        for model in structure:
            for chain in model:
                cid = chain.get_id()
                if chain_id is not None and cid != chain_id:
                    continue

                residues = []
                for residue in chain:
                    # Skip hetero residues and waters
                    if residue.get_id()[0] != " ":
                        continue
                    # Map 3-letter code to 1-letter
                    resname = residue.get_resname().strip()
                    one_letter = _aa3to1.get(resname, "X")
                    # Only keep standard amino acids
                    if one_letter in standard_aa:
                        residues.append(one_letter)
                    else:
                        residues.append("X")

                if residues:
                    chain_sequences[cid] = "".join(residues)

        if not chain_sequences:
            raise ValueError(f"No valid protein chains found in {pdb_path}")

        # Get embeddings for all chains
        chain_ids = list(chain_sequences.keys())
        sequences = [chain_sequences[cid] for cid in chain_ids]
        embeddings = self.embed_from_fasta(sequences)

        return dict(zip(chain_ids, embeddings))

    def embed_from_sequence_file(
        self,
        fasta_path: Union[str, Path],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from a FASTA file.

        Args:
            fasta_path: path to FASTA file

        Returns:
            dict mapping header → per-residue embedding tensor
        """
        from Bio import SeqIO

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}")

        headers = [record.id for record in records]
        sequences = [str(record.seq) for record in records]
        embeddings = self.embed_from_fasta(sequences)

        return dict(zip(headers, embeddings))


# ---------------------------------------------------------------------------
# Three-letter to one-letter amino acid mapping
# ---------------------------------------------------------------------------

_aa3to1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    # Non-standard but common
    "MSE": "M",  # selenomethionine → methionine
    "HIE": "H", "HID": "H", "HIP": "H",  # histidine protonation states
    "CYX": "C", "CYM": "C",  # cysteine variants
    "ASH": "D", "GLH": "E",  # aspartic/glutamic acid protonation
    "LYN": "K",  # neutral lysine
    "ARN": "R",  # neutral arginine
}
