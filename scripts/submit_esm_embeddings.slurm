#!/bin/bash
#SBATCH --job-name=esm2-embed
#SBATCH --partition=hp_a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/esm2_%j.out
#SBATCH --error=logs/esm2_%j.err

# ===== ESM2 Batch Embedding Generation =====
# Generates per-residue ESM2 embeddings for all protein structures in the
# input directory. Uses deduplication to avoid recomputing identical sequences.
#
# Usage:
#   sbatch scripts/submit_esm_embeddings.slurm
#
# Config:
#   INPUT_DIR  — directory of PDB/mmCIF files
#   OUTPUT_DIR — where to save embeddings
#   MODEL      — HuggingFace model ID
#   BATCH_SIZE — sequences per inference batch
# =====

set -euo pipefail

# ---- Configuration ----
INPUT_DIR="data/pdb_mmcif"
OUTPUT_DIR="data/esm_embeddings"
MODEL="facebook/esm2_t33_650M_UR50D"
BATCH_SIZE=4

# ---- Setup ----
# Use official Hugging Face endpoint by default; set HF_ENDPOINT explicitly if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

CONDA_BASE="/data/run01/scwb792/miniconda3"
CONDA_ENV="tricoopdiff"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== ESM2 Embedding Generation ==="
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Input:      ${INPUT_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Model:      ${MODEL}"
echo "Batch size: ${BATCH_SIZE}"
echo "================================"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# ---- Run ----
python scripts/generate_esm_embeddings.py \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda \
    --resume

echo "=== Done at $(date) ==="
