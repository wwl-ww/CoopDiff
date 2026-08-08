#!/bin/bash
#SBATCH --job-name=dd-weights
#SBATCH --partition=gpu_a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --output=logs/dd_weights_%j.out
#SBATCH --error=logs/dd_weights_%j.err

set -euo pipefail

DEST="/data/home/scwb792/run/mwj/CoopDiff/checkpoints"
mkdir -p "${DEST}" logs

echo "=== Download DiffDock weights ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

# Try multiple mirrors in order of preference
URLS=(
    "https://ghproxy.net/https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip"
    "https://gh.con.sh/https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip"
    "https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip"
)

for URL in "${URLS[@]}"; do
    echo "Trying: ${URL}"
    # Use wget with retry for better reliability
    if wget --timeout=30 --tries=5 --retry-connrefused -O "${DEST}/diffdock_models.zip" "${URL}" 2>&1; then
        SIZE=$(stat -c%s "${DEST}/diffdock_models.zip" 2>/dev/null || echo 0)
        if [ "${SIZE}" -gt 100000000 ]; then
            echo "Success! Downloaded ${SIZE} bytes."
            echo "Unzipping..."
            cd "${DEST}"
            unzip -o diffdock_models.zip
            echo "Done at $(date)"
            exit 0
        fi
    fi
    echo "Failed or file too small, trying next URL..."
    rm -f "${DEST}/diffdock_models.zip"
done

echo "All URLs failed."
exit 1
