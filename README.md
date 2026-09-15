# TriCoopDiff

TriCoopDiff predicts protein–ligand–protein ternary complexes with two immutable
DiffDock experts and a cooperative residual TriFuse module. The selected model
uses source-conditioned mixed SDPI/BiTSeed training; LoRA expert adaptation was
tested and rejected because it worsened combined validation loss. Run all
commands from the repository root in the direct local GPU environment.

## 1. Clone the validated branch

```bash
git clone --branch exp/dual-init https://github.com/wwl-ww/CoopDiff.git
cd CoopDiff
git rev-parse HEAD
```

Record the exact commit together with every result.

## 2. Create the software environment

The checked-in environment is an export of the environment used for training:

```bash
conda env create -f environment.yml
conda activate tricoopdiff
```

On a new cluster, first confirm that its NVIDIA driver supports the CUDA stack
in `environment.yml`. Do not casually upgrade PyTorch, PyG, DGL, RDKit, e3nn,
or ProDy independently; these packages are sensitive to binary compatibility.

## 3. Download the frozen DiffDock expert

TriCoopDiff requires the DiffDock-L score-model configuration and weights. The
confidence model is not used by the current evaluator, but is included in the
official archive.

```bash
mkdir -p checkpoints
curl -L \
  https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip \
  -o checkpoints/diffdock_models.zip
unzip checkpoints/diffdock_models.zip -d checkpoints
test -s checkpoints/score_model/model_parameters.yml
test -s checkpoints/score_model/best_ema_inference_epoch_model.pt
```

Do not substitute another DiffDock release without revalidating the architecture
and metrics.

## 4. Cache ESM2 before evaluation

Protein features use `facebook/esm2_t33_650M_UR50D`. Runtime loading is strictly
local/offline. Choose persistent storage visible to compute nodes and preserve
the same `HF_HOME` when running evaluation.

```bash
export HF_HOME=/path/to/persistent/huggingface
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "facebook/esm2_t33_650M_UR50D",
    cache_dir="/path/to/persistent/huggingface/hub",
)
PY
```

The model is public, so authentication is usually unnecessary. If the cluster
cannot reach Hugging Face, download the cache on another machine and copy the
entire cache directory without changing its internal layout. Set `HF_ENDPOINT`
explicitly if your environment requires `https://hf-mirror.com`.

## 5. Download DeepTernary code and structural data

DeepTernary supplies the official split metadata, TernaryDB structures, PROTAC22
unbound inputs, and the exact full-atom DockQ implementation used for comparison.

```bash
git clone https://github.com/youqingxiaozhua/DeepTernary.git DeepTernary
git -C DeepTernary checkout 827821dccca31a5918bd0355e2d6bf70c072b6dd

mkdir -p data
curl -L \
  https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.1/TernaryDB.tar.gz \
  -o data/TernaryDB.tar.gz
mkdir -p DeepTernary/data
tar -xzf data/TernaryDB.tar.gz -C DeepTernary/data

curl -L \
  https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.0/output.zip \
  -o DeepTernary/output.zip
unzip DeepTernary/output.zip -d DeepTernary
```

Expected inputs include:

```text
DeepTernary/data/PROTAC/train_clusters_poc6.json
DeepTernary/data/MolecularGlue/train_clusters_poc3_exUNL.json
DeepTernary/data/TernaryDB/pdbs/
DeepTernary/output/protac22/
```

The generated Phase 1 indexes are intentionally not distributed as authoritative
inputs. Rebuild them from the pinned sources:

```bash
python scripts/make_deepternary_index.py
python scripts/check_phase1_readiness.py \
  --training-only \
  --data-index data/ternary_train.json \
  --out results/phase1_readiness.json
```

The index command must report 7,198 cases and 653 merged clusters. See
[`doc/Phase 1 数据协议.md`](doc/Phase%201%20数据协议.md) for split and leakage rules.

## 6. Install the selected dual-initialization checkpoint

The Phase 1 checkpoint is not stored in Git. Obtain `best_model.pt` directly
from the project owner and keep its supplied SHA256 alongside it:

```bash
mkdir -p checkpoints/dual_init
cp /path/from/handoff/best_model.pt checkpoints/dual_init/best_model.pt
sha256sum checkpoints/dual_init/best_model.pt
```

Use the exact advertised digest; a filename alone is not provenance. The
selected run is source-conditioned arm C at epoch 81 with validation loss
`3.1384998601`. A valid checkpoint has `checkpoint_version=20`, architecture
`residual_consensus_v2`, empty expert-adapter state, and both raw and EMA
TriFuse weights. Do not use LoRA experiment checkpoints for formal evaluation.

## 7. First validation gate

In a GPU compute environment, first run exactly one PROTAC case with one walker
and 20 steps:

```bash
mkdir -p logs results
export HF_HOME=/path/to/persistent/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/evaluate_entry.py \
  --device cuda \
  --case 5T35_D_A_759 \
  --n-walkers 1 \
  --steps 20 \
  --trifuse-ckpt checkpoints/dual_init/best_model.pt \
  --out results/gate_5T35.json
```

Accept the gate only if the job exits successfully, reports `1/1` valid walkers,
loads a non-null TriFuse checkpoint, and computes DeepTernary full-atom DockQ.
Then repeat the same case with 32 walkers. Do not tune `t_anchor`, choose epochs,
or select hyperparameters on PROTAC22/MGD94 test results.

## 8. Artifact and safety rules

- `data/`, `checkpoints/`, `results/`, `logs/`, and `DeepTernary/` are ignored by
  Git; transfer large artifacts separately.
- Preserve the repository-relative layout because scripts resolve caches and
  inputs from the project root.
- Record commit, checkpoint SHA256, expert SHA256, seed, command, and output
  paths for each experiment.
- Consult [`doc/README.md`](doc/README.md) before older research notes; some old
  23-case, 70-case, zero-FiLM, and checkpoint-v3 procedures are obsolete.
