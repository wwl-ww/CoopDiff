# TriCoopDiff

TriCoopDiff predicts protein–ligand–protein ternary complexes with two frozen
DiffDock experts, independent LoRA adapters, and a cooperative TriFuse module.
This branch implements two independent ligand hypotheses, annealed consensus,
and explicit POI motion. Read [the current model contract](doc/DualConsensus.md)
before training or evaluation. Run all commands from the repository root. Do not import PyTorch or
load models on a shared cluster login node; use a GPU compute environment.

## 1. Check out the experimental branch

```bash
git clone --branch model-v2-dual-consensus https://github.com/wwl-ww/CoopDiff.git
cd CoopDiff
git rev-parse HEAD
```

Record the commit together with every result. This experimental branch starts
from master `52d5e5b`; benchmark improvement has not yet been established.

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
hf download facebook/esm2_t33_650M_UR50D --cache-dir "$HF_HOME/hub"
hf cache scan --dir "$HF_HOME/hub" | grep esm2_t33_650M_UR50D
```

The model is public, so authentication is normally unnecessary. If the cluster
cannot reach Hugging Face, download the cache on another machine and copy the
entire cache directory without changing its internal layout.

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

## 6. Train a new dual-consensus checkpoint

Old Phase 1 checkpoints cannot initialize this new architecture. Run the
small-data gates in [DualConsensus.md](doc/DualConsensus.md), then train with
`scripts/train_entry.py --dual-stage annealed` and an independent output directory.
After training, install the best validation checkpoint and record its digest:

```bash
mkdir -p checkpoints/dual_consensus
cp results/dual_consensus_phase1/best_model.pt checkpoints/dual_consensus/best_model.pt
sha256sum checkpoints/dual_consensus/best_model.pt
```

Use the exact advertised digest; a filename alone is not provenance. A valid
checkpoint has `checkpoint_version=18`, `architecture=dual_consensus_v1`, 336 LoRA
tensors, `generative_process=dual_consensus_flow_v1`, and the full recorded runtime contract.
Do not use the old `checkpoints/phase1/`
smoke checkpoints for formal evaluation.

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
  --trifuse-ckpt checkpoints/dual_consensus/best_model.pt \
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
