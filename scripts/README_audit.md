# DeepTernary Baseline Audit

> 给服务器同学的运行说明 | 2026-08-05

## 概述

对 DeepTernary 在 PROTAC benchmark（22个测试结构，每结构40个 seed）上进行完整评估，输出 `deepternary_audit.json` 作为 TriCoopDiff 的 baseline 对比数据。

## 前置条件

1. **Conda 环境**：需要 `tricoopdiff` 环境（或修改 `audit_slurm.sh` 中的 `CONDA_ENV`）
2. **DeepTernary 已安装**：`DeepTernary/` 目录已经 clone 在项目根目录
3. **预训练权重**：已下载并解压到 `DeepTernary/output/checkpoints/PROTAC/`
4. **PROTAC 数据**：所有22个测试结构存在于 `DeepTernary/output/protac22/`

### 快速检查

```bash
cd /data/run01/scwb792/mwj/CoopDiff

# 检查权重
ls DeepTernary/output/checkpoints/PROTAC/checkpoint.pth

# 检查数据（应显示22个目录）
ls DeepTernary/output/protac22/ | wc -l

# 检查测试集列表（22个case name）
cat DeepTernary/data/PROTAC/protac22.txt
```

## 文件说明

```
scripts/
├── audit_run_predictions.py     # Phase 1: 运行40-seed预测，保存所有输出PDB
├── audit_compute_metrics.py     # Phase 2: 计算所有审计指标，输出JSON
├── audit_slurm.sh               # SLURM一键提交脚本
└── README_audit.md              # 本文件
```

## 运行方式

### 方式一：SLURM 一键提交（推荐）

```bash
cd /data/run01/scwb792/mwj/CoopDiff
sbatch scripts/audit_slurm.sh
```

这会运行完整的两个 Phase：
- Phase 1: 22 cases × 40 seeds 预测（~3-6小时，GPU）
- Phase 2: 计算所有指标（~5-10分钟，CPU）

### 方式二：手动分步运行

**Phase 1 — 运行预测（需要 GPU）：**

```bash
cd /data/run01/scwb792/mwj/CoopDiff
conda activate tricoopdiff

# 所有 22 个 cases（耗时 ~3-6 小时）
python scripts/audit_run_predictions.py \
    --work-dir DeepTernary/output/checkpoints/PROTAC \
    --out-dir audit_results \
    --seeds 40 \
    --device cuda

# 或单个 case 测试（~5分钟）
python scripts/audit_run_predictions.py \
    --case 5T35_D_A_759 \
    --out-dir audit_results \
    --seeds 10 \
    --device cuda
```

**Phase 2 — 计算指标（CPU 即可）：**

```bash
python scripts/audit_compute_metrics.py \
    --pred-dir audit_results \
    --output deepternary_audit.json
```

## 输出文件

### 预测输出（`audit_results/`）

```
audit_results/
├── prediction_progress.json                    # 进度文件（可断点续跑）
├── 5T35_D_A_759/
│   ├── complex_pred_5T35_D_A_759_0.pdb         # seed 0 预测的三元复合物
│   ├── complex_pred_5T35_D_A_759_1.pdb
│   ├── ...                                     # seed 2-39
│   ├── lig_pred_5T35_D_A_759_0.pdb             # seed 0 预测的配体
│   ├── p2_pred_5T35_D_A_759_0.pdb              # seed 0 预测的蛋白2
│   └── summary_5T35_D_A_759.csv                # 40-seed 指标汇总
├── 5T35_H_E_759/
│   └── ...
└── ...
```

每个 `summary_{case}.csv` 包含字段：
`seed, fnat, irms, Lrms, dockq, pred_p2_rmsd, gt_p2_rmsd, lig_rmsd, sm_rmsd, overall_rmsd`

### 最终审计 JSON（项目根目录）

`deepternary_audit.json` — 指标口径与权威基线见 `doc/DeepTernary 审计完成交接文档.md`。

## 断点续跑

如果预测中断，可以从上次保存的进度恢复：

```bash
python scripts/audit_run_predictions.py \
    --resume-from audit_results/prediction_progress.json
```

## 配置修改

| 需求 | 修改 |
|------|------|
| 不同 conda 环境 | 编辑 `audit_slurm.sh` 的 `CONDA_ENV` |
| GPU 内存不足 | 加 `--device cpu`（慢很多）或用 `--case` 逐个跑 |
| 减少 seeds 加快测试 | `--seeds 10` |
| 跳过 clash rate（加速） | `audit_compute_metrics.py --skip-clash` |

## 注意事项

1. **不要手动运行 SLURM** — 脚本已写好，`sbatch` 提交即可
2. **Phase 1 需要 GPU**，Phase 2 只需要 CPU
3. **40 seeds × 22 cases = 880 次前向传播**，预计 GPU 上 3-6 小时
4. 如果某 case 在 RDKit 嵌入步骤卡住，脚本会自动 fallback 到 ideal.sdf
5. DockQ 计算依赖 `DeepTernary/DockQ/fnat` 二进制文件（x86-64 Linux）
6. 如果 `fnat` 二进制不存在或不可执行，运行 `cd DeepTernary/DockQ && make`
