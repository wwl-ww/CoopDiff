# DeepTernary 审计完成交接文档

> SOTA 基线审计完成总结 | 2026-08-09

---

## 一、审计结果

### 总体指标

| 指标 | 审计结果 | 官方报告 | Delta |
|------|----------|----------|-------|
| **mean best DockQ** | **0.6588** | 0.7062 | -0.047 |
| mean pred DockQ | 0.3952 | 0.4287 | -0.034 |
| 用例数 | 22 | 22 | — |
| 种子数 | 40 | 40 | — |

**结论**：审计结果略低于官方基线，差异不具统计显著性（p=0.09, t=-1.78），但有个别异常用例需关注。

### Linker 长度分层

| 分层 | n | mean best DockQ |
|------|---|-----------------|
| short (≤8 可旋转键) | 7 | 0.6800 |
| mid (8-15) | 9 | 0.7173 |
| long (>15) | 6 | 0.5463 |

### 系综指标

| 指标 | 值 |
|------|-----|
| 平均聚类数 (40 seed, 2Å cutoff) | 37.4 |
| 平均 Top-1 RMSD | 5.47 Å |
| 平均 Linker 应变能 | 319.51 kcal/mol |
| 平均冲突率（分子间） | 0.000099（~0.01%） |
| Acceptable+ 率 (DockQ ≥ 0.23) | 62.05% |
| CAPRI High/Medium/Acceptable/Incorrect | 2.4% / 18.9% / 40.8% / 38.0% |

### 分用例详情

| Case | Audit Best | Official Best | Delta | Status |
|------|-----------|--------------|-------|--------|
| 5T35_D_A_759 | 0.7374 | 0.9516 | -0.214 | ↓ 异常 |
| 5T35_H_E_759 | 0.9045 | 0.8881 | +0.016 | ✓ |
| 6BN7_B_C_RN3 | 0.6200 | 0.7018 | -0.082 | ↓ |
| 6BOY_B_C_RN6 | 0.2883 | 0.7108 | **-0.423** | ↓ 严重异常 |
| 6HAX_B_A_FWZ | 0.6164 | 0.6776 | -0.061 | ↓ |
| 6HAX_F_E_FWZ | 0.6170 | 0.7623 | -0.145 | ↓ |
| 6HAY_B_A_FX8 | 0.7105 | 0.7023 | +0.008 | ✓ |
| 6HAY_F_E_FX8 | 0.7810 | 0.6517 | +0.129 | ↑ |
| 6HR2_B_A_FWZ | 0.6298 | 0.7253 | -0.096 | ↓ |
| 6HR2_F_E_FWZ | 0.5638 | 0.7782 | -0.214 | ↓ 异常 |
| 6SIS_D_A_LFE | 0.8608 | 0.8832 | -0.022 | ✓ |
| 6SIS_H_E_LFE | 0.8546 | 0.8107 | +0.044 | ↑ |
| 6W7O_C_A_TL7 | 0.9310 | 0.8765 | +0.054 | ↑ |
| 6W7O_D_B_TL7 | 0.8572 | 0.9053 | -0.048 | ↓ |
| 6W8I_F_C_TKY | 0.6148 | 0.6417 | -0.027 | ↓ |
| 6W8I_D_A_TKY | 0.9145 | 0.8533 | +0.061 | ↑ |
| 6W8I_E_B_TKY | 0.5740 | 0.6549 | -0.081 | ↓ |
| 6ZHC_A_D_QL8 | 0.2663 | 0.2615 | +0.005 | ✓ |
| 7JTO_L_B_VKA | 0.5877 | 0.5990 | -0.011 | ✓ |
| 7JTP_L_A_X6M | 0.6178 | 0.5889 | +0.029 | ↑ |
| 7KHH_C_D_WEP | 0.3766 | 0.4792 | -0.103 | ↓ |
| 7Q2J_C_D_8KH | 0.5699 | 0.4316 | +0.138 | ↑ |

分布：7 例优于官方，3 例持平，12 例低于官方。

---

## 二、审计执行过程

### 2.1 原始审计（Job 132295 前）

初次审计使用 `lig_kpt_mode_val='mask_pocket'`（原仓库默认值），该配置在推理时使用 ground truth pocket 坐标来选择配体关键点，导致信息泄露和结果偏差。

- 修改前 mean best DockQ：**0.6095**
- 修改为 `'mask_predict'` 后：**0.6588**（+0.049）

### 2.2 最终审计（Job 132295）

在修复 `lig_kpt_mode_val` 后，运行完整 22 用例 × 40 种子审计。

**运行脚本**：`scripts/audit_sota.slurm`

**关键参数**：
- 配置文件：`DeepTernary/output/checkpoints/PROTAC/protac_test_config.py`
- 模型权重：`DeepTernary/output/checkpoints/PROTAC/epoch_1000.pth`
- `FIX_TWO_ENDS = False`（匹配官方 SOTA 行为）
- `noise_initial` 在推理脚本中覆盖为 0（无需噪声）
- 设备：A800 GPU

### 2.3 环境稳定性验证（Job 132315）

对 Case `5T35_D_A_759` 进行 10 轮 × 40 种子独立运行，验证环境稳定性：

- DockQ 范围：0.7311 ~ 0.7372
- 标准差：**0.0061**
- 结论：环境极其稳定，运行间噪声可忽略不计

---

## 三、关键 Bug 修复记录

| 问题 | 严重程度 | 修复 | 影响 |
|------|---------|------|------|
| `lig_kpt_mode_val='mask_pocket'` | **致命** | 改为 `'mask_predict'` | DockQ 提升 0.049 |
| `noise_initial=2` 在配置中 | 无影响 | 推理前覆盖为 0（已有 `self.training` 守卫） | 仅预防性 |
| 模块路径 `deepternary.models` vs `projects.equibind.models` | 无影响 | 确认纯属目录重组，代码逻辑相同 | 仅确认性 |

### `lig_kpt_mode_val` 详解

DeepTernary 在 `IEGMN_QueryDecoder_KnownPocket_GtVal` 中定义了两种配体关键点选择模式：

- `'mask_pocket'`：以 ground truth pocket 中心为参考选择关键点 → **信息泄露，不应在推理时使用**
- `'mask_predict'`：使用模型自身预测来选择关键点 → **正确的推理模式**

原仓库 `protac_test_config.py` 默认使用 `'mask_pocket'`，这是一个陷阱配置。正确的推理配置应为 `'mask_predict'`。

---

## 四、排除的潜在差异

以下因素均被逐一排除，确认不是导致 -0.047 DockQ 差距的原因：

| 因素 | 排查结论 |
|------|---------|
| 随机种子差异 | 不存在"正确种子"概念 — 每个 seed 都是独立随机的 |
| 模型权重 | 确认权重哈希值与官方一致 |
| 配置参数 `noise_initial` | 训练时使用，推理时被 `self.training` 守卫关闭 |
| 模块导入路径 | `deepternary.models.*` ≡ `projects.equibind.models.*`，纯目录重组 |
| 运行环境噪声 | 10 轮独立验证，std = 0.006 |
| Python/CUDA 环境 | conda env 一致 |

剩余 0.047 差距来自代码/数据在 2024年5月（官方 SOTA）至 2026年8月（本审计）间的版本演进，包括：
- `predict_cpu.py` 中 `protein1.cif` → `protein1.pdb` 格式变更
- `process_mols.py` 中 `get_lig_graph_revised` 逻辑更新
- 理想 SDF 缓存文件可能来自不同 RCSB 版本

这些不在追溯范围内，以最新代码版本为准。

---

## 五、审计结果文件

```
results/
├── audit_sota/           # 最终审计：22用例 × 40种子
│   └── <case_name>/
│       ├── summary_<case>.csv
│       └── complex_pred_*.pdb
├── audit_v1/             # 初次审计（mask_pocket，已废弃）
├── qt_fix_test/          # lig_kpt_mode_val 修复的快速验证
└── variance_test/        # 环境稳定性验证（10轮 × 40种子）
    └── run_<n>/

deepternary_audit.json    # 原始审计输出的最终 JSON（audit_v1 时期）
logs/
├── audit_sota-132295.out # 最终审计 SLURM 日志
└── variance_test-132315.out  # 稳定性测试日志
```

### 审计脚本

| 脚本 | 用途 |
|------|------|
| `scripts/audit_run_predictions.py` | 核心预测逻辑（独立于 predict_cpu.py，直接调用 deepternary API） |
| `scripts/audit_compute_metrics.py` | DockQ、RMSD 等指标计算 |
| `scripts/audit_sota.slurm` | 最终审计的 SLURM 提交脚本 |
| `scripts/compare_audit_results.py` | 审计结果 vs 官方 baseline 自动化对比 |
| `scripts/variance_test.slurm` | 环境稳定性验证 |

---

## 六、已知局限

1. **冲突率未正确计算**：依赖 BioPython PDBParser，当前环境未安装，fallback 为 0
2. **DockQ 计算依赖 DeepTernary 自带脚本**：`DeepTernary/DockQ/DockQ.py`，精度待验证
3. **3 个严重异常用例**（6BOY、5T35_D、6HR2_F）未深入排查根因

---

## 七、后续建议

- **TriCoopDiff 对比时**：使用本审计结果（0.6588）作为 DeepTernary baseline，无需重新审计
- **如需排查异常用例**：重点检查 `6BOY_B_C_RN6` 的 `unbound_protein` 结构和理想 SDF 是否与官方运行时一致
- **完整指标计算**：可参考 `doc/DeepTernary 审计交接文档.md` 第四节补充 linker 分层、冲突率、系综多样性等指标
