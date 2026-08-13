# DeepTernary 论文分析与审计对比

> 基于 DeepTernary 论文（Nature Communications, 2025）+ 官方结果文件 + 本地审计数据 | 2026-08-08
>
> 📄 **论文 PDF**: [DeepTernary_paper.pdf](DeepTernary_paper.pdf)（arXiv:2502.18875）
> 📝 **论文 Markdown**: [DeepTernary_paper.md](DeepTernary_paper.md)（28页全文提取）
> 📊 **审计数据**: `deepternary_audit.json`（22 cases × 40 seeds）
>
> ⚠️ **基线修正（2026-08-13）**：本文档 §三/§五 引用的审计 DockQ **0.555** 基于公开 `checkpoint.pth` + `FIX_TWO_ENDS=True` + `lig_kpt_mode_val='mask_pocket'`（信息泄露配置），已被 [[DeepTernary 审计完成交接文档]]（08-09）的最终审计取代——**0.6588**（SOTA 权重 `epoch_1000.pth` + `mask_predict` + `FIX_TWO_ENDS=False`）。§五的基线结论以 **0.6588** 为准。

---

## 一、论文核心信息

- **标题**: SE(3)-Equivariant Ternary Complex Prediction Towards Target Protein Degradation
- **作者**: Fanglei Xue, Meihan Zhang, Shuqi Li, Xinyu Gao, James A. Wohlschlegel, Wenbing Huang, Yi Yang, Weixian Deng
- **期刊**: Nature Communications, Volume 16, Article 5514 (2025)
- **DOI**: 10.1038/s41467-025-61272-5
- **代码**: github.com/youqingxiaozhua/DeepTernary

### 1.1 方法概述

DeepTernary 是一个端到端的 SE(3)-等变图神经网络，带 query-based decoder，用于预测 PROTAC 和分子胶 (MGD) 诱导的三元复合物结构。

- **训练数据**: TernaryDB — 从 PDB 中筛选的 ~20,000 个三元复合物结构（排除 PROTAC 类似物，实现 zero-shot 泛化）
- **架构**: IEGMN (Intra/Inter-Equivariant Graph Matching Network) + QueryDecoder，含三元 inter-graph attention
- **推理速度**: PROTAC ~7 秒，MGD ~1 秒
- **输入**: E3 连接酶 + POI 的 unbound 结构 + PROTAC 配体
- **策略**: 刚性对齐 (rigid alignment)，非扩散生成

### 1.2 论文核心声明

| 声明 | 数值 |
|------|------|
| PROTAC benchmark DockQ | **0.65** |
| 对比 RosettaDock-based 方法 DockQ | 0.44 |
| 对比 BOTCP DockQ | 低于 DeepTernary |
| 对比 AlphaFold 3 / Chai-1 | 低于 DeepTernary |
| MGD benchmark DockQ | 0.23 |
| 对比 EquiDock（MGD） | 0.04 |
| BSA 最优范围 | 1100–1500 Å² |
| 推理时间 | ~7 秒/PROTAC |

---

## 二、官方结果 vs 论文声明

### 2.1 关键发现：论文结果文件使用的模型权重未公开发布

在 `DeepTernary/output/checkpoints/PROTAC/result_20240515193524.txt` 中发现了官方完整结果表。该文件引用的模型配置为：

```
work_dir: output/SOTA/PROTAC
checkpoint: output/SOTA/PROTAC/epoch_1000.pth
experiment_name = 'PP_No4_poc_alpoc_poc_poc_poc_poc'
```

而公开发布的权重路径为 `output/checkpoints/PROTAC/checkpoint.pth`，`output/SOTA/` 目录不存在于发布版本中。**论文报告的 0.65/0.71 DockQ 来自未发布的 SOTA 版本模型**。

### 2.2 官方完整结果（22 cases, 40 seeds each）

官方结果文件包含以下 summary 行：

```
| pred dockq | best dockq | best rank | best SmRMSD | sm rank | RMSD<10 | hit_count | hit_rate |
|   0.4287   |   0.7062   |   2.8182  |    1.6182   | 12.4091 |  0.6386 |           |  0.4068  |
Acceptable Rate: 1.0
```

| 指标 | 官方 SOTA 模型（未发布） | 论文声称 |
|------|------------------------|---------|
| Mean best DockQ | **0.706** | 0.65 |
| Mean average DockQ | 0.429 | 未报告 |
| Acceptable Rate (best) | 100% (22/22) | 未明确 |
| 平均 SmRMSD | 1.62 Å | 未报告 |
| RMSD<10Å 比例 | 63.9% | 未报告 |

> 论文中引用的 0.65 可能是其他计算口径（如中位数、top-1 选择、或不同 seed 设置），但与官方结果文件的 0.706 存在差异，具体原因不明。

---

## 三、本地审计结果 vs 官方结果

### 3.1 总体对比

| 指标 | 官方 SOTA (result.txt) | 本地审计 (checkpoint.pth) | 差异 |
|------|----------------------|--------------------------|------|
| Mean best DockQ | **0.706** | **0.555** | **-0.151** |
| Mean average DockQ | 0.429 | 0.263 | -0.166 |
| Median best DockQ | ~0.72 (估算) | 0.555 | ~-0.17 |
| Acceptable Rate (best) | 100% | 100% | 一致 |
| CAPRI High (best) | 多个 >0.8 | 1 个 | 显著差距 |

### 3.2 逐 case 对比

| Case | 官方 Best | 审计 Best | Δ | 官方 Avg | 审计 Avg | Δ |
|------|----------|----------|---|---------|---------|---|
| 5T35_D_A_759 | 0.9516 | 0.7035 | **-0.248** | 0.8617 | 0.3508 | -0.511 |
| 5T35_H_E_759 | 0.8881 | 0.7918 | -0.096 | 0.5834 | 0.3992 | -0.184 |
| 6BN7_B_C_RN3 | 0.7018 | 0.4449 | **-0.257** | 0.4301 | 0.1706 | -0.260 |
| 6BOY_B_C_RN6 | 0.7108 | 0.2662 | **-0.445** | 0.0429 | 0.0812 | **+0.038** |
| 6HAX_B_A_FWZ | 0.6776 | 0.4939 | -0.184 | 0.4413 | 0.3202 | -0.121 |
| 6HAX_F_E_FWZ | 0.7623 | 0.4697 | **-0.293** | 0.2020 | 0.3068 | **+0.105** |
| 6HAY_B_A_FX8 | 0.7023 | 0.5014 | -0.201 | 0.5375 | 0.3300 | -0.208 |
| 6HAY_F_E_FX8 | 0.6517 | 0.5172 | -0.135 | 0.3538 | 0.3093 | -0.045 |
| 6HR2_B_A_FWZ | 0.7253 | 0.4637 | **-0.262** | 0.6502 | 0.3691 | -0.281 |
| 6HR2_F_E_FWZ | 0.7782 | 0.4989 | **-0.279** | 0.6219 | 0.3577 | -0.264 |
| 6SIS_D_A_LFE | 0.8832 | 0.6693 | -0.214 | 0.5619 | 0.2837 | -0.278 |
| 6SIS_H_E_LFE | 0.8107 | 0.7037 | -0.107 | 0.5413 | 0.3283 | -0.213 |
| 6W7O_C_A_TL7 | 0.8765 | 0.7553 | -0.121 | 0.4241 | 0.2418 | -0.182 |
| 6W7O_D_B_TL7 | 0.9053 | 0.6912 | -0.214 | 0.4819 | 0.2349 | -0.247 |
| 6W8I_F_C_TKY | 0.6417 | 0.5460 | -0.096 | 0.3188 | 0.1851 | -0.134 |
| 6W8I_D_A_TKY | 0.8533 | 0.8618 | **+0.009** ✓ | 0.7486 | 0.4842 | -0.264 |
| 6W8I_E_B_TKY | 0.6549 | 0.6417 | -0.013 | 0.4972 | 0.3706 | -0.127 |
| 6ZHC_A_D_QL8 | 0.2615 | 0.4134 | **+0.152** ✓ | 0.2484 | 0.1422 | -0.106 |
| 7JTO_L_B_VKA | 0.5990 | 0.6088 | **+0.010** ✓ | 0.1642 | 0.1289 | -0.035 |
| 7JTP_L_A_X6M | 0.5889 | 0.3555 | **-0.233** | 0.1818 | 0.0926 | -0.089 |
| 7KHH_C_D_WEP | 0.4792 | 0.3532 | -0.126 | 0.3650 | 0.0863 | **-0.279** |
| 7Q2J_C_D_8KH | 0.4316 | 0.4611 | **+0.030** ✓ | 0.1728 | 0.2109 | **+0.038** |

- **审计优于官方的 case**: 4/22 (18%) — 6W8I_D_A_TKY, 6ZHC_A_D_QL8, 7JTO_L_B_VKA, 7Q2J_C_D_8KH
- **审计显著劣于官方的 case** (Δ < -0.2): 10/22 (45%) — 差距最大的集中在 5T35、6BN7、6BOY、6HR2、6HAX_F等

### 3.3 按 linker 长度分层

#### 官方 SOTA（估算，按 case 分组对齐）

| Linker 分组 | N | 官方 Best DockQ | 审计 Best DockQ | Δ |
|------------|---|----------------|----------------|-----|
| Short (≤8) | 7 | ~0.720 | 0.522 | -0.198 |
| Mid (8-15) | 9 | ~0.726 | 0.598 | -0.128 |
| Long (>15) | 6 | ~0.652 | 0.529 | -0.123 |

#### 审计数据（实际计算）

| Linker 分组 | N | Best DockQ | Avg DockQ | CAPRI High/Med/Acc/Inc |
|------------|---|-----------|----------|----------------------|
| Short (≤8) | 7 | 0.522 | 0.262 | 0/4/3/0 |
| Mid (8-15) | 9 | 0.598 | 0.299 | 1/6/2/0 |
| Long (>15) | 6 | 0.529 | 0.246 | 0/3/3/0 |

> 注意：审计中的 linker 分组基于可旋转键计数（与 paper 的定义可能不同）。

### 3.4 CAPRI 分布（审计数据，40 seeds × 22 cases = 880 predictions）

| CAPRI 等级 | 数量 | 比例 |
|-----------|------|------|
| Incorrect (DockQ < 0.23) | 394 | 44.8% |
| Acceptable (0.23 ≤ DockQ < 0.49) | 390 | 44.3% |
| Medium (0.49 ≤ DockQ < 0.80) | 93 | 10.6% |
| High (DockQ ≥ 0.80) | 3 | 0.3% |

官方未报告 CAPRI 分布（仅报告了 Acceptable Rate = 100% 指至少 1 个 seed 达标）。

---

## 四、审计发现的额外指标（论文未报告）

这些是论文中未涉及的指标，属于 TriCoopDiff 审计补充：

| 指标 | 数值 | 说明 |
|------|------|------|
| 平均配体 RMSD | 5.21 Å | 预测 PROTAC 坐标 vs 真实坐标 |
| 平均冲突率 | 1.35×10⁻⁵ | 分子间 Cβ-Cβ 距离 < 3.0Å 比例（极低） |
| 平均 Linker 应变能 | 356.9 | MMFF94 能量（部分 case 为 0，因 RDKit 力场分配失败） |
| 平均 40-seed 聚类数 | 38.0 | 几乎每种 seed 产生独特构象 |
| 平均 Top-1 聚类 RMSD | 5.25 Å | 最大聚类 centroid 到真值的距离 |

### 4.1 关键发现

1. **DeepTernary 的刚性对齐几乎不产生空间冲突**（冲突率接近零），这是一个重要的正向发现
2. **40 seed 几乎没有聚类收敛**（平均 38/40 个独特构象），说明 seed 多样性极高但缺乏向正确构象的收敛
3. **pred_p2_rmsd 作为 top-1 选择标准的可靠性有限**：best DockQ (0.555) 远高于 mean DockQ (0.263)，说明需要有更好的构象选择策略
4. **种子稳定性**：5→10→20→40 seed 的 best DockQ 持续上升（0.422→0.472→0.523→0.555），40 seed 尚未饱和

---

## 五、对 TriCoopDiff 竞争定位的影响

### 5.1 基线修正

| 维度 | 原定位（基于论文声称 0.65） | 修正后定位（基于最终审计 0.6588） |
|------|--------------------------|---------------------------|
| TriCoopDiff 正面追赶难度 | 需要 ≥0.65 | 需要 ≥0.66（仅需略超审计值） |
| 与 DeepTernary 的结构预测差距 | 声称差距 0.1 | 实际差距取决于所用 checkpoint（最终审计 0.6588 已逼近官方 0.706） |
| 功能排序壁垒 | DeepTernary 无此能力 | 壁垒不变 |

### 5.2 关键情报

1. **论文报告值与公开权重的差距**：官方 SOTA 模型 (DockQ 0.706) 的权重经 Zenodo 获取后，最终审计 (08-09) 复现为 **0.6588**，与官方仍差 0.047（归因于代码/数据 2024→2026 版本演进，非权重差异）
2. **这降低了 TriCoopDiff 结构预测方面的追赶难度**：目标从 "≥0.65" 修正为 "≥0.66"
3. **基线已确定**：以 [[DeepTernary 审计完成交接文档]]（08-09）的 **0.6588** 作为唯一对比基线，无需重新审计
4. **困难 case 一致**：6BOY、6ZHC、7KHH、7JTP、7Q2J 在官方 SOTA 和审计版本下都是 DockQ 最低的几个

### 5.3 建议调整

- 实验计划 Phase 1 的 DockQ 复现目标从 "确认 0.65 基线" 修正为 "确认最终审计 0.6588 基线"（已完成，见审计完成交接文档）
- TriCoopDiff FT 版本的 DockQ 目标从 "≥ 0.65" 修正为 "≥ 0.66"，这已追平本地审计基线
- 长 linker 子集的预期优势保持不变（DeepTernary 的刚性对齐在长 linker 上系统性地表现更差）

---

## 六、论文中值得关注的其他发现

### 6.1 BSA-活性相关

论文报告埋藏表面积 (BSA) 与实验降解活性相关，最优 BSA 范围 1100–1500 Å²。这是一个重要的功能预测维度——TriCoopDiff 可以在这方面进行对比：从扩散系综计算 BSA 分布 vs DeepTernary 单结构 BSA。

### 6.2 Pocket Points Decoder

论文的 ablation 表明 QueryDecoder 优于 IEGMN（尤其在 DockQ > 0.49 的中高质量区间）。TriCoopDiff 不使用 pocket points 范式（使用完整的 protein surface graph），这个设计差异需要在相关工作中讨论。

### 6.3 MGD Benchmark

MGD (分子胶) benchmark 上 DeepTernary DockQ 仅 0.23（vs EquiDock 0.04），远低于 PROTAC 上的表现。MGD 的界面更小、更依赖诱导契合，这是扩散方法的天然优势场景。

---

## 七、SOTA 权重调查 ⭐ 重要发现

> 调查日期: 2026-08-08
> 结论: **SOTA 权重 (`epoch_1000.pth`) 确实存在且可获取，但未包含在 GitHub Release 中，而是存档在 Zenodo。**

### 7.1 三份模型权重对比

| 来源 | 文件 | 大小 | MD5 | DockQ |
|------|------|------|-----|-------|
| **Zenodo public.zip** | `public/DeepTernary/PROTAC/epoch_1000.pth` | **199.78 MB** | 待下载后验证 | ~0.706（官方结果） |
| GitHub Release v1.0.0 | `output/checkpoints/PROTAC/checkpoint.pth` | 64.2 MB | `b9dd85bc58ed6e86fb408612aba4baac` | 0.555（审计） |
| Zenodo public.zip | `public/DeepTernary/MGD/epoch_1000.pth` | 251.44 MB | 待验证 | ~0.046 (MGD) |

### 7.2 调查过程

#### (A) GitHub Release 分析

| 版本 | 大小 | 内容 |
|------|------|------|
| v1.0.0 (Feb 2025) | 193 MB | 代码 + checkpoint.pth (65MB) + 结果文件等 |
| v1.0.1 (Feb 2025) | 144 MB (声称) | 减去 48MB（推测移除了部分辅助文件） |

GitHub Release 中的 `output/` 仅包含 `checkpoint.pth` (65MB)，**不含** `output/SOTA/` 目录。官方结果文件 `result_20240515193524.txt` 引用的路径 `output/SOTA/PROTAC/epoch_1000.pth` 在发布版本中不存在。

#### (B) Zenodo 存档 (DOI: 10.5281/zenodo.12727661)

Zenodo 上有一个 **500.7 MB** 的 `public.zip`（MD5: `5be73d036ad483e5f411dcd79e52ae24`），包含 471 个文件。通过 byte-range 请求只下载 ZIP 末尾的 Central Directory（3MB，节省 500MB 流量），解析得到完整文件列表：

**关键文件**：
```
public/DeepTernary/PROTAC/epoch_1000.pth          199.78 MB  ← SOTA PROTAC 权重！
public/DeepTernary/MGD/epoch_1000.pth             251.44 MB  ← SOTA MGD 权重！
public/DeepTernary/PROTAC/20240301_081911/        训练日志 + TensorBoard
public/DeepTernary/PROTAC/protac.py               测试配置
public/DeepTernary/MGD/glue.py                     MGD 测试配置
```

通过下载 PROTAC `epoch_1000.pth` 的 ZIP local header 验证：该文件为有效的 PyTorch checkpoint（ZIP 格式，含 `archive/data.pkl`）。

#### (C) 为什么 Github Release 和 Zenodo 版本不同？

SOTA 模型是用内部代码库 `projects.equibind`（见训练日志中的 `type='projects.equibind.models.triplet_dock.TripletDock'`）训练的，包含非公开的 Baidu 内部基础设施。公开发布时将代码重构为 `deepternary` 命名空间并做清理，但 SOTA 权重本身通过 Zenodo 存档包含在 `public.zip` 中。

### 7.3 预测流程差异

对比官方 `predict_cpu.py` 和审计脚本，发现一个关键配置差异：

| 配置 | 官方 predict_cpu.py | 审计 audit_run_predictions.py |
|------|-------------------|-----------------------------|
| `FIX_TWO_ENDS` | **False** | **True** |
| `CORRECT_LIGAND` | True | True |
| `noise_initial` | 0 | 0 |
| `translation_distance` | 5 | 5 |

`FIX_TWO_ENDS=True` 在 RDKit 生成初始构象时固定 anchor + warhead 位置（仅生成 linker 构象），理论上给模型更多先验信息。这可能导致审计结果**偏高**（而非偏低）。注：08-09 最终审计改用 SOTA 权重 `epoch_1000.pth` + `FIX_TWO_ENDS=False` 后复现 **0.6588**——即 0.555→0.6588 的差距来自权重差异，而 0.6588→官方 0.706 的剩余 0.047 差距归因于代码/数据 2024→2026 版本演进（见 [[DeepTernary 审计完成交接文档]] §四）。

### 7.4 下一步行动（以下行动已于 08-09 完成，见 [[DeepTernary 审计完成交接文档]]）

1. **下载 SOTA 权重**：从 Zenodo 下载 `public.zip` 中提取 `epoch_1000.pth`（199.78 MB）。Zenodo 支持 byte-range 下载，可以精确 range 请求 `270355142-479835258`，仅下载 ~200MB 而非完整 500MB。
2. **重新运行审计**：使用 SOTA 权重替换 `checkpoint.pth`，设置 `FIX_TWO_ENDS=False`（对齐官方配置），预期复现 DockQ ~0.706。
3. **更新基线**：将 TriCoopDiff 的追赶目标校准为 SOTA 级别的 0.706。

### 7.5 下载命令

```bash
# 方法 1：仅下载 PROTAC epoch_1000.pth（~200MB）
curl -r 270355142-479835258 \
  -o epoch_1000.pth \
  "https://zenodo.org/api/records/12727661/files/public.zip/content"

# 方法 2：下载完整 public.zip（~500MB，较慢）
curl -L -o public.zip \
  "https://zenodo.org/api/records/12727661/files/public.zip/content"

# 方法 3：仅下载 MGD epoch_1000.pth（~251MB）
curl -r 3345914-266996508 \
  -o epoch_1000_mgd.pth \
  "https://zenodo.org/api/records/12727661/files/public.zip/content"
```

> ⚠️ Zenodo 下载速度实测约 15-25 KB/s，下载 200MB 预计约 2-3 小时。建议用 `--continue-at -` 支持断点续传或在网络较好的时段下载。

---

## 八、数据来源

- **论文信息**: Nature Communications 16, 5514 (2025), DOI: 10.1038/s41467-025-61272-5
- **官方结果**: `DeepTernary/output/checkpoints/PROTAC/result_20240515193524.txt`
- **审计结果**: `deepternary_audit.json` (2026-08-08, 22 cases × 40 seeds)
- **公开权重 (GitHub)**: `DeepTernary/output/checkpoints/PROTAC/checkpoint.pth` (65MB, DockQ 0.555)
- **SOTA 权重 (Zenodo)**: `public/DeepTernary/PROTAC/epoch_1000.pth` (200MB, DockQ 0.706, DOI: 10.5281/zenodo.12727661)
- **PROTAC benchmark 列表**: `DeepTernary/data/PROTAC/protac22.txt`
