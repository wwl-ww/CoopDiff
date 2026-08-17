# doc/ 文档索引

本目录存放 TriCoopDiff 项目的全部交接文档（中文）。按**权威度**分为三档：最新（可直接引用）、部分过时（以回改标注为准）、已被取代（仅供追溯）。

> ⚠️ 快速定位：**模型架构 → `TriCoopDiff 模型交接文档.md`（v2.4）**；**DeepTernary 基线 → `DeepTernary 审计完成交接文档.md`（0.6588）**。

## 一、权威 / 最新（可直接引用）

| 文档 | 主题 | 说明 |
|------|------|------|
| `TriCoopDiff 模型交接文档.md` | 模型架构（v2.4） | **唯一权威架构文档**。逐层 FiLM、多速率扩散、SE3 刚体、口袋引导、TriReflect、生产性构象定义、ADR 决策记录。与代码一致。 |
| `TriCoopDiff 执行计划.md` | 9 周执行计划 | Phase 0-5 里程碑、Go-No-Go 决策链、验收标准。与模型文档 v2.4 一致。 |
| `三分子初始化策略方案.md` | t=1 初始化 | 球壳随机（方案 A，MVP 默认）/ 自适应范围（方案 B）等 5 方案。与代码 `TernaryInitializer` 一致。 |
| `DeepTernary 审计完成交接文档.md` | DeepTernary 基线（08-09） | **唯一权威基线结论：mean best DockQ = 0.6588**。含 `lig_kpt_mode_val` 致命 bug 修复记录。 |
| `DiffDock 审计文档.md` | DiffDock 源码+论文分析 | Corso et al. ICLR 2023 综合分析，二体专家参考。 |
| `DeepTernary 论文分析与审计对比.md` | 论文分析 + 审计对比（08-08） | 已按 08-09 最终审计回改（顶部有修正 banner）。SOTA 权重（Zenodo）调查仍有效。 |
| `问题清单与修复状态 交接文档.md` | **当前问题总清单（08-17）** | 快照：旋转 sign bug、#1/#2/#3/#4 均已修；验证已跑（smoke PASS、零-FiLM 基线 DockQ 0.215 vs 0.844）→ **新问题：采样保真度未达标**，待 agent 调查。**先读这个**。 |
| `采样保真度调查任务 brief.md` | 采样保真度调查任务书 | **派给 agent 的只读诊断任务**。零-FiLM 基线 0.215→0.844 的逐项对照清单（TriCoopDiff 反向循环 vs DiffDock 原生 inference 六个对照点），含环境约束（SLURM-only、不改代码）。agent 只调查不改代码，报告后主线程决定是否修。 |
| `Score合并 双重计数与零-FiLM基线 交接文档.md` | 问题 #1：`s_q1+s_q2` 合并 | **已落地**。方案 B（residual-gate `s_q=s_q1+g*(s_q2-s_q1)`，g 零初始化）已实现于 `merging.py`/`model.py`/`tribrain.py`，帧修正（问题 A）也已落地；候选 A/B/C 讨论见《问题清单与修复状态 交接文档.md》。 |

## 二、部分过时（以回改标注为准）

| 文档 | 主题 | 过时点 |
|------|------|--------|
| `TriCoopDiff vs DeepTernary 竞争定位与实验计划.md` | 竞争定位 + 实验计划（v2.1） | 已回改：版本交叉引用、基线 0.65→0.6588、瓶颈 FiLM→逐层 FiLM、待办状态。§五实验计划的目标值仍偏乐观（0.6x 档），实际以 0.6588 基线校准。 |
| `Experiment1 失败案例交接文档.md` | DiffDock 基线 NaN 诊断 | 已回改：DiffDock 路径 `./DiffDock`、基准规模 23/46（含 7PI4，比 DeepTernary 的 22 多一个）、git 分支已合并。NaN 根因诊断与修复结论仍有效。 |

## 三、已被取代（仅供追溯）

| 文档 | 主题 | 被谁取代 |
|------|------|---------|
| `DeepTernary 审计交接文档.md` | DeepTernary 审计任务说明（08-04） | 被 `DeepTernary 审计完成交接文档.md`（08-09）取代 |

## 四、论文原文（参考）

| 文件 | 内容 |
|------|------|
| `DeepTernary_paper.md` / `DeepTernary_paper.pdf` | DeepTernary 论文全文（arXiv:2502.18875，Nat. Commun. 2025） |
| `DiffDock_paper.pdf` | DiffDock 论文（arXiv:2210.01776，ICLR 2023） |

## 五、关键共识数字（避免再次漂移）

| 主题 | 权威值 | 来源 |
|------|--------|------|
| DeepTernary 本地审计 mean best DockQ | **0.6588**（22 cases × 40 seeds） | 审计完成交接文档（08-09） |
| DeepTernary 论文声称 DockQ | 0.65 | 论文（口径未明） |
| DeepTernary 官方 SOTA 结果文件 DockQ | 0.706 | 未发布权重，经 Zenodo 获取后审计复现 0.6588 |
| TriCoopDiff FT 版 DockQ 目标 | ≥ 0.66 | 竞争定位 §3.2（已回改） |
| PROTAC benchmark 规模 | DeepTernary 审计 **22**；DiffDock 基线 **23**（含 7PI4） | `protac22.txt` 22 个；DiffDock 基线多 1 个 7PI4 |
| 可训练参数量 | ~2.2M（TriBrain） | 模型文档 v2.4 |
| FiLM 方案 | 逐层独立 FiLM（6 层 × 2 子图 = 12 head，DiT adaLN-Zero） | 模型文档 v2.4 |
| DiffDock 位置 | **已完全 vendor 到 `tricoopdiff/diffdock/`**（模型核心 + 特征化 `datasets/`/`utils/inference_utils.py`），无需外部 checkout | 代码 + DiffDock 实验交接文档 |
