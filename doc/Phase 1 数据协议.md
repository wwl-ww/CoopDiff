# 数据与指标协议

更新：2026-09-14。区分历史训练池设计、具体 checkpoint 的训练来源和当前开发评估；三者不能互相替代。

## 训练池与内部验证

历史 Phase 1 取 DeepTernary 的 `PROTAC/train_clusters_poc6.json` 与 `MolecularGlue/train_clusters_poc3_exUNL.json` 成员交集，排除 PROTAC22/MGD94 同四位 PDB ID，得到 7,198 cases、653 merged clusters。原配置 split_seed=0，588 train / 65 validation clusters；每 epoch 按 cluster 采样，角色交换概率 50%。

这些是原索引设计，不是所有已训练权重的实际数据。历史 train/val 曾发现 **83 个同 PDB 重叠**：cluster-disjoint 不自动保证 PDB-disjoint。生成或接收索引时检查成员、显式 split 标签、PDB 交集、benchmark 排除、输入可解析性及内容哈希；不要仅凭 7,198/653 计数放行。来源见 [make_deepternary_index.py](../scripts/make_deepternary_index.py)、[check_phase1_readiness.py](../scripts/check_phase1_readiness.py)。

内部验证用于 checkpoint、时间步、筛选阈值和来源比例选择。固定路径验证不等于未见结构验证；少量 cluster smoke 不代表完整验证集。MGD94 已用于开发消融，PROTAC22 也已用于协议比较，不能再称完全未触碰测试集，更不能用于 selector 监督训练。最终泛化声明需要另设未参与决策的 holdout。

## 当前输入口径

- MGD94 全池审计：94 例、bound 蛋白；索引 `results/reproduction_20260911/index_current/mgd94_full_v1.json`。当前配体化学为 93 例 CCD SDF、3IQV 一例 PDB fallback，须在报告中保留该例。
- PROTAC22 比例消融：unbound 蛋白与完整 ligand；bound 模板仅用于事后评分。
- 早期从晶体拆分蛋白的 PROTAC smoke 仍是 bound 输入；重新生成 ligand 不能使蛋白变成 unbound。
- 所有输入、模板、链映射、原子顺序、化学策略、torsion policy 和缓存版本须进入 manifest；不可静默丢弃失败例。

当前 BiTSeed 通过冻结专家在线生成两侧完整 ligand pose，不使用测试三元 native 来选择配对或 fallback。它不是预存 pose bank，也不是已知 warhead/口袋先验的替代名称。

## DeepTernary 对照的公平性

本地历史源码对照显示不同入口的条件不一致；对照时固定 checkout/入口/参数，不把其历史行为当作所有版本的规则：

| 路径 | 需要记录的差异 |
| --- | --- |
| MGD 历史 `predict_cpu.py` | bound 内部结构，但随机化 ligand/P2 刚体 |
| MGD `predict.py` bound 路径 | 保留输入 P2 布局；ligand 图构造可能对齐输入 bound pose |
| PROTAC unbound | 两端原子 mask、二元端点构象与蛋白口袋条件；记录 `FIX_TWO_ENDS/use_fix_two_ends` |

训练时组件的 5 Å 随机平移不是“native 附近 5 Å 精修”。如果复现对方官方配置，MGD 的 1 seed 和 PROTAC 的 40 seeds 也要注明；当前 64-walker 对照不自动成为相同预算的官方 benchmark。

## 主指标和三张榜单

使用 [deepternary_metrics.py](../tricoopdiff/deepternary_metrics.py) 的完整原子 DockQ 口径；近似 DockQ 单列，不能混算。每个 case 固定候选池与 selector 集合 S：

- Initial：`I=max(DockQ_initial[j], j∈S)`。
- Final：`F=max(DockQ_final[j], j∈S)`。
- Union：`U=max(I,F)`，只比较初末态，不含中间步。
- 跨 case 分别取均值/中位数和 ≥0.23、≥0.49 命中数，报告失败分母。
- Retention：逐例 selected-oracle / full-pool-oracle 的比值再平均，不是两个均值相除；命中候选 ID 与命中同分最大值分开统计。

这些都是读取真实结构后的 oracle；native-free selector 选出 64 个候选，不等于已经从中选出部署 top-1。Union 尤其不能解释为模型会自动保存更好初态。

初态 `acceptable` 是粗 Cα 几何代理；采样 `valid` 是数值状态。均不能替代末态全原子 clash、化学几何和界面审计。独立二元的 DockQ-small_molecules 是另一套量，见[二元评估](独立DiffDock小分子评估.md)。

## 报告与复现要求

比较必须共享 case、seed、初态、权重和数值协议；独立旧 baseline 只作历史背景。全池审计只精修每个候选一次，再离线比较 selector，避免每组另采一池。记录候选数、实际来源、失败阶段、选中 ID、初末态指标及耗时；有 fallback 的 94 例主表与纯成功子集分开。

模型或协议改变后重新评估，不能用旧 checkpoint 的 selector 排名推断新模型最优配置。配对效应和不确定性优先于 `0.000x` 均值名次。
