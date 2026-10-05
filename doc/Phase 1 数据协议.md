# 数据与指标协议

更新：2026-10-05。区分历史训练池设计、具体 checkpoint 的训练来源和当前开发评估；三者不能互相替代。

## 训练池与内部验证

历史 Phase 1 取 DeepTernary 的 `PROTAC/train_clusters_poc6.json` 与 `MolecularGlue/train_clusters_poc3_exUNL.json` 成员交集，排除 PROTAC22/MGD94 同四位 PDB ID，得到 7,198 cases、653 merged clusters。原配置 `split_seed=0`，588 train / 65 validation clusters；每 epoch 按 cluster 采样，角色交换概率 50%。

这些是原索引设计，不是所有已训练权重的实际数据。历史 train/val 曾发现 **83 个同 PDB 重叠**：cluster-disjoint 不自动保证 PDB-disjoint。生成或接收索引时检查成员、显式 split 标签、PDB 交集、benchmark 排除、输入可解析性及内容哈希；不要仅凭 7,198/653 计数放行。

内部验证用于 checkpoint、时间步、筛选阈值和来源比例选择。固定路径验证不等于未见结构验证；少量 cluster smoke 不代表完整验证集。MGD94 已用于开发消融，PROTAC22 也已用于协议比较，不能再称完全未触碰测试集，更不能用于 selector 监督训练。

## model-4.5 数据修复

当前共享旧索引实测存在83个同PDB交集。新索引在保留两个官方来源聚类关系的基础上，
传递合并同PDB成员后再划分；不移动或改写共享结构文件。
复现：`python scripts/make_deepternary_index.py --reuse-index data/ternary_train.json data/ternary_val.json --index-dir results/model45_partition`。
得到训练6705例、验证382例，总数7087不变，111个已有不支持配体排除不变。
原始653个合并簇进一步合为491个PDB连接组；材料化后训练441组、验证49组，
另1组全部成员属于既有配体排除。审计PDB重叠为0，原始两个来源1985个聚类无跨划分成员。
完整指纹见`results/model45_partition/partition_audit.json`；生产训练启动和resume执行隔离与指纹检查。
这不代表重新完成全体蛋白序列同源聚类。旧三元权重见过部分新验证成员，不能用作干净验证对照。

## 联合训练来源

B/C 训练以 `SDPI:BiTSeed = 1:1` 交替取样，验证集固定包含 65 个 SDPI 与 65 个 BiTSeed 路径。BiTSeed endpoint 在首次访问 case 时在线生成，固定 `C=1, M=1`：一个 ETKDGv3 conformer，两个冻结专家各生成一次。任一专家在 OOD 输入上失败时整例回退 SDPI，实际来源记为 SDPI，不丢弃样本。

训练分支按内部 validation loss 选择 source-conditioned C epoch 81 EMA（`3.1384998601`），不等于 DockQ 最优。PROTAC22 完整配对评估后，本分支以 C40 EMA 为精修工作基线，C81作为验证loss最佳的对照。详细协议和分项 loss 见 [SDPI–BiTSeed 联合训练交接](SDPI-BiTSeed联合训练交接.md)，下游结果见[调研第2节](../RANKING_PROTAC_RESEARCH.md)。

## 当前输入口径

- MGD94 全池审计：94 例、bound 蛋白；索引 `results/reproduction_20260911/index_current/mgd94_full_v1.json`。当前配体化学为 93 例 CCD SDF、3IQV 一例 PDB fallback。
- PROTAC22 比例消融：unbound 蛋白与完整 ligand；bound 模板仅用于事后评分。
- 早期从晶体拆分蛋白的 PROTAC smoke 仍是 bound 输入；重新生成 ligand 不能使蛋白变成 unbound。
- 所有输入、模板、链映射、原子顺序、化学策略、torsion policy 和缓存版本须进入 manifest。

## 指标与报告

使用 [deepternary_metrics.py](../tricoopdiff/deepternary_metrics.py) 的完整原子 DockQ 口径；近似 DockQ 单列。每个 case 固定候选池与 selector 集合 S：

- Initial：`I=max(DockQ_initial[j], j∈S)`。
- Final：`F=max(DockQ_final[j], j∈S)`。
- Union：`U=max(I,F)`，只比较初末态。
- Retention：逐例 selected-oracle / full-pool-oracle 的比值再平均。

这些都是读取真实结构后的 oracle；native-free selector 选出候选不等于部署 top-1。比较必须共享 case、seed、初态、权重和数值协议。模型或协议改变后必须重新评估。
