# DeepTernary 数据与评估协议

## 目标

TriCoopDiff 的主目标是在 DeepTernary 官方协议下复现并超过其结果。数据处理、输入设定、采样数、DockQ 实现和候选排序应与 DeepTernary 对齐；训练数据同时隔离 PROTAC 与 Molecular Glue（MGD）测试集，使一个统一模型可以评估两个任务。

## 官方数据读取方式

DeepTernary 并非每个训练簇永久选择一个代表结构。`TernaryPreprocessedDataset` 将一个 cluster 视为一个训练项，每个 epoch 从该 cluster 采样一个结构：

- `unrepresent_aug_rate=0.8`：80% 概率从全部 `items` 随机采样；
- 其余情况从 `representative` 采样；
- representative 为空时回退到全部 members；
- 每个结构可从预生成的 ligand conformers 中随机选择，并随机交换蛋白角色、扰动 ligand 与 protein2。

官方任务训练池为：

| 任务 | cluster 文件 | clusters | members |
|---|---|---:|---:|
| PROTAC | `DeepTernary/data/PROTAC/train_clusters_poc6.json` | 713 | 7,965 |
| MGD | `DeepTernary/data/MolecularGlue/train_clusters_poc3_exUNL.json` | 1,272 | 12,776 |

## 统一训练池

统一候选池取上述两个 member 集合的精确 case 交集：7,200 cases。该交集不包含 PROTAC-22 或 MGD-94 的同名 case。再删除与任一测试 case 共享 PDB 的：

- `5EXA_B_C_5SO`
- `6Q0R_B_C_O6M`

最终候选为 **7,198 cases / 1,528 PDB**。合并两套 cluster 关系后得到 653 个连通 cluster。train/validation 必须按 cluster 划分，不得把同一 cluster 的成员拆到两侧。训练时保留 DeepTernary 的 80% member / 20% representative 动态采样。

PROTAC-22、MGD-94 及其 ground truth 不得参与 checkpoint 选择。内部 validation 从上述653个 cluster 中固定划分，并记录 seed、case 清单及索引 SHA256。

## PROTAC 正式评估

- Case：`DeepTernary/data/PROTAC/protac22.txt` 中官方22例；`7PI4_A_D_7QB` 属于官方 validation，不计入 test。
- 输入：`unbound_protein1.pdb`、`unbound_protein2.pdb`、完整 PROTAC，以及 unbound anchor/warhead与对应 atom masks。
- 采样：每例40 seeds。
- 主结果：按模型预测的 `p2_rmsd_pred` 或等价的无标签置信度排序所得 top-1。
- 辅助结果：oracle-best DockQ，必须明确标为采样上限，不能替代 top-1。
- 指标：DeepTernary `DockQ/dockq_util.py::cal_dockq` 的 DockQ、Fnat、iRMS、LRMS，并报告 DockQ > 0.23 的 acceptable rate。

## MGD 正式评估

- Case：`DeepTernary/data/MolecularGlue/test_all.txt` 中94例。
- 输入：官方 bound setting。
- 采样：每例1 seed。
- 指标：使用同一完整原子 DockQ实现，并报告 DockQ、Fnat、iRMS、LRMS和成功率。

## 报告边界

当前 job 154339 使用791个固定 MGD representatives，未复现上述动态 cluster 采样，且接触过 PROTAC benchmark相关结构，只能作为工程训练。job 155257 使用23例 bound输入、32 walkers和oracle选择，只能作为诊断。二者结果不得用于正式“超过 DeepTernary”的结论。

正式报告必须记录 Git commit、checkpoint与索引 SHA256、每例 seed、输入模式、有效/无效 walker、排序指标及失败样本。DeepTernary 的 top-1与oracle-best必须分别汇总。
