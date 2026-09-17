# PROTAC 信息建模交接

## 分支与职责

- 分支：`protac`
- worktree：`/mnt/disk/wl/MoRE-tri/CoopDiff-protac`
- 基线：`model-v3` 的合并提交 `9002dda`
- 唯一主问题：**哪些 native-free、部署时真实可得的 PROTAC 信息，能够提高 TriCoopDiff 精修后的 DockQ？**

本分支负责条件资产、条件编码、训练—部署一致性和结构级验证。候选池排序、256→64 selector、post-ranker 与 walker 间排序通信属于 `rank` 分支，不在这里并行修改。需要共享接口时先写清张量或 manifest 合同，再协调合并。

## 已知结论，不要重复

当前候选与精修基线为 Hybrid-56/8、64 walkers、20步、seed 31000，父模型为 C40 EMA。PROTAC22 上三组严格配对结果为：

| 条件训练 / 推理输入 | Final | Union | Final ≥0.23/0.49 |
| --- | ---: | ---: | --- |
| default / pocket | 0.3066 | 0.3289 | 15/1 |
| default / absent | 0.3244 | 0.3374 | 17/2 |
| control / absent | 0.2936 | 0.3175 | 16/0 |

同一权重下 `pocket - absent` 的 Final 为 `-0.0178`（9胜/13负）。完整256池的 case-best 也没有改善高质量尾部。因此：

1. 6 Å Cα pocket-only mask **不能作为默认条件**。
2. default-absent 相对 control 的提升只能说明训练方式可能有益，不能声称模型利用了 pocket。
3. PROTAC22 中严格双侧 warhead 化学映射为 0/22；禁止用坐标最近邻或宽松子图结果伪造 atom map。
4. v2 pocket manifest 有修饰残基节点错位，已废弃；只能使用与 DiffDock/ProDy 节点顺序一致的 v3 资产。

完整证据见 `todo.md`、`Q.md` 和 `RANKING_PROTAC_RESEARCH.md`。不要从失败或 partial 目录拼接结果。

## 当前后台作业

原仓库 `/mnt/disk/wl/MoRE-tri/CoopDiff` 正在运行 P90/P100 多 seed 的训练与配对评估，使用封存 runtime。它们检验 pocket-only 训练—部署匹配，不代表新方案。不要停止、覆盖或从本 worktree 重启这些任务；等统一审计报告落盘后只补充结论。

生成数据、checkpoint 与缓存仍位于原仓库的 `data/`、`results/`、`checkpoints/`。本 worktree 不复制大文件；需要时使用绝对路径，并把新输出写入带协议和日期的独立目录。

## 下一阶段优先级

### P0：恢复可信的两端角色

先解决 warhead/靶蛋白侧的身份问题，再训练模型。优先使用有明确化学来源的片段 SMILES、键级/电荷修复和人工不可知的确定性消歧规则。任何 atom map 必须验证 canonical ligand identity、原子数、立体化学、唯一子图、连接键和两侧来源。无法唯一确认时标记 missing，不猜测。

### P1：同侧局部二元几何

在每个 protein–warhead 自己的局部坐标系中编码信息，例如：

- warhead 原子角色、linker attachment bond 与 linker atom mask；
- protein–fragment 距离/RBF、接触和方向性局部边；
- DiffDock 置信度或多姿态不确定性；
- 条件存在性、来源和可信度 mask。

不得输入 native ternary 坐标、两个蛋白的真实相对变换或由其派生的跨蛋白距离。条件应对每侧共同刚体变换保持不变或等变。

### P2：先做“信息是否被使用”的小实验

在正式长训前至少比较正确条件、全缺失条件、侧别交换、错误配对条件和几何扰动条件。要求同 checkpoint、同候选、同初态和同 diffusion seed。若正确条件与错误或缺失条件没有可重复差异，不进入大规模训练。

## 实验门槛

所有结构实验必须报告 Initial、Final、Union，以及 ≥0.23/0.49；同时给出逐例胜/平/负。主要因果量是**同一权重下正确条件减 absent**，而不是新权重减旧权重。进入默认协议至少要求：

- 全部预注册 case 完成，不能只统计成功子集；
- proposal pool hash、selected candidate ID、初态坐标/DockQ 和随机种子完全一致；
- Final 均值提高，Final ≥0.23 不减少，胜数不少于负数；
- 至少两个训练 seed 方向一致；
- 条件打乱或侧别交换后收益明显下降。

PROTAC22 是开发集，只能用于机制筛选；最终结论需要独立的 PROTAC holdout。case-best/oracle 只能判断上限，不能写成部署 top-1。

## 代码入口与检查

优先复用：

- `tricoopdiff/protac_conditions.py`：化学映射与本侧口袋审计；
- `tricoopdiff/condition_manifest.py`：严格部署 manifest；
- `tricoopdiff/condition_features.py`：条件张量；
- `tricoopdiff/conditional_consensus.py`：条件融合；
- `scripts/train_conditioned_entry.py`：配对训练；
- `scripts/evaluate_selected_refinement.py`：固定候选结构评估。

最小回归：

```bash
python tests/test_protac_conditions.py
python tests/test_condition_manifest.py
python tests/test_condition_features.py
python tests/test_conditioned_training.py
python tests/test_selected_refinement_contract.py
```

运行入口必须先导入 `tricoopdiff.diffdock_compat`。冻结两个 DiffDock 专家，不恢复 LoRA，不引入 SLURM，不让测试自动下载模型。提交前记录命令、checkpoint SHA256、manifest 指纹、数据 split 和输出路径。

## 首个交付物

先提交一份“可用信息资产审计”：逐 case 列出两侧 fragment 来源、严格映射状态、attachment/linker 定义、局部二元几何是否可用及失败原因。只有审计证明存在足够覆盖率后，才实现新 encoder；否则应明确报告数据缺口，而不是用 pocket-only 重跑替代。
