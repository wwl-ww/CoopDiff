# 低 DockQ 根因排查交接

更新时间：2026-09-11。本文件是根因与设计方向的短版交接；当前执行队列见
[`todo.md`](../todo.md)，逐实验报告见 `results/`。

## 一句话结论

当前模型不是被一个隐藏实现错误卡住，而是从宽 SDPI prior 到唯一 native endpoint
存在多模态配对和梯度冲突；MGD 同时缺正确位置与朝向，且模型缺少可靠的姿态排序信号。
合理路线是“联合 SE(3) 多假设生成 + 显式几何 ranker”，不是继续微调单一速度 loss。

## 已排除的实现问题

- CCD/single-bond 化学恢复、重原子顺序和 v7 cache 已审计。
- neighbor→query 图方向、无自环和 CUDA radius cap 语义已有回归检查。
- native 重建能得到 DockQ 1.0；完整原子 DockQ 是主指标，近似 DockQ 只作诊断。
- 训练参数获得梯度，固定单状态能够记忆，输出头不存在简单“完全学不动”。
- checkpoint 的 frozen expert 身份、EMA、无 LoRA 和 torus 运行语义均显式验证。
- sampler 从 SDPI t=1 开始；确定性实验会保存初始化 hash 并复核精确重放。

这些检查说明继续寻找原子映射、loader 或单个符号错误的预期收益很低。

## 根因证据

### 1. coupling 不可辨识

固定状态训练可降低训练误差，但 holdout coupling 变差。同 case/time 的不同 coupling
在高噪声段会给出冲突梯度，MSE 因而趋向平均方向。长训练、更多重复遍数或更大
accumulation 不能恢复不存在于条件中的 endpoint 身份。

### 2. 位置和朝向存在协同缺失

固定 MGD 五例 × 8 seeds 的 40 对因果 oracle：

| 干预 | mean DockQ | median DockQ | DockQ≥0.23 |
|---|---:|---:|---:|
| 无干预 | 0.04058 | 0.02613 | 0/40 |
| 仅 translation | 0.06021 | 0.04600 | 0/40 |
| 仅 rotation | 0.06617 | 0.03108 | 3/40 |
| 全部刚体分量 | 1.00000 | 1.00000 | 40/40 |

joint-rigid 读取 native ligand conformer 与 native target，因此只能解释“缺哪些自由度”，
不能报告为模型能力。它证明 translation-only anchor 不足；位置与朝向必须联合提出。

### 3. 覆盖存在，但排序失败

MGD94 × 32 的 oracle-best mean/median 为 `0.21922/0.17427`，33/94 达到
DockQ 0.23；但全部 walker 的 mean 只有 `0.05397`。现有 consensus latent ranker
在独立 test 只保留 44.1% oracle，在 MGD 五例只保留 15.2%。这说明生成覆盖与
top-1 选择必须分别解决。

### 4. 小幅 loss 变体没有实际收益

windowed、soft-time、direction、agreement、prior-replay 和 joint-translation
直接监督均未达到 `+0.01` 且未改善稳定命中。典型结果：direction 最佳 epoch 约
`+0.00706`；agreement 全部 240 对为 `-0.00143`；600-epoch prior-replay 的
严格 85 对只有 `+0.00076`。这些路线已关闭，不再扩训。

## 当前设计方向

### 联合生成器

`tricoopdiff/pose_anchors.py` 已提供 proper rotation + translation 的 SE(3) anchor
原语，并支持同一表面 patch 对枚举多个绕法向 twist。下一步需要 native-independent
的蛋白表面 patch proposal，再把 anchor 作为 broad SDPI 之外的完整初始化假设。

候选必须同时包含：接触位置、相对法向、切向/twist；只有距离或质心平移不够。
先在内部 PDB-disjoint 面板验证 coverage，再接触 MGD94。

### 几何 ranker

不要继续只喂 128 维 consensus context。下一版输入至少包括：

- E3–POI、E3–ligand、POI–ligand 的多阈值接触计数/距离分位数；
- 原子穿插与短程排斥；
- 两侧配体接触是否平衡；
- 界面面积和表面法向互补的代理量；
- 模型自身速度残差或跨步稳定性，但不能使用 native 派生指标。

训练和选择只使用内部 PDB-disjoint 32/8/10 split；MGD94 只作最终评估。

## 关键代码

- `tricoopdiff/trajectory_experiment.py`：已试验的 loss 变体与配对训练逻辑。
- `scripts/local_exact_gate.py`：确定性完整原子 DockQ 和因果 oracle。
- `scripts/prepare_local_mgd_panel.py`：固定 MGD 诊断/全量协议。
- `tricoopdiff/pose_anchors.py`：联合 SE(3) anchor 几何原语。
- `tricoopdiff/pose_ranking.py`、`scripts/train_pose_ranker.py`：独立 listwise ranker。
- `scripts/evaluation_runner.py`、`scripts/evaluate_index.py`：可恢复逐 case 评估与
  实验 checkpoint 的显式门禁。

## 接手顺序

1. 先读 `todo.md` 的当前四卡状态，不重跑已拒绝的 loss sweep。
2. 检查每个任务的 `batch_status.json.complete`，再计算 paired metrics。
3. 完成 deployable 几何特征及 PDB-disjoint ranker 门禁。
4. 实现 protein-surface joint anchors；比较 broad、anchor、mixed 三种完整初始化。
5. 只有内部 mean `+0.01`、median >0 且命中改善时，才跑新的 MGD94。

## 报告边界

- oracle-best 必须明确标注，不可称为 ranked top-1。
- native-reading oracle 不可称为可部署模型。
- 五例诊断不能外推为 MGD94 总体结果。
- `0.000x` 的均值差、单一 case 拉动和纯 loss 改善都不算模型改善。
- 实验产物是事实来源；文档只保留能改变下一步决策的结论。
