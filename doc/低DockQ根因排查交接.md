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

两套独立 x64 完成后，单套 acceptable 为 38/94 与 37/94，两套池化为
43/94；再加入两套完整 x32 后，统一 192-walker oracle 达到 48/94，mean/median
为 `0.30152/0.23137`。这是搜索覆盖上界，不是模型可部署成功率。

### 4. 小幅 loss 变体没有实际收益

windowed、soft-time、direction、agreement、prior-replay 和 joint-translation
直接监督均未达到 `+0.01` 且未改善稳定命中。典型结果：direction 最佳 epoch 约
`+0.00706`；agreement 全部 240 对为 `-0.00143`；600-epoch prior-replay 的
严格 85 对只有 `+0.00076`。这些路线已关闭，不再扩训。

## 当前设计方向

### 联合生成器

`tricoopdiff/pose_anchors.py` 已提供 proper SE(3) 原语；
`tricoopdiff/expert_anchors.py` 会从两个 frozen DiffDock 刚体 proposal 集合组合
共享 ligand 的 ternary anchor，并做无 native 的碰撞/双侧接触过滤。

候选必须同时包含：接触位置、相对法向、切向/twist；只有距离或质心平移不够。
早期五例的 geometry 选择不稳定，但扩大到 20-case seed 31000 后，`8×8`
geometry-anchor 相对严格配对 broad x8 将 oracle mean/median
`0.06717/0.06348→0.16487/0.12156`、acceptable `0/20→5/20`，case 为
15 胜/3 平/2 负。这是首个通过扩大 DockQ 门禁的生成器改动，支持“联合位置+朝向
多假设”根因判断。同面板 consensus-anchor mean/median 为 `0.12956/0.08772`、
acceptable 5/20，弱于 geometry 但仍通过门禁，适合作为筛选特征或补充分支。
最终应与 broad 混合，不能让 anchor 完全替代宽先验。

替换 ligand 构象生成器不是主要修复：20-case DiffDock conformer 的 oracle
mean/median 为 `0.10023/0.07406`，低于 ETKDG 的 `0.12059/0.08985`，且
acceptable 为 0/20 对 3/20；该路线已关闭。

### 确定性在线候选生成与两级 ranker

该在线初始化模块现正式命名为 **BiTSeed**（Conformer-Diverse Binary-to-Ternary
Basin Seeding）；与 CoopDiff 精修组成系统时称为 **BiTSeed-CoopDiff**。当前 A1/B1/
C1/A2/B2 proposal-oracle 的当前结论、v4 补采语义和正式协议以
[`BiTSeed-Hybrid 在线初始化规格`](BiTSeed-Hybrid在线初始化规格.md)为准。
BiTSeed 名称只覆盖 native-free 候选生成，不包含 oracle、selector 或 post-ranker。

下一代架构不是预存 pose bank，也不是继续线性增加完整 walker。正式推理只接收原始
蛋白、ligand、公开固定 seed 和 frozen checkpoint，当场生成候选；候选生成代码路径
禁止接收 native，正式测试禁止加载 pose cache。每侧各运行 M 个二元专家 pose，以约
`2M` 次专家推理通过共享 ligand 对齐组成 `M²` 个 ternary anchor。候选可按预先冻结
的比例覆盖不同 seed、twist、broad、geometry 和 consensus 模式，不允许按测试 case
人工增删。native 只在预测封存后离线计算候选 oracle 上界。

pre-ranker 先在无 native 的硬物理过滤后，从数百候选中选择少量跨姿态簇 top-K 进入
CoopDiff；post-ranker 再从精修结果中输出可部署 top-1。两级选择都必须 diversity-aware，
不能让一个高分错误模式占满预算。

不要继续只喂 128 维 consensus context。下一版输入至少包括：

- E3–POI、E3–ligand、POI–ligand 的多阈值接触计数/距离分位数；
- 原子穿插与短程排斥；
- 两侧配体接触是否平衡；
- 界面面积和表面法向互补的代理量；
- 二元专家分数/轨迹稳定性、候选簇密度和 conformer strain；
- 模型自身速度残差或跨步稳定性，但部署特征不能使用 native 派生指标。

训练使用 case 内 pairwise/listwise 排序并预测 DockQ 0.23/0.49 命中概率；监督数据只
来自内部 PDB-disjoint 32/8/10 split。MGD94 已用于 BiTSeed development oracle ablation，
不能再称未触碰的 zero-shot holdout，也不得提供 selector 监督标签。执行门禁依次是候选
oracle、pre-ranker retention、固定预算精修 coverage、post-ranker top-1；前一项不过则
不扩张。

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
3. 完成双专家 geometry/consensus 20-case 门禁。
4. 实现确定性在线候选生成并先测候选 oracle 上界，再训练 PDB-disjoint 两级 ranker。
5. 固定精修预算比较 broad、anchor 与 diversity-aware mixed；门禁通过才跑 MGD94。

## 报告边界

- oracle-best 必须明确标注，不可称为 ranked top-1。
- native-reading oracle 不可称为可部署模型。
- 五例诊断不能外推为 MGD94 总体结果。
- `0.000x` 的均值差、单一 case 拉动和纯 loss 改善都不算模型改善。
- 实验产物是事实来源；文档只保留能改变下一步决策的结论。
