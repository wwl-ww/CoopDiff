# CoopDiff 当前任务

更新时间：2026-09-11。这里只保留当前决策、待办和足以复核决策的关键数字。
历史逐轮日志不再写入本文；实验细节以 `results/` 中的报告为准。

## 当前目标

主目标改为 Molecular Glue（MGD）。未来 PROTAC 会加入 warhead/口袋信息，
因此现阶段不使用 PROTAC 的容易提升代替 MGD 的真实困难。

验收门禁：短面板 mean DockQ 至少 `+0.01`、median > 0、没有稳定坏亚组，
并改善 DockQ `>=0.23` 或 top-k 命中。`0.000x` 变化一律不作为进展。

## 已确认结论

- 化学恢复、缓存、图方向、checkpoint 加载和 native DockQ 重建已通过门禁；
  MGD 五例 native 重建均为 DockQ 1.0。
- 低 DockQ 不是单一代码错误。SDPI 的宽先验面对多模态、近似不可辨识的 native
  endpoint；对冲突方向做 MSE 会产生平均化速度。固定状态可记忆，但跨 coupling
  泛化恶化，且同 case/time 的不同 coupling 梯度会冲突。
- translation 是因果瓶颈之一，但不是完整解释。MGD 五例 × 8 seeds 的 baseline
  mean/median 为 `0.04058/0.02613`；translation oracle 为
  `0.06021/0.04600`，rotation oracle 为 `0.06617/0.03108`，而 joint-rigid
  oracle 为 40/40 DockQ 1.0。后者读取 native，只是因果上界。
- leave-one-component oracle 表明 POI rotation 最不可缺少，E3 translation
  相对最可替代；任何单分支仍不充分。下一代生成器应联合提出位置与朝向，不能只做
  translation anchor。
- teacher epoch 2 EMA 的 MGD94 × 32 已完整结束：94/94 case、3008/3008 walker
  有效；oracle-best mean/median `0.21922/0.17427`，33/94 达到 0.23、
  7/94 达到 0.49、1/94 达到 0.8。walker mean 仅 `0.05397`。
  生成覆盖和排序是两个独立瓶颈，oracle-best 不是可部署 top-1。
- 两套独立 MGD94 × 64 均已完整结束，单套 acceptable（DockQ >= 0.23）为
  38/94 与 37/94；两套 x64 池化为 43/94。再与两套 x32 池化后，统一的
  192-walker oracle-best mean/median 为 `0.30152/0.23137`，acceptable
  48/94、medium 17/94、high 1/94。增加样本能提高覆盖，但不替代生成器改进。
- 双 DiffDock 刚体 anchor 已首次通过扩大门禁。20-case seed 31000 的 `8×8`
  geometry-anchor 相对同 seed broad x8，将 oracle mean/median
  `0.06717/0.06348→0.16487/0.12156`，acceptable `0/20→5/20`；case
  15 胜/3 平/2 负，平均提升 `+0.09770`。这证明联合位置/朝向候选是有效生成器
  改动；需在下一分支解决 92/160 几何接受后候选的筛选。
- 同一 20-case 的 consensus-anchor 也通过：oracle mean/median
  `0.12956/0.08772`，acceptable 5/20，case 15 胜/2 平/3 负；但弱于
  geometry，应作为筛选特征或补充分支，而不是当前主选择器。
- 20-case conformer 门禁已否决 DiffDock conformer：其 oracle mean/median
  `0.10023/0.07406`，低于 ETKDG 的 `0.12059/0.08985`，acceptable 为
  0/20 对 3/20；两者池化没有新增 acceptable case。
- 现有 128 维 consensus context 不适合直接排序。首版 listwise ranker 在独立
  test 上 selected/oracle DockQ 为 `0.02436/0.05428`，保留率 44.1%；
  MGD 五例为 `0.02360/0.23112`，保留率 15.2%，低于 walker-0。
  下一版必须加入可部署的接触、冲突和界面互补几何特征。
- 20→40 steps 的五例 oracle-best 平均 `+0.01893`，但中位数只有
  `+0.00086`，由单个 case 主导；增加积分步数不构成稳定修复。

## 已拒绝路线

以下路线均未过实际 DockQ/命中门禁，不再追加训练：windowed、soft-time、
direction、agreement-gated、joint-translation 直接监督、长程 prior-replay，
以及单纯增加 gradient accumulation。最好结果仍只在 `0.000x–0.007x` 范围，
或由单个 case 拉动。标准 teacher Reflow 只能做少步蒸馏，不能解决 prior-endpoint
配对，继续后置。

## 分支收束

`investigate/model-v2-followup` 的使命已完成：排除实现错误和低收益 loss 路线，定位
联合位置/朝向覆盖瓶颈，并用双专家 geometry-anchor 通过 20-case 门禁。所有未完成
任务已停止，其 partial 产物不用于上述结论；下一分支集中把初始化条件做到最好。

## 下一分支

1. 使用确定性在线候选生成，不发布或加载可人工编辑的 pose bank。正式评估从原始
   输入、固定 seed 和 frozen checkpoint 当场生成候选；候选路径不得接收 native。
2. 每侧在线生成 M 个二元 pose，以约 `2M` 次专家成本组合成 `M²` 个 ternary
   anchor；可包含固定比例的 seed、twist、broad、geometry 和 consensus 模式。
   首先离线测候选 oracle 上界，不增加最终 CoopDiff walker 预算。
3. 构建两级 native-free 选择器：硬物理过滤后，由学习式 pre-ranker 选出跨簇的
   diversity-aware top-K 进入 CoopDiff；精修后再由 post-ranker 输出 top-1。
   特征包括专家分数/稳定性、候选共识、接触分布、穿插/排斥、界面互补、双侧平衡
   和 conformer strain。只在 PDB-disjoint internal split 训练，MGD94 仅最终评估。

核心门禁顺序固定为：候选 oracle 上界提升 → pre-ranker oracle retention → 固定预算
精修后的 coverage → post-ranker top-1。若在线候选集合本身没有更好的 oracle，停止训练
ranker；若 ranker 只集中选择同一错误模式，强制按姿态簇分配 top-K。

## 数据与代码约束

- ranker 数据按 PDB 去重，train/val/test 为 32/8/10 case，三者 PDB 两两无重叠；
  MGD94 不参与训练。曾含 `3CXH` 泄漏的 val 结果已移到
  `results/ranker_pose_data/quarantine/`。
- MGD94 中 93 例使用 CCD；`3IQV_A_P_FSC` 因 deposited FSC 缺六个 CCD
  重原子，显式使用 `pdb_inferred_fallback` 并保留在 94 例分母中。
- 实验 factorial-v1 checkpoint 必须显式 `--allow-experimental-checkpoint`，仍校验
  frozen expert、无 LoRA、expert identity 与 EMA；生产 v18 门禁不放宽。
- 旧 checkpoint 未记录 torus 表，当前协议统一使用 deterministic quadrature；
  结果可复现，但不是旧轨迹逐位复放。
- 不创建 worktree；集中使用 `investigate/model-v2-followup`；减少 Git 提交次数。

## 快速验证

```bash
python tests/test_model_build.py
/home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_consensus.py
/home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_pose_anchors.py
/home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_pose_ranking.py
git diff --check
```
