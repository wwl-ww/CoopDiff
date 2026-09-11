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

## 正在运行（每项占一张卡，不分片同一测试）

- GPU2：MGD94 × 32，保存 rank features：
  `results/mgd_followup/mgd94_x32_rankfeatures/`
- GPU3：MGD94 × 64，测 walker 数量与覆盖率：
  `results/mgd_followup/mgd94_x64_seed31000/`
- GPU4：base checkpoint 的 MGD94 × 32，对照 teacher 是否只改善少数 case：
  `results/mgd_followup/mgd94_base_x32_seed0/`
- GPU5：teacher 的第二初始化 MGD94 × 32，测随机覆盖稳定性：
  `results/mgd_followup/mgd94_teacher_x32_seed32000/`

每 10 分钟检查一次。完成条件以 `batch_status.json` 的 `complete=true` 为准，
不能把启动时生成的空 summary 当成最终结果。

## 下一步

1. 汇总四个全量任务，报告覆盖率随 walker 数、checkpoint 和初始化 seed 的变化；
   不用 MGD94 选择 checkpoint。
2. 为 ranker 增加不依赖 native 的几何特征：蛋白-蛋白与蛋白-配体接触分布、
   原子穿插/排斥、界面面积代理、两侧结合的平衡性；先做 label-blind 特征审计，
   再在 PDB-disjoint internal train/val/test 上训练。
3. 将 `tricoopdiff/pose_anchors.py` 的 proper SE(3) 原语接入 deployable 的蛋白表面
   patch 候选；同一位置枚举多个 twist，形成联合 position+orientation hypotheses。
4. 固定内部面板比较 broad SDPI、surface-anchor 与二者混合的 coverage/top-k；
   只有过门禁才跑新的 MGD 全量测试。

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
