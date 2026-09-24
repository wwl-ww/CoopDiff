# model-v3 loss 诊断分支交接

## 分支与基线

- 当前分支：`diagnose/model-v3-loss`
- 基线提交：`9002dda`（`model-v3`，也是 `rank` 与 `protac` 的共同祖先）
- 当前分支提交：
  - `f0984b2`：记录局部速度 loss 与组合位姿脱节的代码诊断
  - `89a10fe`：定义通用验证子集协议
- `rank` 后续提交没有合并进本分支。尤其不要把 rank 分支上的实验性 AB loss 实现直接当成 model-v3 基线。

## 诊断目标

首要假设是：局部 E3--配体和 POI--配体速度监督，与采样时由两个刚体变换组合出的 E3--POI 相对位姿不一致。

当前代码的生产 loss 监督四个刚体局部速度和一个共享 linker torsion：

```text
loss_e3l_tr + loss_e3l_rot
loss_pol_tr + loss_pol_rot
loss_score_tor
residual_weight * loss_residual
```

`ConsensusTrainer` 把两个专家自己的 torsion auxiliary 权重设为 0。没有独立的 E3--POI SE(3)、body frame、蛋白–蛋白 contact map 或界面残基 loss。

flow 标签在 `tricoopdiff/sdpi_flow.py:path` 中构造，POI 分支用 `body.T` 映射到 E3/global frame；采样在 `tricoopdiff/sdpi_flow.py:step` 中用：

```python
poi, increment = assemble_interface_pair(poi, ra, ta, rb, tb)
body = increment @ body
```

目前没有发现明确的 `body`/`body.T` 符号错误。更可信的风险是 loss 只约束 branch velocity，最终关心的组合相对位姿却没有直接进入生产目标。

完整分析见：[model-v3-loss-coupling-diagnosis.md](model-v3-loss-coupling-diagnosis.md)。

## 固定诊断数据

通用验证源索引：

```text
data/ternary_val.json
```

该索引包含 253 个 entries、65 个官方 cluster，且 metadata 中列出的 MGD benchmark PDB 已排除。诊断阶段使用固定的 49-case representative 子集：

```text
results/loss_diagnosis_20260924/general_val49.json
```

这个文件是生成 artifact，不是训练源代码；不要手工修改。所有 loss 变体必须使用同一份清单、同一组 seed、同一 walkers/steps，才能做 paired comparison。native 结构只能用于最终指标计算，不能用于训练或选择候选。

## 已完成的相关实验

此前在 `rank` 分支上做过一个快速 AB consistency ablation：

- baseline checkpoint：`results/ab_loss_ablation_baseline_20260924/best_model.pt`
- AB consistency checkpoint：`results/ab_loss_ablation_consistency_20260924/best_model.pt`
- 两者使用 64 train clusters / 64 validation clusters，属于快速筛选，不是最终通用验证。

MGD 评估结果目录：

```text
results/ab_loss_mgd_eval_baseline_20260924/
results/ab_loss_mgd_eval_baseline_20260924_shard1/
results/ab_loss_mgd_eval_consistency_20260924/
results/ab_loss_mgd_eval_consistency_20260924_shard1/
```

baseline 已完成 94 cases；AB consistency 当时已完成 81 个 unique cases，剩余分片没有形成完整的 94-case paired 结果。因此这些结果只能作为方向性证据：在共同完成的 81 cases 上，AB consistency 的蛋白–蛋白 DockQ top1 均值约提高 0.016，但三元 DockQ top1 只提高约 0.003，尚不足以作为最终结论。

## model-v3 中已有的可复用实现

`tricoopdiff/trajectory_experiment.py` 已有实验性可微 rollout objective：

- `differentiable_sdpi_step` 复用与采样相同的刚体组合顺序；
- `TrajectoryConsistencyTrainer` 对多个 reverse step 后的 ligand、POI 和 body 与解析 SDPI path 做比较；
- `StagedPOITrajectoryTrainer` 提供中后期 POI/body 约束窗口。

这条实现是当前最适合诊断首要假设的路径。不要先修改冻结 DiffDock 专家或改变采样器；先做 loss-only 对照。

## 推荐下一步

在当前分支上固定 49-case validation，做三组最小对照：

1. `local`：现有生产 loss；
2. `one-step`：local loss 加单步组合后的 AB/body/POI state loss；
3. `rollout`：local loss 加 `TrajectoryConsistencyTrainer` 的短 rollout loss。

每组都要报告：

- AB relative rotation/translation error；
- 蛋白–蛋白 contact F1 或 POI DockQ；
- 三元 DockQ top1 和阈值命中数；
- E3--L、POI--L 是否退化；
- 每个 case 的 paired delta，而不是只看平均训练 loss。

只有通用验证集上的 AB 和最终 top1 同时改善，才保留新增 loss。若只降低训练/验证 loss而 DockQ 不升，应视为代理目标失效。

## 工作区与后台任务注意事项

此前未跟踪的支线脚本和说明已存入 stash：

```text
stash@{0}: wip: untracked experiment scripts before loss diagnosis branch
```

需要恢复时先确认文件用途，再执行 `git stash show --stat stash@{0}`；不要直接覆盖当前诊断分支。MGD 评估是在切换分支前启动的，重新启动评估时应明确指定工作树和 checkpoint，避免把 rank 分支的 checkpoint 与 model-v3 代码混用。

