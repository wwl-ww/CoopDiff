# model-v3 loss 与三元相对位姿耦合诊断

## 诊断范围

本诊断基于 `model-v3` 分支提交 `9002dda`，也就是 `rank` 与 `protac` 的共同祖先。没有引用后续 rank 分支新增的 AB consistency 实现。目标是判断：局部 E3--配体、POI--配体速度监督，是否与最终组合出的 E3--POI 相对位姿脱节。

## 当前训练目标

生产训练入口是 `TriCoopDiffTrainer.compute_loss`（`tricoopdiff/train.py`）。每个 flow 状态输出五类监督量：

```text
E3--L translation       loss_e3l_tr
E3--L rotation          loss_e3l_rot
POI--L translation      loss_pol_tr
POI--L rotation         loss_pol_rot
shared PROTAC torsion   loss_score_tor
```

四个刚体项和 torsion 项分别与 SDPI flow 的局部目标做 MSE。若模型输出 residual，还会加入 `residual_penalty * residual_weight`。`ConsensusTrainer` 将两个专家 torsion auxiliary 项的权重设为 0，因此 `loss_e3l_tor` 和 `loss_pol_tor` 在生产训练中不进入总 loss。

总目标实际上是：

\[
L_{local}=L_{E3L}^{tr}+L_{E3L}^{rot}+L_{POL}^{tr}+L_{POL}^{rot}
          +L_{tor}+\lambda_rL_{residual}.
\]

代码没有对 E3--POI 的相对旋转、相对平移、body frame、蛋白–蛋白接触图或界面残基做独立监督。`direction_disagreement` 只是 detached 诊断量，不产生梯度。

## 标签和坐标变换

`SDPIFlow.path`（`tricoopdiff/sdpi_flow.py`）从同一组 coupling 生成一个 flow 状态：

1. 分别为 E3--L 和 POI--L 生成旋转、中心和 torsion 的 endpoint 插值；
2. 用 `assemble_interface_pair` 组合出当前 POI 坐标和 `body` 旋转；
3. E3--L 标签保持在 E3/global frame；
4. POI--L 标签通过 `body.T` 映射到 E3/global frame。

对应代码是：

```python
target['e3_ligand'] = {'tr': tr[0:1], 'rot': rot[0:1]}
target['poi_ligand'] = {
    'tr': tr[1:2] @ body.T,
    'rot': rot[1:2] @ body.T,
}
```

这说明标签构造目前有统一的 frame 约定；仅从代码审计还没有发现一个可以直接认定为 `body` 或 `body.T` 写反的错误。

## 采样时的组合路径

`SDPIFlow.step` 将归一化预测还原成物理速度，然后对两个 branch 各自构造刚体变换：

```python
rotation = exp(-dt * velocity['rot'])
center = old_center - dt * velocity['tr']
offset = center - rotation @ old_center
```

随后：

```python
poi, increment = assemble_interface_pair(poi, ra, ta, rb, tb)
body = increment @ body
```

因此训练比较的是两个局部速度，采样执行的却是：

\[
D_{AB}=D_{E3L}\circ D_{POL}^{-1}.
\]

这两个函数并不等价。即使两个局部速度的 MSE 都下降，组合后的 AB 位姿仍可能有较大误差；两个局部误差也可能在组合时互相抵消。因此“局部速度监督与融合后的相对位姿脱节”是一个成立的结构性风险，而不是仅凭 loss 名称推测出来的现象。

## 代码中已经存在的直接验证路径

model-v3 已经包含实验性 trajectory objective（`tricoopdiff/trajectory_experiment.py`）：

- `differentiable_sdpi_step` 与 `SDPIFlow.step` 使用相同的刚体组合顺序，并保留 autograd；
- `TrajectoryConsistencyTrainer._trajectory_loss` 连续执行多个 reverse step；
- 每一步同时比较 ligand、POI 坐标和 `body` 与解析 SDPI path 的目标；
- 相关配置是 `trajectory_steps`、`trajectory_weight` 和 `trajectory_min_time`。

这条路径正好可以检验首要假设。它与单步局部 velocity loss 的差别是：前者直接监督“预测更新后得到的三元状态”，后者只监督 branch velocity。

## 当前结论

1. 没有发现足以单独解释问题的明显 frame 符号错误。
2. 生产 loss 确实没有对组合后的 E3--POI 相对位姿做直接监督。
3. `residual_penalty` 只限制偏离 frozen proposal 的幅度，不能表达蛋白–蛋白协调。
4. model-v3 已有可微 trajectory loss，可以作为最小改动的诊断实验，不需要先改采样器。

## 建议的诊断实验

固定相同数据、初始化、flow coupling 和验证集，比较以下三臂：

| 实验 | 训练目标 | 目的 |
|---|---|---|
| local | 现有 `L_local` | 基线，复现现有模型 |
| one-step | `L_local + λ·L_AB` | 判断单步组合误差是否是主要瓶颈 |
| rollout | `L_local + λ·L_trajectory` | 判断多步累积误差是否是主要瓶颈 |

每臂都必须报告训练 loss 之外的指标：AB relative rotation/translation error、AB contact F1、三元 DockQ top1、AB DockQ top1，以及 E3--L / POI--L 是否退化。只有后面这些指标改善，才能认为新增 loss 服务于目标。

## 通用验证子集

诊断阶段使用通用验证索引 `data/ternary_val.json`，不再用小训练子集作为主要判断依据。该索引有 253 个 entries、65 个官方 cluster representative。为控制 GPU 时间，固定选择 49 个代表 case，按官方 representative 顺序均匀抽取，生成的封存清单是：

```text
results/loss_diagnosis_20260924/general_val49.json
```

这个子集只用于固定 checkpoint 的比较和指标报告，不参与训练、不用于选择 native 或 top1。若 49-case paired 结果仍然稳定，再扩展到完整 253-case validation 做最终确认。
