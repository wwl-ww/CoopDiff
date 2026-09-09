# 低 DockQ 根因排查交接

## 目标与当前结论

本轮工作的目标是解释 TriCoopDiff 在训练 loss 收敛后仍得到很低的全原子
DockQ。当前最强证据不是单一化学或初始化缺陷，而是训练状态与自由采样状态的
分布错位：模型在解析的 teacher path 上降低了速度 MSE，但自由 rollout 很快离开
该路径，随后刚体速度方向失效，尤其是 POI 装配。

全原子 DockQ 重建链已经用原生坐标验证。`3CXH_C_E_6PH`、
`6U9D_Q_R_TPP`、`4I0P_D_H_PGE` 均得到 DockQ=1、Fnat=1、LRMSD=0，
因此当前低 DockQ 不是坐标写回或 DockQ 调用错误。

下一项优先实验应是 rollout augmentation：从 teacher path 的较高时间点执行一到
数步当前模型，再在模型实际到达的偏离状态上监督返回同一 coupling teacher path
的速度。不要继续只增加普通 teacher-forced epoch，也不要根据内部近似 DockQ
调采样倍率。

## 已完成的代码修复

1. 修复受体 kNN 的 directed edge 方向、小受体自环以及大坐标下距离消减误差；
   特征缓存版本提升到 v7。
2. 修复 CUDA radius fallback 的源索引扫描和 neighbor cap 语义。
3. 将 componentwise activation clamp 替换为按不可约表示模长限幅，恢复旋转
   等变性；checkpoint 显式记录 clamp mode。
4. 修正动态受体裁剪后空边图导致的非有限值路径，同时保留裁剪诊断。
5. 新增 CCD 配体化学恢复工具及顺序/坐标检查。
6. 新增 chemistry × torsion × expert LoRA 的顺序 factorial 实验框架、固定配对
   validation 和原始专家审计。
7. 修正 factorial 实验的随机数配对和 single-bond torsion coupling 投影。
8. 新增可选的 DiffDock v1.1 构象策略：先执行普通 ETKDGv2，三次失败后才启用
   random coordinates 并执行 MMFF。默认仍是已有的 `etkdgv3_random`。
9. 评估与采样入口新增显式 `translation_velocity_scale` 和
   `rotation_velocity_scale`，默认均为 1.0。它们只用于可复现实验，未改变生产
   推理结果。

## 十小时 factorial 实验

有效运行目录：

```text
/tmp/coopdiff/results/overnight_root_cause_v3_run04
```

10 个 arm 全部完成。主要训练 validation 最优值：

| chemistry | torsion | adaptation | best validation loss |
|---|---|---:|---:|
| old | legacy | frozen | 3.8437 |
| old | legacy | LoRA | 3.8246 |
| old | single | frozen | 3.6974 |
| old | single | LoRA | 3.6877 |
| CCD | legacy | frozen | 3.7360 |
| CCD | legacy | LoRA | 3.7160 |
| CCD | single | frozen | **3.5819** |
| CCD | single | LoRA | 3.5845 |

固定的 22-case × 3-time validation 表明：

- old chemistry 下 LoRA 能显著补偿输入缺陷；
- CCD chemistry 下 LoRA 收益接近零，CCD single-bond 下 torsion 甚至变差；
- CCD 主要改善 translation，但 frozen legacy torsion 的 rotation 变差；
- single-bond torsion restriction 对 rotation 更有帮助。

结果文件：

```text
/tmp/coopdiff/results/overnight_root_cause_v3_run04/fixed_validation_evaluation.json
/tmp/coopdiff/results/overnight_root_cause_v3_run04/paired_contrasts.json
/tmp/coopdiff/results/overnight_root_cause_v3_run04/raw_expert_factorial.json
```

这些是内部 velocity loss，不是 DockQ，不能据此宣称 docking 质量改善。

## DiffDock 初始化与建图审计

原 DiffDock 的关键流程：

- 训练预处理生成 ETKDGv2 conformer，通过 differential-evolution torsion
  matching 对齐 native，再在该构象上施加扩散噪声；
- 推理从输入或生成的 RDKit conformer 出发，uniform randomize torsion，再执行
  Haar rotation 和 Gaussian translation；
- `generate_conformer()` 先尝试普通 ETKDGv2 三次，只有失败后的 `else` 路径才
  设置 `useRandomCoords=True` 并执行 MMFF；
- 配体图包含化学边和 5 Å radius edges；受体边在预处理时建立；cross edges
  根据 `3 * tr_sigma + 20` 动态生成；
- 上游 `crop_beyond` 严格按距离裁剪。当前项目为了避免空图数值崩溃，会保留
  至少 8 个最近残基，因此高噪声图契约并非完全相同。

此前的 `validate_initialization.py` 只在固定共享构象、`no_torsion=True` 时验证
刚体核一致性，没有验证完整 conformer/torsion 分布。

39 个成功盲初始化的实测结果：

- ETKDG+torsion 对晶体构象的 Kabsch RMSD 平均 2.73 Å，90 分位 5.21 Å；
- 5 Å ligand radius edge Jaccard 平均 0.792，最差 0.419；
- 只替换内部骨架并固定 torsion、质心、刚体 pose 和 t=1 后，专家 translation
  输出方向余弦中位数 0.998，rotation 0.300，torsion 0.177。

DiffDock v1.1 初始化在 21 个独立二元 case 上将 ODE ligand RMSD 平均降低
1.27 Å，95% bootstrap CI 为 [-2.56, -0.02]，但三元小样本结果混合，因此没有
替换默认初始化。

进一步生成训练风格 torsion-matched references：20/22 case 成功，距晶体构象
平均 0.56 Å。冻结专家 DSM 上，matched reference 的 translation MSE 反而增加
0.024，rotation 也没有稳定改善，所以没有投入 GPU 训练这条路线。

相关结果：

```text
/tmp/coopdiff/results/root_cause_v2/round2/initialization_contract_audit.json
/tmp/coopdiff/results/root_cause_v2/round2/paired_endpoint_expert.json
/tmp/coopdiff/results/root_cause_v2/round2/diffdock_v2_binary.json
/tmp/coopdiff/results/root_cause_v2/round2/matched_conformer_geometry.json
/tmp/coopdiff/results/root_cause_v2/round2/matched_reference_dsm.json
```

## Rollout 漂移与速度审计

对 `ccd_single_bond_frozen_seed1729` 的 EMA checkpoint，用完全相同的初始化
coupling 比较自由 rollout 与解析 teacher path：

| state | ligand center error | POI absolute RMSD |
|---|---:|---:|
| t=1 | 0.000002 Å | 0.000007 Å |
| first reverse step | 1.65 Å | 4.99 Å |
| t=0.5 | 21.02 Å | 39.19 Å |
| t=0 | 54.62 Å | 73.41 Å |

Ligand internal-distance RMSE 始终约 1.7–2.4 Å，没有随刚体误差同步爆炸；主要
故障是 translation/rotation rollout，而不是逐步破坏分子内部几何。

Teacher-path 上的方向余弦：

| t | translation final/target | rotation final/target |
|---:|---:|---:|
| 0.1 | 0.612 | 0.050 |
| 0.5 | 0.132 | -0.086 |
| 0.9 | 0.619 | -0.076 |

模型降低 rotation MSE 的主要方式是缩小向量，不是学习正确方向。关掉 rotation
在 8-case 代理评估中只有小幅改善；6× translation 是主要代理收益来源。组合
策略将内部近似 DockQ 从 0.0509 提到 0.1331，但 3-case × 2-walker 的精确
全原子 DockQ 均值仅从约 0.0154 变为 0.0159，没有稳定改善。不要把 6×倍率
设为默认值。

相关结果：

```text
/tmp/coopdiff/results/root_cause_v2/round2/consensus_correction_audit.json
/tmp/coopdiff/results/root_cause_v2/round2/rollout_teacher_divergence.json
/tmp/coopdiff/results/root_cause_v2/round2/velocity_calibration_8cases.json
/tmp/coopdiff/results/root_cause_v2/round2/factorial_initializer_evaluation.json
```

## 下一步建议

1. 为 experimental trainer 增加一到数步 scheduled rollout augmentation。
   从 `t + Δt` 的 teacher state 出发，用当前 EMA/model 无梯度走到 `t`，然后在
   到达状态上监督相同 coupling 的 forward velocity。
2. 同时保留普通 teacher states，建议先做 50/50 混合，避免完全丢掉解析路径。
3. 优先使用 CCD + single-bond + frozen expert arm；现有 factorial 表明此时 LoRA
   没有可靠收益。
4. validation 必须同时报告 teacher-path loss、free-rollout center error 和精确
   full-atom DockQ。内部 approximate DockQ 只用于调试，不能用于选 checkpoint。
5. 先用 8 个非 BCR case 做 5-epoch pilot；确认 rollout center error 明显下降后，
   再运行 20–25 epoch 正式对照。ETKDG/matching 必须设置逐 case 超时。

GPU3 UUID 为：

```text
GPU-3ac265b1-130f-cf53-b680-8730370a5881
```

当前 GPU3 已释放。工作树位于 `/tmp/coopdiff/repair_knn`，运行命令应从该目录
执行；模型、数据和实验结果位于 `/tmp/coopdiff`，共享盘版本仅作为源仓库。

## Rollout augmentation 小试验

已实现 `RolloutAugmentedTrainer`，它在普通 teacher state 中按概率混入当前模型
从更高时间点自由积分 1 或 5 步后到达的 off-path state。实验均从上述最佳
checkpoint 的 EMA 权重开始，用 32 个训练 cluster、8 个验证 cluster 和 50%
混合率做短程微调。

一阶段 rollout（5 epochs）的最佳 checkpoint 在 8-case × 2-seed、20-step
自由采样上没有改善：近似 DockQ 为 0.06925（base 0.06940），中心误差也基本
不变。五阶段 rollout（3 epochs）的最佳 checkpoint 有弱正信号：近似 DockQ
0.07407（base 0.06939），ligand/POI center error 分别从 31.37/41.48 Å 降至
31.23/41.29 Å，但 PROTAC RMSD 从 2.51 Å 升至 2.55 Å。改善远小于总体刚体
误差，且尚未用精确 DockQ 证实，不能视为根因修复或据此启动长训练。

这说明单纯把同一 coupling 的解析速度目标搬到模型产生的 off-path state，监督
信号仍可能不充分。下一步应优先验证 reflow/重新配对的 off-path target，或改为
直接预测去噪 endpoint；任何候选都必须以 free-rollout 精确 DockQ 选型。

相关结果：

```text
/tmp/coopdiff/results/rollout_pilot_32x8_v1/free_rollout_evaluation.json
/tmp/coopdiff/results/rollout_pilot_32x8_v1/free_rollout_latest_evaluation.json
/tmp/coopdiff/results/rollout_pilot_32x8_steps5_v1/free_rollout_evaluation.json
```
