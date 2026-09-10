# 低 DockQ 根因排查交接

## 目标与当前结论

本轮工作的目标是解释 TriCoopDiff 在训练 loss 收敛后仍得到很低的全原子
DockQ。当前最强证据不是单一化学或初始化缺陷，而是训练状态与自由采样状态的
分布错位：模型在解析的 teacher path 上降低了速度 MSE，但自由 rollout 很快离开
该路径，随后刚体速度方向失效，尤其是 POI 装配。

全原子 DockQ 重建链已经用原生坐标验证。`3CXH_C_E_6PH`、
`6U9D_Q_R_TPP`、`4I0P_D_H_PGE` 均得到 DockQ=1、Fnat=1、LRMSD=0，
这排除了这三个 native 自检案例中的明显坐标写回/调用错误；不代表所有预测
构象、原子映射和所有 case 已获得完整验证。

本次交接已完成 rollout augmentation、off-path target 审计和单分支 endpoint
translation 小试验，均未证明精确 DockQ 改善。下一步是先修复实验比较的可重复性、
建立精确指标闭环，再验证双界面联合 endpoint 目标的数学与训练可行性。
联合目标目前尚未实现，也没有成功的 reflow 或 endpoint prediction 模型。

本文结果路径是实验产物的逻辑标识，不是接手设备必须存在的目录。
Git 提交包含源代码和本文记录的数值结论，不包含数据集、缓存、checkpoint 或完整
实验 JSON。接手者需要重新生成或另行取得这些产物，不能把 clone 等同于恢复实验。

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
results/overnight_root_cause_v3_run04
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
results/overnight_root_cause_v3_run04/fixed_validation_evaluation.json
results/overnight_root_cause_v3_run04/paired_contrasts.json
results/overnight_root_cause_v3_run04/raw_expert_factorial.json
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
results/root_cause_v2/round2/initialization_contract_audit.json
results/root_cause_v2/round2/paired_endpoint_expert.json
results/root_cause_v2/round2/diffdock_v2_binary.json
results/root_cause_v2/round2/matched_conformer_geometry.json
results/root_cause_v2/round2/matched_reference_dsm.json
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
results/root_cause_v2/round2/consensus_correction_audit.json
results/root_cause_v2/round2/rollout_teacher_divergence.json
results/root_cause_v2/round2/velocity_calibration_8cases.json
results/root_cause_v2/round2/factorial_initializer_evaluation.json
```

## Rollout augmentation 小试验

已实现 `RolloutAugmentedTrainer`，它在普通 teacher state 中按概率混入当前模型
从更高时间点自由积分 1 或 5 步后到达的 off-path state。实验均从上述最佳
checkpoint 的 EMA 权重开始，用 32 个训练 cluster、8 个验证 cluster 和 50%
混合率做短程微调。

1 步 rollout（5 epochs）的最佳 checkpoint 在 8-case × 2-seed、20-step
自由采样上没有改善：近似 DockQ 为 0.06925（base 0.06940），中心误差也基本
不变。5 步 rollout（3 epochs）的最佳 checkpoint 有弱正信号：近似 DockQ
0.07407（base 0.06939），ligand/POI center error 分别从 31.37/41.48 Å 降至
31.23/41.29 Å，但 PROTAC RMSD 从 2.51 Å 升至 2.55 Å。改善远小于总体刚体
误差，且尚未用精确 DockQ 证实，不能视为根因修复或据此启动长训练。

这说明单纯把同一 coupling 的解析速度目标搬到模型产生的 off-path state，监督
信号仍可能不充分。下一步应优先验证 reflow/重新配对的 off-path target，或改为
直接预测去噪 endpoint；任何候选都必须以 free-rollout 精确 DockQ 选型。

进一步的 off-path target 审计提供了差异证据，但没有证明原 flow-matching
目标在数学上错误。4 case × 2 coupling seeds、t≈0.75/0.50/0.25 上，E3--ligand translation 的原
coupling target 与当前状态直接返回 clean endpoint 的 target 平均方向余弦仍为
0.972，但尺度投影中位数仅为 0.595；rotation 的方向余弦只有 0.293。模型预测
对 endpoint target 的方向余弦分别为 0.323 和 -0.032。teacher-path 自检中
translation 方向/尺度均为 1，证明 translation endpoint 计算没有坐标系错误。

直接使用 endpoint translation 会因除以剩余时间在低 t 产生奇异大 target，短试验
的训练 loss 从 43.6 增至 478.3，已提前停止。有界版本只在 t>=0.25 使用 target，
将修正限制在一个受体尺度并按 25% 混合，训练恢复稳定；但 8-case × 2-seed
自由采样中 ligand center error 仅从 31.35 降到 31.27 Å，POI error 从 41.09
升到 41.42 Å，近似 DockQ 从 0.0732 降到 0.0703。这次短试验未显示收益。它尚不足以证明单分支修正必然失败，也不足以确认
POI 误差变化由双界面不一致造成；联合目标是待检验假设。

相关结果：

```text
results/rollout_pilot_32x8_v1/free_rollout_evaluation.json
results/rollout_pilot_32x8_v1/free_rollout_latest_evaluation.json
results/rollout_pilot_32x8_steps5_v1/free_rollout_evaluation.json
results/root_cause_v2/round3/offpath_target_audit.json
results/endpoint_translation_bounded_32x8_v1/free_rollout_evaluation.json
```

## 接手必须知道的实验局限

- `audit_offpath_targets.py` 从晶体内部构象的 teacher t=1 出发，自由走 15 步；
  它不是从盲 ETKDG 初始化出发的完整推理评估。`offpath_center_error_A` 实际
  记录距 native 中心的距离，不是距同时间 teacher 中心的偏差。
- translation endpoint target 是 `(current_center - native_center) / t`，
  符号对应 clean→prior 的 forward velocity；实际 solver 用负方向积分。
  teacher 自检误差约 2.9e-6。rotation 是全配体 Kabsch 最佳刚体旋转除以 t，
  不能精确代表带 torsion 的瞬时切向速度：2-case 自检余弦约 0.897，误差约
  1.226。应先关闭 torsion 做精确验证，再推导 gauge 修正。
- 沿原路径的速度和重新连向终点的速度属于不同路径定义，两者有差异并不自动
  说明训练目标错误。本文 endpoint 用到了真实终点，只能作训练监督或 oracle
  诊断，不能在盲推理中读取 native 坐标。
- 各 pilot 训练 seed 不同，teacher validation 每轮重新采样 coupling/time。
  目前并非严格同种子的训练消融；`best.pt` 仍由随机 teacher loss 选择。
- `evaluate_rollout_pilot.py` 的 base 在不同运行中也有指标波动。虽然传入相同
  seed，尚未验证两模型的完整起始坐标逐元素一致。应保存并复用初始化状态和
  构象，检查坐标校验值，再做配对统计。现有不足 1 Å 的差异不应作因果结论。

## 2026-09-09 本地接手进展

已建立严格确定性的全原子 exact DockQ evaluator。native 重建自检 DockQ/Fnat
均为 1；相同初始化在多 GPU 上逐元素一致。默认 CUDA `index_add_` 的微小
非确定性会沿自由轨迹放大，因此后续比较全部启用确定性 scatter 路径并记录
初始化、checkpoint、index 和 DockQ 实现哈希。

输出基不是当前瓶颈：18 个状态的几何子基全部满秩，teacher 刚体目标最大相对
投影误差为 1.14e-15。主要失败在旋转系数学习：EMA rotation cosine 平均
-0.120，36 项中 23 项为负，30 项幅度不足目标四分之一。固定单状态又能把
loss 降到 0.001--0.02，说明局部表达能力存在，但单点速度 MSE、状态信息或
条件平均没有产生正确的自由轨迹向量场。

解析 torsion gauge 与 solver 的有限差分误差约 1e-9，production teacher path
无需修改。旧 off-path transported target 会运输 teacher gauge；新实验变体先
移除 teacher 公共 spin，只运输刚体项，再按 rollout 配体构象和 torsion rate
重算 spin 并同时加入两个界面角速度。严格 RNG 配对、PDB 前缀去重后，短程
`rollout + gauge` 相对 teacher 曾在 49 个 case 配对中提升 46 个，平均 exact
DockQ +0.00243；但只比较新旧 gauge、保持 rollout 相同时，首批 28 个配对仅
提升 15 个，平均约 +0.00057。当前证据更支持“短程 rollout 有益、gauge 修正
贡献较小且不稳定”，不能把组合收益归因于 gauge。

另一个稳定结论是 checkpoint 时点：两个目标变体在训练 seed 22001 都于
epoch 1 达峰，在 seed 22002 都于 epoch 6 达峰，而 teacher validation loss
不能选出这些点。下一步应使用 PDB/homology 去重的独立 exact DockQ 面板验证
短程 rollout，并测试轨迹、终点或界面接触监督；不要直接扩大长训练。
- 短训 96 step 使用 EMA decay=0.999，初始权重仍占约 90.8%；小变化可能受
  EMA 滞后影响。下一轮同时比较 raw 与 EMA，先验证梯度、参数变化和可拟合性。
- 无界 endpoint 版出现大 loss 与 1/t 放大机制一致，但未记录逐样本 t/target
  norm 来定位全部异常来源，不能把这一推断写成完成的因果验证。
- 最新 endpoint trainer 已增加最小时间、混合和限幅字段。早期无界版本的
  checkpoint 没有这些 runtime 字段，不能用最新版 loader 无修改加载；需要
  对应历史实现或显式迁移。禁止静默套用新默认值。

## 代码导航与已完成验证

| 文件 | 职责及当前状态 |
|---|---|
| `tricoopdiff/train.py` | 提取 `_training_state` / `_forward_state` 扩展点，默认仍走解析路径 |
| `tricoopdiff/consensus_experiment.py` | factorial checkpoint；短 rollout 增强；有界 E3 endpoint translation 实验 trainer |
| `scripts/train_rollout_pilot.py` | 从 base EMA 开始、fresh optimizer、lr=5e-5；32 train cluster / 8 val cluster；保存 history、best/latest |
| `scripts/evaluate_rollout_pilot.py` | checkpoint 加载和 8 case × 2 seed × 20 step 自由采样；仅近似 DockQ |
| `scripts/audit_offpath_targets.py` | off-path E3 刚体 target 比较及 teacher-path 对照；不含联合 POI target |
| `tricoopdiff/sdpi_flow.py` / `tricoopdiff/rigid.py` | 路径切向速度、SO(3) 与双界面逆装配；推导必须以此为准 |

入口参数：审计使用 `--checkpoint --out --cases --paths --seed`；训练使用
`--base-checkpoint --out --train-cases --val-cases --epochs --seed
--rollout-probability --rollout-steps --offpath-target`，有界 target 另有
`--rollout-min-time --endpoint-blend --endpoint-correction-ratio`；评估使用
`--base --pilot --out`。这些研究脚本的数据/模型定位仍有硬编码，移植前应参数化
路径，并保持相同数据成员、化学处理、专家身份和 checkpoint contract。

固定评估成员为 `3CXH_C_E_6PH`、`6U9D_Q_R_TPP`、`4I0P_D_H_PGE`、
`5W7C_A_C_FTT`、`6HS4_A_B_GON`、`4IL6_C_I_LMT`、`7L0F_G_H_GSP`、
`3ZCB_A_B_ATP`；采样 seeds 为 12001/12002。基线为 factorial 的
`ccd_single_bond_frozen_seed1729/best.pt`，不是任意生产 checkpoint。
有界试验训练 seed=14004，3 epochs，rollout probability=0.5，steps=5，
min time=0.25，blend=0.25，correction ratio=1.0；原 5 步增强 seed=11024。

本轮实际通过 Python 编译检查、`git diff --check`、
`python tests/test_model_build.py`、真实数据训练 smoke 和自由采样 checkpoint
加载。尚未添加 endpoint 分支边界/目标的专用自动化测试；模型 smoke 不能替代
这些数学测试。最新 pilot 没有精确全原子 DockQ 评估。

提交导航：`6a5b46f` 为集成修复与根因实验，`731fee6` 为 rollout pilot，
`516e5e3` 为 off-path 审计与有界 endpoint translation 实验。

## 下一位接手者的行动顺序

1. 固定病例、构象、初始化坐标、训练 coupling/time 和种子；对同 checkpoint
   重复评估，建立可重复性基线。将原子映射/写回自检接入完整全原子 DockQ。
2. 在无 torsion 的合成刚体数据上验证双界面 target：反向一步至终点能同时恢复
   ligand 和 POI，body frame 一致；随机整体旋转/平移后结果等变；teacher path
   极小步长的有限差分一致。再加入 torsion gauge，不能直接用 Kabsch 当精确目标。
3. 先验证单 case 可拟合、各分支梯度和 raw/EMA 变化，再以相同训练随机状态做
   原 teacher、原 coupling rollout、联合 endpoint/reflow 的 3–5 epoch 消融。
   配套记录 t、target norm、各分量 loss、每步漂移；明确低 t 的目标与加权定义。
4. 用自由采样的精确 DockQ、LRMS/iRMS/Fnat 和中心误差比较多数病例及配对分布。
   不以 teacher loss 或近似 DockQ 选择结论；确认可复现收益后再投入正式训练。
   正式训练通常以 20–25 epoch 为上限候选，按收敛和真实指标提前停止。
5. ETKDG/matching 增加逐 case 超时及失败记录，避免构象生成阻塞整个评估。

当前工作止于上述审计和短试验；没有待接管的本轮训练任务，也没有实现联合
endpoint/reflow 或新的 endpoint head。保持生产默认值，先完成验证再考虑替换。
