# TriCoopDiff 模型与坐标契约

更新：2026-10-05。本 worktree 基于 `1a9c6f6` 的 v9，用约束速度模型替换原运动输出路径。蛋白刚体是任务边界。

## model-4.5：约束驱动的联合速度

```text
同一三元状态 q_t（E3固定、POI刚体、ligand刚体＋torsion）
  ├─ 冻结条件 DiffDock：E3–ligand ─┐
  └─ 冻结条件 DiffDock：POI–ligand ├─ 特征与预条件后的运动建议
三元关系编码、共同时间与来源 ───────┘
                 ↓
预测终态距离／相对权重＋二元建议权重
                 ↓
当前距离与预测终态之差 → 时间相关的距离变化率
                 ↓
可微加权最小二乘 → 独立刚体速度＋唯一共享 torsion
                 ↓
原 SDPIFlow 对称水平 torsion / 刚体更新器，反向20步
```

profile 为 `ternary_constraints_v1`。复用 v9 的 FiLM、局部蛋白几何、专家标量／向量、
16个蛋白软锚点和关系主干。两个二元读取来自同一个冻结 DiffDock 参数实例；无 LoRA。
求解器替换直接运动头和先验相加模块，不执行两套更新。当前仍以 rollout 终点监督训练，
未添加 flow-matching 目标，不宣称获得热力学采样分布。多模态接触拓扑、校准不确定性、
自由能和主动候选分配尚未实现。

归一化未知量 `v=[t_L, omega_L, t_P, omega_P, theta]`，维度 `12+K`。
平移乘受体尺度 `s_E/s_P`，角速度和 torsion 乘 π，均为正向时钟速度；采样器应用 `-dt`。
专家A的映射为 `[t_L, omega_L]`。令 `lever=c_P-c_L`，专家B的映射为：

```text
B.tr  = s_E/s_P * t_L - t_P + π/s_P * (omega_P × lever)
B.rot = omega_L - omega_P
```

B平移定义在ligand质心，不能直接当作POI平移。该映射与 `independent_targets` 互逆。
专家刚体建议用有界各向同性压缩，torsion用tanh；不把原始score当作物理力。

约束覆盖E3–L、POI–L、E3–POI，接收端保留完整ligand原子／POI残基，对端为蛋白锚点，
不按硬半径删除远距离约束。网络预测正的终态距离与相对权重。
距离为 `sqrt(||delta||²+.01)`，用10 Å归一化。预测归一化终态距离 `d_hat=softplus(raw)`；
正向时钟的目标变化率为 `2*tanh((d_current-d_hat)/(2*max(t,.2)))`。
采样应用负时间步，因此距离过大时靠近、过小时远离；时间下限避免末步奇异。
这是有界反馈速度场，不是保证每步误差下降的全局最优装配器。
蛋白锚点是当前刚体的瞬时材料点：运动Jacobian不包含特征引起的锚点重新分配，
但外层训练仍对锚点位置和分配反传。ligand Jacobian含v9对称水平torsion切向量，
包括相同的质心/spin去除和 `1e-6` 相对惯性正则；以实际更新器有限差分验收。

```text
min_v  Σ_interfaces ||J_i v-b_i||²_Wi
     + ||A_E v-u_E||²_WE + ||A_P v-u_P||²_WP
     + Σ_torsion_sources ||v_tor-u_source||²_Wsource + .01 ||v||²
```

额外torsion建议仅来自两个冻结专家，三元键上下文只预测其相对权重；自由三元torsion速度头已删除。
所有torsion仍参与界面距离Jacobian，因此三元几何也能决定torsion。
每个界面按行数、每个刚体块按3分量、每种torsion来源按键数归一化；原始权重
为 `.001+sigmoid(raw)`，最后对全部约束权重归一化到总和1，避免整体降权变成减速开关。
这些是相对残差权重，不是校准概率。
正规方程在float64求解后还原dtype；正阻尼在归一化坐标中各向同性，避免伪逆秩截断，保留SE(3)等变性。
诊断记录正规方程残差、加权约束残差、专家权重、约束行数和自由度数。

## 单结构监督与数据隔离

每个体系只提供一个观测结构。模型每步的预测终态距离都受到该结构的监督，
而非为假想多模态构造标签。以当前软锚点的材料权重计算真实结构中的对应锚点，
标签构造停止梯度；真实坐标不进入模型forward。监督为归一化距离的`log1p`上的
smooth-L1，先对每个界面取均值，再对三个界面等权平均；所有界面共同选取同一个
等价配体原子置换。配体使用现有运动学可达目标，蛋白保持输入刚体形状。
每段rollout的距离监督按实际步数平均，乘固定0.1加入原终点损失；不乘模型置信权重。
恢复分支沿用原有分支权重。随机初态与扰动是恢复任务，不是独立结构观测。

索引构建保留DeepTernary两个来源的聚类关系，再传递合并同PDB条目后划分。
`results/model45_partition`保留全部7087个可用条目：训练6705，验证382；
PDB重叠0，原始两个来源共1985个聚类关系审计通过。没有重新推断额外序列同源簇。
训练入口强制检查两个索引与原始来源聚类，checkpoint保存索引及来源聚类SHA256，
恢复时拒绝划分变化。旧默认索引有83个PDB交集，不得用于新架构质量对照。
划分改变后旧三元权重不能作为干净验证基线；对照双方需在相同新训练集重新训练。

## model-4.5 运行与版本

本分支仅构造约束速度模型，不再提供旧运动头开关；GPU号为示例，运行前检查占用。
本地A1/B/C启动脚本也默认读取`results/model45_partition`，可用`PARTITION_DIR`显式指定
其他已审计划分；启动任何GPU进程前先进行隔离检查。

```bash
python scripts/make_deepternary_index.py \
  --reuse-index data/ternary_train.json data/ternary_val.json \
  --index-dir results/model45_partition
CUDA_VISIBLE_DEVICES=0 python scripts/train_entry.py \
  --train_data results/model45_partition/ternary_train.json \
  --val_data results/model45_partition/ternary_val.json \
  --output_dir results/model45_train \
  --training-arm A1 --cache-only --cache_dir /absolute/path/to/compatible_cache \
  --batch_size 1 --lr 1e-4 --device cuda
python tests/test_consensus.py
python scripts/run_cpu_tests.py --timeout 180
CUDA_VISIBLE_DEVICES=0 python scripts/validate_consensus_model.py \
  --train-index results/model45_partition/ternary_train.json \
  --cache-dir /absolute/path/to/compatible_cache --out results/model45_target_smoke
```

A1/B/C来源协议、20步终点训练、辅助恢复和已有结构监督不变。
`expert_coordinate_grad` 近似默认与基线一致且可显式切换。新模型要求checkpoint≥38，
并记录 `constraint_solver_contract=supervised_terminal_distance_v2` 和 `target_geometry_loss_weight=0.1`。
v37自由变化率权重只能在提交`13799b6`运行，不可解释为终态距离。
旧profile元数据仍可审计，但不能在本分支构造旧运动模型；旧checkpoint训练/推理使用model-v4。拒绝通过resume改架构。
真实smoke只用训练索引的一例和已有缓存，验证专家冻结、梯度、训练/采样终点一致、
蛋白刚性和EMA恢复。`smoke_only.pt`不作为质量选优权重。

2026-10-05 验收：46个CPU测试文件共439项通过（全量运行后定向复跑修改项）。
真实训练案例`4X28_B_C_FDA`完成2次优化和20步采样；专家冻结、所有可训练参数梯度有限、
训练/采样终点一致及EMA checkpoint恢复通过。蛋白内部距离最大误差`4.39e-5 Å`，
峰值CUDA分配约1.56 GB。距离监督损失`0.49512 → 0.49497`，仅为执行验证，
不代表泛化改善。记录在`results/model45_target_smoke/summary.json`；
划分审计在`results/model45_partition/partition_audit.json`。

## model-4.5 局限与下一阶段

- 当前预测随状态更新的终态距离，并非一次预测后全程锁定的装配图；不学习真实多态占比。
- 软锚点对应关系依赖当前状态，可能聚集或漂移；监督目标停止梯度避免直接操纵标签，但不保证锚点可辨识。
- 共线／共面／对称布局存在未观察的运动，重合点的距离Jacobian可能为零。
  阻尼保证有限，不创造缺失信息；后续可比较有向约束，不能将稳定性写成可辨识性。
- 权重可能学成忽略难约束，需要分析权重、残差和运动范数；权重下限只是数值保护。
- 保留16锚点和固定模板；瞬时线性约束不保证有限步严格保持接触或无碰撞。
- 代码验收和单例优化不证明泛化／DockQ改善。

下一阶段固定共同编码器初始化、数据/来源/seed/优化步数/候选预算，比较v9与新profile；
分别报告E3–L、POI–L、PP DockQ、碰撞、好界面保留率、时间和峰值显存。
先验证终态距离监督和相对权重归一化；当前单结构数据下暂不加入多模式分支。


原直接运动头、先验相加和历史运动分支已删除；新求解器直接构造，不采用“先创建旧模块再删除”的方式。旧运动头专项测试随代码退役，公共几何、专家与流更新测试保留。
