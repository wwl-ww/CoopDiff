# TriCoopDiff 模型与坐标契约

更新：2026-10-05。本 worktree 基于 `1a9c6f6` 的 v9，用约束速度模型替换原运动输出路径。蛋白刚体是任务边界。

## model-4.5：约束驱动的联合速度

```text
同一三元状态 q_t（E3固定、POI刚体、ligand刚体＋torsion）
  ├─ 冻结条件 DiffDock：E3–ligand ─┐
  └─ 冻结条件 DiffDock：POI–ligand ├─ 特征与预条件后的运动建议
三元关系编码、共同时间与来源 ───────┘
                 ↓
界面距离变化率／权重＋二元建议权重
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
不按硬半径删除远距离约束。网络预测有符号距离变化率与正权重。
距离为 `sqrt(||delta||²+.01)`，变化率除以10 Å；预测变化率为 `2*tanh(raw)`。
蛋白锚点是当前刚体的瞬时材料点：运动Jacobian不包含特征引起的锚点重新分配，
但外层训练仍对锚点位置和分配反传。ligand Jacobian含v9对称水平torsion切向量，
包括相同的质心/spin去除和 `1e-6` 相对惯性正则；以实际更新器有限差分验收。

```text
min_v  Σ_interfaces ||J_i v-b_i||²_Wi
     + ||A_E v-u_E||²_WE + ||A_P v-u_P||²_WP
     + Σ_torsion_sources ||v_tor-u_source||²_Wsource + .01 ||v||²
```

torsion来源为三元键上下文和两个专家。每个界面按行数、每个刚体块按3分量、每种torsion来源
按键数归一化，避免节点数决定相对贡献。权重为 `.001+sigmoid(raw)`，不等于校准概率。
正规方程在float64求解后还原dtype；正阻尼在归一化坐标中各向同性，避免伪逆秩截断，保留SE(3)等变性。
诊断记录正规方程残差、加权约束残差、专家权重、约束行数和自由度数。

## model-4.5 运行与版本

本分支仅构造约束速度模型，不再提供旧运动头开关；GPU号为示例，运行前检查占用。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_entry.py \
  --train_data data/ternary_train.json --val_data data/ternary_val.json \
  --output_dir results/model45_train \
  --training-arm A1 --cache-only --cache_dir /absolute/path/to/compatible_cache \
  --batch_size 1 --lr 1e-4 --device cuda
python tests/test_consensus.py
python scripts/run_cpu_tests.py --timeout 180
CUDA_VISIBLE_DEVICES=0 python scripts/validate_consensus_model.py \
  --cache-dir /absolute/path/to/compatible_cache --out results/model45_smoke
```

A1/B/C来源协议、20步终点训练、辅助恢复和已有结构监督不变。
`expert_coordinate_grad` 近似默认与基线一致且可显式切换。新profile要求checkpoint≥37，
并记录 `constraint_solver_contract=joint_distance_velocity_v1`，绑定Jacobian、尺度、阻尼及归一化。
旧profile元数据仍可审计，但不能在本分支构造旧运动模型；旧checkpoint训练/推理使用model-v4。拒绝通过resume改架构。
真实smoke只用训练索引的一例和已有缓存，验证专家冻结、梯度、训练/采样终点一致、
蛋白刚性和EMA恢复。`smoke_only.pt`不作为质量选优权重。

2026-10-05 验收：46个CPU测试文件共432项通过（全量运行后定向复跑修正项）。
`4X28_B_C_FDA`真实冻结专家、2次优化、20步采样通过，蛋白内距离最大误差
`5.15e-5 Å`，峰值CUDA分配约1.52 GB；完整记录在
`results/model45_smoke_clean/summary.json`。这是数值和执行契约验收，不是质量评估。

## model-4.5 局限与下一阶段

- 当前是瞬时距离变化率的单模式原型，没有预测完整终态接触图。
- 共线／共面／对称布局存在未观察的运动，重合点的距离Jacobian可能为零。
  阻尼保证有限，不创造缺失信息；后续可比较有向约束，不能将稳定性写成可辨识性。
- 权重可能学成忽略难约束，需要分析权重、残差和运动范数；权重下限只是数值保护。
- 保留16锚点和固定模板；瞬时线性约束不保证有限步严格保持接触或无碰撞。
- 代码验收和单例优化不证明泛化／DockQ改善。

下一阶段固定共同编码器初始化、数据/来源/seed/优化步数/候选预算，比较v9与新profile；
分别报告E3–L、POI–L、PP DockQ、碰撞、好界面保留率、时间和峰值显存。
仅在这一对照支持时增加多模式约束和按未确定自由度分配搜索。


原直接运动头、先验相加和历史运动分支已删除；新求解器直接构造，不采用“先创建旧模块再删除”的方式。旧运动头专项测试随代码退役，公共几何、专家与流更新测试保留。
