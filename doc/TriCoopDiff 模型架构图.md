# TriCoopDiff 模型与坐标契约

更新：2026-10-05。`model-4.5` 当前为 checkpoint v39：有限响应驱动的两子步专家流。
蛋白保持刚体，两个二元分支共享冻结 DiffDock 参数，无 LoRA。

## 有限响应控制与实际执行

```text
同一三元状态 q（E3固定、POI刚体、ligand刚体＋共享torsion）
  ├─ 专家A：E3–ligand ─┐
  └─ 专家B：POI–ligand ├─ 基础建议、节点特征、三元编码
                       └─ 终态几何预测 → 可微求解修复场 G(q)
  ├─ 沿A有限试探 → 查询B；沿B有限试探 → 查询A
  └─ 控制器读取交叉响应、预测目标误差变化及三元编码
                  ↓
       顺序偏好 α、时长比例 s、两次修复强度 g1/g2
                  ↓
第一子步：组合基础专家场与修复场 → 刚体/torsion积分 → q_mid
                  ↓
在实际 q_mid 重新读取A、B和几何修复场
                  ↓
第二子步：反转专家偏好、组合新场 → 刚体/torsion积分 → q_next
```

这是保留二元专家的可学习流更新。仍使用 rollout 终点监督，未增加 flow-matching
目标，也不宣称热力学采样分布。控制器读取几何干预后的专家变化，学习如何执行专家。
每个体系一个真实结构足以提供当前的终点与距离监督，不需要多构象标签。

令 `A(q), B(q)` 为提升到共同自由度的专家场，`G(q)` 为几何修复场，`U` 为现有
SDPI 刚体/对称水平 torsion 更新器（内部应用负时间步）。实际执行为：

```text
v1 = (1-g1) [α A(q) + (1-α) B(q)] + g1 G(q)
q_mid = U(q, v1, s dt)
v2 = (1-g2) [(1-α) A(q_mid) + α B(q_mid)] + g2 G(q_mid)
q_next = U(q_mid, v2, (1-s) dt)
```

`α,g1,g2∈(0,1)`，`s∈(.25,.75)`。这是连续顺序偏好，没有离散选择严格AB/BA，
也不平均两套候选坐标。第二子步真正查询已执行的中间状态，不复用初态专家输出。
控制器初始接近平衡，顺序及修复偏好是否形成有意义的分工，需要训练与消融确认。

试探固定使用 `.025` 时长：`q_A=U(q,A(q),.025)`、`q_B=U(q,B(q),.025)`。
额外查询 `B(q_A)` 与 `A(q_B)`，固定原状态的 FiLM 和专家噪声时间，隔离几何干预；
正式中间状态重新生成 FiLM，专家噪声时间仍固定在当前外层步，三元时间条件中的
剩余时长改为 `(1-s)dt`。这是固定时间的两阶段更新，没有宣称高阶ODE求解精度。
交叉响应使用刚体向量范数、内积与torsion统计，共30个不变量；另有三个界面的
预测目标误差及两次试探变化，共9个不变量。所有目标来自模型预测，真实结构不进入forward。
有限试探不除以步长，不解释为收敛的 Lie bracket 或生物学协同性。

每个外层步6次逻辑二元评估：原状态2次、交叉试探2次、中间状态2次，打包为3轮专家调用。
默认20个外层步执行40个正式积分子步、120次二元评估（原版本40次）；试探另有40次积分。
评估数量增加3倍，实际耗时需实测。训练和采样共用 `SDPIFlow.apply_output` 返回的实际组合状态。
输出中的平均速度仅供诊断，不能用它再次做单步积分；trace的 `velocity_semantics` 明确标注。
不支持对其做采样后置速度缩放、velocity_transform 或 freeze_poi，入口明确报错。

## 共同自由度与几何修复

归一化未知量 `v=[t_L, omega_L, t_P, omega_P, theta]`，维度 `12+K`。
平移乘受体尺度 `s_E/s_P`，角速度与torsion乘π。专家A的刚体映射为 `[t_L,omega_L]`；
令 `lever=c_P-c_L`，专家B的映射为：

```text
B.tr  = s_E/s_P * t_L - t_P + π/s_P * (omega_P × lever)
B.rot = omega_L - omega_P
```

B平移定义在ligand质心，不能直接作为POI平移。每个专家经有界各向同性压缩后，
使用最小归一化刚体范数提升 `Mᵀ(MMᵀ)⁻¹u`；torsion经tanh后进入唯一共享自由度。
所有实际更新沿用刚体变换和对称水平torsion更新器，零torsion配体同样支持。

几何网络复用三元关系主干、局部蛋白几何、专家标量/向量与16个蛋白软锚点，
预测E3–L、POI–L、E3–POI终态距离及相对权重。保留完整接收端，不按硬半径删除远距离约束。
距离为 `sqrt(||delta||²+.01)/10 Å`，正的预测终态距离为 `softplus(raw)`。
正向时钟目标变化率为 `2*tanh((d_current-d_hat)/(2*max(t,.2)))`；
反向积分在距离过大时靠近、过小时远离。

```text
G(q) = argmin_v Σ_interfaces ||J_i v-b_i||²_Wi + .01 ||v||²
```

修复器只包含三元几何约束。旧的专家精度头、torsion来源权重头及生产路径的键上下文编码器已移除，
不再把二元建议同时加入最小二乘。专家通过上述两阶段控制器执行，避免重复控制。
几何Jacobian仍包含共享torsion；自由三元torsion输出头未恢复。
材料锚点在一次求解中随所属刚体运动，Jacobian不包含特征变化导致的锚点重新分配；
外层训练仍对锚点权重反传。torsion Jacobian沿用对称水平更新器的质心/spin去除与惯性正则。

界面按行数归一化，权重为 `.001+sigmoid(raw)`，再统一归一化到总和1，避免整体降权变成减速开关。
它们是相对残差权重，不是校准置信度。正规方程用float64求解再还原dtype，
各向同性正阻尼保持数值稳定。诊断记录修复残差、界面权重及四个控制量。

## 单结构监督与数据隔离

每个体系只提供一个观测结构。模型每个实际子步的预测终态距离都受到该结构的监督，
而非为假想多模态构造标签。以当前软锚点的材料权重计算真实结构中的对应锚点，
标签构造停止梯度；真实坐标不进入模型forward。监督为归一化距离的`log1p`上的
smooth-L1，先对每个界面取均值，再对三个界面等权平均；所有界面共同选取同一个
等价配体原子置换。配体使用现有运动学可达目标，蛋白保持输入刚体形状。
距离监督先在每个外层步的两个几何预测间平均，再按外层步数平均，乘固定0.1加入原终点损失；不乘模型置信权重。
恢复分支沿用原有分支权重。随机初态与扰动是恢复任务，不是独立结构观测。

索引构建保留DeepTernary两个来源的聚类关系，再传递合并同PDB条目后划分。
`results/model45_partition`保留全部7087个可用条目：训练6705，验证382；
PDB重叠0，原始两个来源共1985个聚类关系审计通过。没有重新推断额外序列同源簇。
训练入口强制检查两个索引与原始来源聚类，checkpoint保存索引及来源聚类SHA256，
恢复时拒绝划分变化。旧默认索引有83个PDB交集，不得用于新架构质量对照。
划分改变后旧三元权重不能作为干净验证基线；对照双方需在相同新训练集重新训练。

## 运行、版本与验收

profile 仍为 `ternary_constraints_v1`，checkpoint 必须匹配 v39 和以下契约：

```text
constraint_solver_contract = terminal_geometry_repair_v3
operator_flow_contract = finite_response_two_stage_v1
operator_probe_step = 0.025
target_geometry_loss_weight = 0.1
```

v38终态距离联合求解模型在提交 `ef92b1e`；v37自由变化率模型在 `13799b6`。
架构不兼容的checkpoint不能通过resume混用。`expert_coordinate_grad` 默认保持False：
冻结专家对坐标的直接反传关闭，FiLM与可微积分/修复/控制器仍可训练；可显式切换此近似。
A1/B/C来源协议、辅助恢复及现有结构监督保持原有定义。

```bash
python tests/test_model_build.py
python tests/test_training_contract.py
python tests/test_consensus.py
python scripts/run_cpu_tests.py --timeout 180
CUDA_VISIBLE_DEVICES=0 python scripts/validate_consensus_model.py \
  --train-index results/model45_partition/ternary_train.json \
  --cache-dir /absolute/path/to/compatible_cache --out results/model45_operator_smoke
CUDA_VISIBLE_DEVICES=0 python scripts/train_entry.py \
  --train_data results/model45_partition/ternary_train.json \
  --val_data results/model45_partition/ternary_val.json \
  --output_dir results/model45_train \
  --training-arm A1 --cache-only --cache_dir /absolute/path/to/compatible_cache \
  --batch_size 1 --lr 1e-4 --device cuda
```

GPU号为示例，运行前检查占用。本地启动脚本默认读取 `results/model45_partition`，
可用 `PARTITION_DIR` 指向其他已审计划分。`smoke_only.pt`不能作为质量选优权重。

2026-10-05验收：完整CPU套件439项通过，随后定向复跑新增/修改项；新增5项覆盖
实际中间状态重读、控制器梯度、等变性与零torsion、有限差分梯度以及多轨迹打包一致性。
真实冻结专家在训练案例 `4X28_B_C_FDA` 完成2次优化及20步采样；所有可训练参数梯度有限，
专家未改变，训练/采样终点一致，EMA checkpoint恢复精确一致。最大蛋白内部距离误差
`8.01e-5 Å`，峰值CUDA分配约2.01 GB。总训练损失 `6.25180 → 6.24368`；
这是执行检查，不代表泛化收益。完整记录为 `results/model45_operator_smoke/summary.json` 与 `trace.json`。

## 当前局限

- 预测终态距离会随状态更新，可能互不兼容；线性修复不保证全局可装配、逐步降误差或无碰撞。
- 软锚点可能聚集、漂移。对称/共线布局仍会缺失运动信息，阻尼不解决不可辨识性。
- 单个观测结构不能识别真实多态占比、自由能或生物学协同性。
- 有限试探尺度固定，专家裁剪或图邻域变化可造成响应突变；没有声称无穷小导数稳定。
- 连续顺序偏好可能塌缩为接近平衡或偏重修复，响应证据是否提高预测质量必须实测。
- 新模型增加专家调用预算。质量对照需同时报告等训练预算与等推理预算，以及DockQ、碰撞、耗时和显存。

后续正式对照需使用同一干净划分、来源、seed与候选协议，重新训练双方；
当前未完成质量训练，不把单例smoke下降写成模型性能改善或已证实的研究突破。
