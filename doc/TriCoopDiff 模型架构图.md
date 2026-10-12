# TriCoopDiff 模型与坐标契约

更新：2026-10-07。本文记录当前无语义 token 协商的实现；早期架构说明仅作历史参考。

## 当前架构

```text
同一三元状态（固定 E3，一份完整 ligand）
  ├─ 冻结 DiffDock：E3–ligand ─┐
  └─ 冻结 DiffDock：POI–ligand ├─ 共同 E3 frame 的逐节点标量、向量
三元几何、共享时间与实际来源 ───┘
                 ↓
       组分内部结构编码 ──→ 每层读取固定节点结构表示
                 ↓
       三层分来源多头等变注意力（当前姿态、专家与时钟）
                 ↓
       LegalPoseDecoder
         ├─ ligand 整体平移、有限轴角旋转
         ├─ POI 整体平移、有限轴角旋转
         └─ 逐键 torsion（两侧专家对应键建议＋两侧片段表示）
                 ↓
       按剩余时间转换为原速度单位
                 ↓
       SDPIFlow：刚体与合法 torsion 执行
```

当前 profile 为 `ternary_endpoint_v14`，类为 `TernaryAssembly` / `TernaryAssemblyModel`，训练器为 `AssemblyTrainer`。E3 固定；蛋白内部不变；配体仅刚体与可旋转键变化。全部节点参与信息交互，不代表节点可自由运动。

删除 v7–v9 的自由 Cartesian 目标、逐对约束头、Schur 求解、目标刚体配准、torsion 雅可比逆解及相应辅助监督。当前输出是相对当前状态的合法位姿增量，不是初始化不变的绝对答案，也没有跨起点或全局最优保证。

v12 在进入动态交互前建立独立的逐节点结构表示：蛋白使用自身特征和内部距离；配体使用化学身份、化学边属性与键长，仅允许自环/化学边注意力。不读取跨组分坐标、专家输出、时钟或来源。该表示在同一次forward的三层交互中不被动态状态覆盖，每层通过节点对应的读取层与动态表示合并；没有全局摘要token或新增缓存。相对刚体初始化不改变此结构表示；合法torsion保留配体键长，因此不改写这一路化学身份。蛋白内部实际形变仍会改变结构表示，动态几何通路仍感知完整当前构象。这里是表示分工，不是宣称动态表示或最终答案初始化不变。

v14 删除固定四跳化学传播，替换为逐原子的全连通化学读取：注意力读取化学图最短路径距离、直接化学边属性与键长，关系信息同时进入权重和消息内容。只允许同一化学连通分量内交换信息，不读取非键合Cartesian距离，因此保留刚体/合法torsion不变性。12原子链的端到端梯度检查证明读取不受四跳限制，但最短路径编码仍不等于完备图同构表达。当前最短路径计算为Floyd–Warshall，时间O(L³)、关系存储O(L²)，没有增加缓存；真实运行成本未评估。

逐键动作几何读取新增：相对键轴的有符号轴向位置、径向力臂、受影响原子掩码，以及节点向量沿轴和沿原始旋转切向运动的投影。切向运动为axis×lever乘旋转掩码；该信息进入两片段及三组分注意力的权重和消息内容。它不读取目标，不做雅可比逆解；原始rotor切向也不冒充执行器完整水平规范的雅可比。有限差分检查原始rotor导数的符号和单位，全局刚体变换不改变这些标量特征。

此前替换的联合读取层继续保留：每条键的查询由两个端点、向量沿键轴的投影和两个专家对应键建议构造，按原子身份与距键中点的距离，分别在旋转片段与补片段内做选择性注意力，而非无条件均值。两个刚体查询与全部键查询再分别读取 E3、POI、配体的节点证据（分组归一化），最后在物理自由度之间自注意力交互。查询一一对应两个移动组分与真实旋转键，没有额外可学习全局语义 token；初始刚体查询仍用组分均值，但之后可重新读取全部节点，并非仅凭均值输出动作。

刚体头读取每个本组分节点与其联合决策表示，生成向量系数和节点选择权重，再汇总等变向量；平移单位10 Å，旋转为小于π的有界轴角。逐键头读取交互后的键决策，输出π·tanh范围内的角增量。不同键及刚体决策可相互影响；v14联合注意力显式读取物理自由度间的距离、受影响原子Jaccard重叠及方向性覆盖率，避免仅依赖隐状态猜测动作耦合。这只是表达通道，不保证学会最优协同。当前canonical键方向仍来自输入图。

按真实正剩余时间转换为反向速度，不再截断到0.05；t=0仅作有限值诊断读取，沿用现有 20 步时钟与执行器。刚体在 dt=t 时执行全部预测增量；torsion 使用既有连续水平规范与顺序键旋转，不宣称独立有限旋转彼此可交换。零输出保持输入结构。

监督仅保留实际执行后的 endpoint 几何、界面及已配置的排斥项，并沿用近目标辅助入口。模型不读取 native。checkpoint 记录 `topological_joint_legal_pose_v5` 输出合同；不允许把旧自由坐标目标权重当作本模型续跑。

冻结专家、全节点注意力和完整 20 步坐标反传仍保留；尚未证明跨起点 DockQ 或长训稳定性改善。架构/梯度、恒等执行、蛋白刚体保持、配体键长保持、两侧逐键专家梯度，以及键间/键到刚体影响和键重排不变性使用轻量检查；v14不启动训练实验，不沿用v10 smoke作为当前版本效果证据。

v10 真实 smoke：GPU0，seed17，node_dim128，缓存训练样本 `1AFQ_B_C_0FG`，两个真实冻结专家，20 步完整坐标反传，lr1e-4，一次更新。total loss0.3457996，裁剪前梯度范数12.7887，峰值分配显存0.8782 GB，train_step79.49秒；没有测 DockQ，也没有保存训练权重。旧 v8 smoke 的梯度约1468、显存2.31 GB、97秒仅为历史记录；因初始化未配对且loss已变化，不据此定量归因或声称模型质量提高。

本轮审查记录：第一轮替换固定四跳限制；第二轮加入明确动作重叠关系；第三轮修正小于0.05时钟的执行合同；第四轮复查结构→动态→自由度读取→执行器→训练/采样合同，未再发现本轮可明确定位的新增P0–P2矛盾。该停止结论不是架构最优、跨起点收敛或DockQ改善的证明。仍保留固定蛋白内部结构、单一确定性策略、独立冻结专家、有限宽度表示等设计限制。

## 模型版本

| 家族 | architecture | checkpoint_version | 说明 |
| --- | --- | ---: | --- |
| 历史模型 | `trifuse_v13` | 17 | 仅用于识别旧 checkpoint |
| 历史 token/残差与节点场模型 | `residual_consensus_v2`, `ternary_geometry_independent_v1`, `ternary_nodefield_independent_v2` | 18–30 | 不兼容当前模型，不应作为当前训练/续跑权重 |
| 标准 evidence 模型 | `residual_evidence_v3` | 31 | 直接使用显式几何 evidence，不含语义 token 协商 |
| 节点场三元模型 | `ternary_nodefield_independent_v3` | 31 | 节点标量条件的局部等变场；专家逐节点向量未桥接 |
| 已移除的直接节点场 | `ternary_nodefield_independent_v4` | 历史权重 | 当前生产构造器拒绝 |
| 已移除的直接动作控制器 | `ternary_equivariant_attention_v6` | 40 | 已被约束解码替代 |
| 已替代的局部约束解码 | `ternary_constraint_assembly_v7` | 41 | 联合切空间运动求解 |
| 已移除的自由坐标目标 | `ternary_joint_target_assembly_v9` | 43 | 自由坐标预测与合法执行空间不一致 |
| 已替换的独立读取头 | `ternary_legal_pose_v10` | 44 | 合法增量，但键片段均值读取与最终决策交互不足 |
| 已替换的混合输入模型 | `ternary_joint_legal_pose_v11` | 45 | 联合读取，但缺少独立的结构信息通路 |
| 已替换的单跳化学记忆 | `ternary_intrinsic_joint_pose_v12` | 46 | 结构与动态装配分离，但配体化学传播浅 |
| 已替换的固定跳数结构 | `ternary_chemical_action_pose_v13` | 47 | 四跳化学传播、动作几何 |
| 当前拓扑联合位姿模型 | `ternary_topological_joint_pose_v14` | 48 | 全连通化学读取、显式动作耦合、真实剩余时间 |
| factorial 实验 | `residual_consensus_factorial_v1` | 1 | 冻结专家实验加载器 |

当前 architecture/checkpoint 标识已升级；旧模型权重不能静默加载到新的输出语义。每次比较历史指标时，需注明权重架构与相应评估协议，不能跨架构把指标当作同一模型结果。

## 参考系与一步更新

E3 固定。概念状态为两个相对 SE(3) 位姿 `T_E3L, T_POL` 和共享 torsion `chi`：

```text
T_L   = T_E3L
T_POI = T_L · inverse(T_POL)
L_new   = D_A(L_old)
POI_new = D_A · inverse(D_B)(POI_old)
```

`D_A/D_B` 都必须从同一个旧 ligand 生成。使用完整刚体乘法与逆，不能独立取负平移或把旋转向量相减当作 SE(3) 组合。随后 torsion 只应用一次，并更新 POI body frame、crop 和 cross-edge。

## 初始化与联合训练

SDPI 从保持 canonical atom 顺序的完整 RDKit 图生成 ETKDGv3 构象，两侧分别采样 Haar SO(3) 和高斯平移。BiTSeed 用两个冻结二元专家生成 endpoint。

B/C 以 `SDPI:BiTSeed = 1:1` 交替训练。训练用 BiTSeed 是 lazy `C=1, M=1`：一个 conformer、每侧专家各一条轨迹、无排序；任一失败时回退 SDPI。C 模型使用 source conditioning，fallback 必须标为 SDPI。推理的 `C=4, M=8` 只提高候选覆盖，不改变训练样本数。

## SDPI flow 与不变契约

当前生成过程为 `sdpi_flow_v1`：clean/native 为 t=0，来源 endpoint 为 t=1，两侧共享时间。采样使用反向 Lie–Euler。较短 `t_start` 只用于 fixed-candidate 诊断，不能把 BiTSeed 任意解释成 SDPI 时间边缘。

检查完整 atom/torsion 映射、SE(3) round trip、等变性、同源双 proposal、单次 torsion、动态 frame、有限输出与缓存身份。不能用零值替换 NaN；activation clamp 必须恢复 checkpoint 配置。完整训练契约见 [SDPI–BiTSeed 联合训练交接](SDPI-BiTSeed联合训练交接.md)。
