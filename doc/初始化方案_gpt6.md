# 双 DiffDock＋联合 TriFuse 三元复合物初始化方案

> 设计规格草案。本文定义共享完整配体、双二元相对状态和统一三元坐标更新；其中实验结论仍需通过本文门禁验证。

## 1. 目标与原则

研究对象为两颗刚性蛋白质 A、B 与一颗化学结构已知、允许扭转的小分子 L。目标不是模拟真实结合动力学，而是在一个合法三元状态中保留两侧 DiffDock 熟悉的蛋白质—配体相对位姿先验。

必须遵守以下不变量：

1. 全局始终只有一份完整、连通的 L。
2. 两个 DiffDock expert 都读取同一完整 L，不将 PROTAC 切成两个片段作为冻结 expert 的输入。
3. 两侧可独立提出刚体更新，但完整 L 的扭转每步只能执行一次。
4. 两个 expert 从同一旧状态计算；联合 TriFuse 仲裁后，由一个坐标更新器原子地更新三元状态。
5. 固定 A（E3）作为全局 gauge；B（POI）相对 A 运动。

这里的“保留二元先验”只表示各侧装配前后的局部相对几何不变，不自动保证三元碰撞、蛋白间界面或最终精度。

## 2. PROTAC 与 MGD 的输入区别

| 项目 | PROTAC：结合头已知 | MGD：结合区域未知或不可划分 |
| --- | --- | --- |
| 物理配体 | 完整 PROTAC `a—linker—b` | 完整分子胶 L |
| expert A 输入 | 完整 L | 完整 L |
| expert B 输入 | 同一完整 L | 同一完整 L |
| A 侧对齐参照 | `lig1_mask` 对应的结合头 a | 完整 L |
| B 侧对齐参照 | `lig2_mask` 对应的结合头 b | 完整 L |
| 额外条件 | 两端原子身份、二元 pose、口袋和局部 frame | 可用口袋/界面候选；默认无端点划分 |

`lig1_mask/lig2_mask` 用于初始化对齐、界面读出和 TriFuse 条件，不改变冻结 DiffDock 的配体图。若未来研究片段 expert，必须单独处理封端、原子特征、片段 torsion 到完整 L 的映射并重新验证；它不属于本方案第一版。

## 3. 模块职责

| 模块 | 职责 |
| --- | --- |
| DiffDock A | 对完整 L 相对 A 的局部刚体与扭转状态提出 score |
| DiffDock B | 对同一完整 L 相对 B 的局部状态提出 score |
| 联合 TriFuse | 同时读取两侧 expert 遥测、三元几何和可选端点/口袋先验；校准两侧刚体建议并产生唯一 torsion 决策 |
| 统一坐标更新器 | 将两个局部 proposal 合成为一份 L 和两颗蛋白的全局坐标 |

当前仓库已有一个联合 `TriFuse`，内部包含双分支 FiLM、layerwise bridge、score gain 和共享 torsion coordinator。初始化第一版不拆成两个独立 fuse，也不改变 checkpoint 架构。

## 4. 共享配体的联合初始化

### 4.1 生成唯一内部构象

1. 从完整 L 的规范化化学图生成一个三维构象。
2. 按选定 DiffDock 配置，只采样一次完整 L 的可旋转键。
3. 保留原子顺序、可旋转键顺序、旋转侧 mask 和立体化学。
4. 复制该内部构象作为两侧局部初始化的共同输入；局部副本只允许有不同的整体刚体位姿。

MGD 两侧使用完整 L 作为对齐参照。PROTAC 可用两端 mask 或二元 warhead pose 确定局部 frame，但 expert 仍读取完整 L。

### 4.2 分别采样两个二元位姿

对每侧 `i ∈ {A,B}`：

1. 按冻结 DiffDock 要求预处理并居中蛋白质 `P_i`。
2. 从共享内部构象复制完整 `L_i`。
3. 对 `L_i` 采样 Haar SO(3) 整体旋转，并使用该侧受体尺寸自适应的高斯平移先验。
4. 保存实际采样的旋转、平移、受体中心和随机配置，不通过事后 Kabsch 猜回已知变换。

两侧分别从 A、B 自身的受体坐标计算平移标准差，因此蛋白尺寸不同时允许具有不同尺度。两侧刚体随机量可以独立；内部构象和 torsion 状态必须相同。新方案不提供固定绝对 Å 数的广域平移模式。

### 4.3 装配为一个三元状态

选择一个中央完整配体 L。分别求出把局部 `L_A`、`L_B` 放到中央 L 对应参照的逆刚体变换，并将同一变换施加到对应蛋白质。删除两个局部配体副本，只保留中央 L。

最终得到：

| 组分 | 初始状态 |
| --- | --- |
| A / E3 | 固定为全局参考 |
| L | 唯一、完整、连通，承载共享 torsion |
| B / POI | 由 B–L 的二元初始化逆装配确定 |

装配必须保持每侧所有蛋白质—配体成对距离，而不只是 COM 距离。

## 5. 刚体变换的精确定义

以下采用列向量。设局部对齐参照为 C，质心为 `c_L`，受体或口袋初始化中心为 `c_R`。局部 DiffDock 初始化为：

$$
\widetilde C=Q(C-c_L)+c_R+\delta,
$$

其中 Q 为随机旋转，δ 为平移噪声。映射回中央参照的逆变换是：

$$
T(x)=Q^\top[x-(c_R+\delta)]+c_L.
$$

因此 `T(\widetilde C)=C`。只有当 `c_R=0` 时才能省略受体中心。实现应保存并反演实际采样变换；使用 Kabsch 兜底时，映射原子必须至少包含三个非共线点，并显式处理对称原子、氢过滤、手性和退化情况。

## 6. 双 expert 的逐步联合更新

DiffDock 的 `tr/rot` 输出是切空间 score，不是可以直接取负或平均的最终 pose。每一步必须先使用各自的 σ、dt、SDE/ODE 漂移和随机项，从同一旧状态生成两个有限刚体 proposal：`D_A` 与 `D_B`。

若选择统一 ligand 刚体更新 C，则全局组合为：

$$
L'=C(L),\qquad P_i'=C\circ D_i^{-1}(P_i).
$$

固定 A 作为 gauge 时取 `C=D_A`：

$$
A'=A,\qquad L'=D_A(L),\qquad B'=D_A\circ D_B^{-1}(B).
$$

随后执行一次 TriFuse 协调后的完整 L torsion 更新。torsion 会同时改变两侧局部几何，因此“精确保留两个 proposal”只对 torsion 前的刚体组合成立，后续步骤负责继续纠偏。

两个 expert 的 torsion tensor 只有在 canonical bond ID、bond direction、旋转侧 mask 和顺序完全一致时才能融合；若键方向相反，必须同步修正 score 符号。

## 7. 唯一允许的广域平移先验

`dual_native` 必须采用 DiffDock 默认推理 YAML 的受体尺寸自适应高斯，不提供固定标准差的兼容分支。对每侧居中后的受体坐标 `p_j`，定义：

$$
s_{rec}=\sqrt{\frac{1}{N}\sum_j\lVert p_j\rVert^2},\qquad
\sigma_{init}=s_{rec}\frac{\rho}{1.73},
$$

其中 `ρ = initial_noise_std_proportion = 1.4601642460337794`，平移按

$$
\delta\sim\mathcal N(0,\sigma_{init}^2 I_3)
$$

采样。A–L 和 B–L 必须分别使用对应受体的 `s_rec`。若使用 pocket center，它只改变高斯均值中心，不改变受体尺寸的计算定义，除非另行定义并版本化新的方法。

`tr_sigma_max` 仍属于冻结 score model 的扩散 schedule 和反向采样公式，但不得再被复用为初始化高斯的标准差。实现时应删除旧的固定绝对尺度初始化路径，而不是保留为隐藏选项、fallback 或对照模式。

实验必须记录：

- DiffDock commit 和 score checkpoint；
- `no_torsion`、`pocket_knowledge`、`choose_residue`；
- `initial_noise_std_proportion`；
- receptor 尺寸、计算得到的 `sigma_init` 以及 receptor/pocket center 的定义；
- NumPy、SciPy、PyTorch 与 Python RNG seed。

`initial_noise_std_proportion` 必须为上述固定正值；负值或任何请求固定绝对尺度的配置必须直接报错。文档、CLI、缓存签名和结果 JSON 均使用 `dual_native` 表示这一唯一语义，禁止同名模式随配置切换分布。

## 8. 训练—推理一致性

建议将扩散状态显式定义为：

$$
X_A=\operatorname{pose}(L\mid A),\quad
X_B=\operatorname{pose}(L\mid B),\quad
\theta=\text{shared torsion}(L).
$$

训练从真实三元结构出发，对 `X_A`、`X_B` 施加独立刚体噪声，但共享时间条件和唯一 torsion 噪声，再通过与推理相同的组合函数重建三元坐标。当前 `add_noise_interface_pair()` 已接近这一表示；主要缺口是推理 initializer 仍分别摆放 ligand 与 POI。

训练 forward marginal、DiffDock 的经验终端初始化和带二元候选的局部 proposal 是三种不同对象：

- 广域 native prior：从 `t_start=1` 开始；
- 二元 pose 周围的局部 proposal：用相同 forward kernel 扰动，并从匹配的 `t_anchor<1` 开始；
- bounded 5 Å 初始化：只能作为明确的 fallback/消融，不能假装等于训练的 `t=1`。

训练和推理必须共用 `compose/decompose`、frame 和 torsion 映射实现。

## 9. 碰撞、口袋和 MGD 边界

两侧独立二元初始化不保证合理的蛋白质—蛋白质界面，尤其 MGD 可能依赖由另一颗蛋白和配体共同形成的协同界面。两个 DiffDock 仅提供局部先验；TriFuse 必须观察 A–B、A–L、B–L 三类关系。

初始蛋白碰撞可以作为未筛选 baseline 被记录，但不能称为可用生产先验。碰撞拒绝、口袋约束或界面筛选都会改变联合分布，应作为显式模式，在训练和评估中分别报告，禁止静默启用。

## 10. 最小落地顺序

1. 新增 full-ligand `dual_native` 联合 initializer，只采样一次 torsion；两侧分别计算受体尺寸自适应平移尺度。
2. 接入 sampler、CLI 和 trace，并从 CLI 与实现中删除固定绝对尺度的旧 `native` 初始化路径；暂不修改模型、训练目标或旧 checkpoint。
3. `coupled5a`、`anchor_local` 若仍用于诊断，必须保持不同名称；它们不能被 `dual_native` 静默调用或作为失败 fallback。
4. 运行两次独立 DiffDock 与“双 expert＋固定 torsion 平均”的配对实验。
5. 再加入版本化 `binary_prior` sidecar：端点 mask、二元 pose、口袋、frame、置信度和来源 SHA256。
6. 先让先验只进入 TriFuse；关闭先验时必须严格复现旧架构输出。
7. 只有片段 expert 被单独训练和验证后，才研究以 warhead 片段替代完整配体输入。

## 11. 验收门禁

### CPU 几何与回归

1. 随机 SE(3) 下验证逆变换 roundtrip、`RᵀR=I`、`det(R)>0`。
2. 装配前后两侧 receptor–ligand 全部 `cdist` 保持，误差不超过 `1e-5`。
3. 验证两侧共享同一内部构象、canonical torsion 映射和方向翻转。
4. 对称、少于三个非共线锚点或无效 atom map 必须硬失败。
5. 分别验证 A、B 的 `sigma_init` 与受体 RMS radius 公式一致；负比例和固定尺度参数必须硬失败。
6. 关闭新先验时的模型输出不得变化。

### GPU 模型与采样

1. 单侧官方 DiffDock 与联合 initializer 做固定配置、固定 seed 的分布/parity 检查。
2. zero-FiLM 下 sub1/sub2 score 均 finite，并满足既定默认容差。
3. 1 case × 1 walker × 20 steps 通过后，再运行 32 walkers。
4. MGD 与 PROTAC 分开报告两侧 interface 指标、full ternary DockQ、Fnat、clash、strain 和 valid rate。
5. 比较：单独 DiffDock A/B、双 expert＋torsion 平均、双 expert＋learned TriFuse；三组使用配对 seed 和相同初始化配置。

## 12. 来源与证据边界

- [DiffDock 官方仓库](https://github.com/gcorso/DiffDock)：模型与版本说明。
- [DiffDock 初始化及采样实现](https://github.com/gcorso/DiffDock/blob/main/utils/sampling.py)：随机 torsion、旋转、平移和反向采样。
- [DiffDock 默认推理配置](https://github.com/gcorso/DiffDock/blob/main/default_inference_args.yaml)：尺寸自适应初始化等运行参数。
- [TriGlue](https://arxiv.org/abs/2607.22143)：MGD 的界面估计和三元联合建模动机；不是本初始化方案的实验验证。

正式实验必须固定代码 commit、权重、配置、数据版本、原子映射和随机种子。本文的共享参照装配、联合更新和 TriFuse 分工仍需通过上述门禁后才能升级为默认方法。
