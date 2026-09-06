# SDPI：共享配体双位姿初始化

SDPI 全称为 **Shared-Ligand Dual-Pose Initialization**。两侧共享一份完整配体内部构象，分别采样配体相对 E3、POI 的二元刚体位姿，再精确装配为一个三元初态。代码和 CLI 使用 `sdpi`，结果版本为 `sdpi_v1`；本次命名不改变采样分布或模型架构。

历史验证产物仍保留当时的 `dual_native` 文件名与记录，以下链接用于追溯原始实验。新运行只接受 `sdpi`；旧名称不是兼容别名，旧结果也不会被当成 `sdpi_v1` 续跑结果。

> 当前 `sdpi_v1` 初始化规范及验收边界。初始化几何已验证；完整模型精度、MGD benchmark 和二元强先验仍需独立实验。

### 当前落地范围（2026-09-06）

`tricoopdiff/initialization.py` 实现唯一 `sdpi_v1`。从保持 atom index 的完整 RDKit 图生成 ETKDGv3 构象，固定 `useRandomCoords=True` 以支持长链 PROTAC；每个 walker 只采样一次完整 torsion。A/B 各用自身 Cα RMS radius，记录实际旋转、平移和中心，通过 `rigid.py` 的解析逆变换装配。训练的界面 forward noise 也使用同一组合函数。两个 expert 继续读取同一完整图，TriFuse 架构和训练目标没有变更。

按本次替换要求，旧 `native`、球壳、固定 Å 高斯、`coupled5a` 和 `anchor_local` 的实现及 CLI 入口全部删除；`t_anchor != 1`、非指定比例及旧固定尺度参数会报错。本文后续描述的局部 proposal 和 sidecar 是未来方案，不代表当前支持。当前不做外部 atom map、对称性匹配或 Kabsch 初始化，因此不存在退化 anchor fallback。

特征缓存版本升至 5，保存保持原子顺序的 RDKit 分子；旧缓存需重新特征化。结果 JSON 的 `initialization` 保存每个 walker 的尺度、变换、torsion 顺序、库版本、设备、种子和专家来源。模型 checkpoint 格式不变；评估续跑签名必须匹配 `init_mode=sdpi` 与 `initialization_version=sdpi_v1`。

Indexed 评估续跑还比对输入/参考结构、TriFuse 权重、专家权重及配置的内容指纹，防止同路径文件被替换后误用旧结果。缺少指纹、不完整 walker 数或非有限 full DockQ 的报告不会跳过重跑。采样拒绝非正整数步数、空 ensemble 和非布尔 torsion mask。

可复现实例命令（仓库根目录运行，模型资产必须已缓存）：

```bash
python tests/test_initialization.py
python scripts/validate_initialization.py --case-dirs DeepTernary/output/protac22/5T35_D_A_759 DeepTernary/output/protac22/6BN7_B_C_RN3 DeepTernary/output/protac22/7JTO_L_B_VKA --official-sampling ../DiffDock/utils/sampling.py --out results/sdpi_geometry_examples.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/evaluate_entry.py --case 5T35_D_A_759 --allow-zero-init --n-walkers 1 --steps 20 --cache-dir results/featurize_cache_sdpi_v5 --out results/sdpi_5T35_etkdg_random_1x20.json
```

`--official-sampling` 仅用于对照本地官方源码的刚体初始化核，不构成完整 DiffDock/训练后 TriFuse 精度对照。`--allow-zero-init` 明确表示未训练 TriFuse；初始化几何门禁、有限采样与最终 docking 精度必须分别报告。

### 本次验证结果

代码实现提交为 `fc385c5`、`49ea618`，验证脚本提交为 `329e1ae`。本地 RTX 4060 GPU，PyTorch 2.6.0+cu124，offline 模式，冻结专家加 zero-init TriFuse，activation clamp 为 100；没有使用训练后 TriFuse checkpoint。

| 验证 | 结果 | 产物 |
|---|---|---|
| 3 个真实 PROTAC × seeds 0/7/42，CPU float64 几何 | 全通过；最大距离/参考系误差 2.6×10⁻¹² Å | [geometry](../results/dual_native_geometry_examples.json) |
| 官方 DiffDock 刚体核，16 seeds，CPU float32 | 全通过；最大坐标差异 3.9×10⁻⁶ Å | 同上，`official_rigid_parity` |
| 5T35_D_A_759，1 walker × 20 steps | 1/1 有效；full DockQ 0.0166；trace 20 steps 全 finite | [1×20](../results/dual_native_5T35_etkdg_random_1x20.json) |
| 6BN7_B_C_RN3，1 walker × 20 steps | 1/1 有效；full DockQ 0.00846 | [6BN7](../results/dual_native_6BN7_1x20.json) |
| 7JTO_L_B_VKA，1 walker × 20 steps | 1/1 有效；full DockQ 不可用 | [7JTO](../results/dual_native_7JTO_1x20.json) |
| 5T35_D_A_759，32 walkers × 20 steps，seeds 0–31 | 32/32 有效，无 NaN；mean DockQ 0.0607，oracle-best 0.2293 | [32×20](../results/dual_native_5T35_etkdg_random_32x20.json) |

初始化、界面坐标、训练契约、数据处理和评估辅助逻辑共 26 项函数检查通过；`tests/test_model_build.py` 亦全部通过。当前环境未安装 pytest，带 `tmp_path` 的函数通过 Python 临时目录直接调用，未跳过其测试体。

7JTO 的完整 DockQ 失败源于全原子刚体拟合中 Cα 数量不匹配（148 对 154），见对应日志；不能把其近似界面指标冒充完整 DockQ。未运行 MGD benchmark、训练后 TriFuse 精度对照或完整 Phase 1 training smoke，也没有启用碰撞筛选。上述结果只支持初始化几何与有限采样结论。

合并前复核：全库 65 项函数测试通过（包含新增续跑/非法输入回归）；SDPI 命名下再次运行 5T35 的 1 walker × 20 steps，1/1 有效，full DockQ 0.0168，见 [复核报告](../results/sdpi_review_5T35_1x20.json)。此处与历史测试数量分开记录，不把小幅 GPU 数值差异解释为精度提升。

## 1. 目标与原则

研究对象为两颗刚性蛋白质 A、B 与一颗化学结构已知、允许扭转的小分子 L。目标不是模拟真实结合动力学，而是在一个合法三元状态中保留两侧 DiffDock 熟悉的蛋白质—配体相对位姿先验。

必须遵守以下不变量：

1. 全局始终只有一份完整、连通的 L。
2. 两个 DiffDock expert 都读取同一完整 L，不将 PROTAC 切成两个片段作为冻结 expert 的输入。
3. 两侧可独立提出刚体更新，但完整 L 的扭转每步只能执行一次。
4. 两个 expert 从同一旧状态计算；联合 TriFuse 仲裁后，由一个坐标更新器原子地更新三元状态。
5. 固定 A（E3）作为全局 gauge；B（POI）相对 A 运动。

这里的“保留二元先验”只表示各侧装配前后的局部相对几何不变，不自动保证三元碰撞、蛋白间界面或最终精度。

## 2. 当前输入与尚未接入的端点条件

当前 MGD/PROTAC 都用完整 L 的原子身份和完整构象作为共同参照，不读取 `lig1_mask/lig2_mask` 或二元 pose。下表中的端点/口袋条件属于未来 `binary_prior` 扩展，不是当前初始化选项。

| 项目 | PROTAC：结合头已知 | MGD：结合区域未知或不可划分 |
| --- | --- | --- |
| 物理配体 | 完整 PROTAC `a—linker—b` | 完整分子胶 L |
| expert A 输入 | 完整 L | 完整 L |
| expert B 输入 | 同一完整 L | 同一完整 L |
| A 侧对齐参照 | `lig1_mask` 对应的结合头 a | 完整 L |
| B 侧对齐参照 | `lig2_mask` 对应的结合头 b | 完整 L |
| 额外条件 | 两端原子身份、二元 pose、口袋和局部 frame | 可用口袋/界面候选；默认无端点划分 |

未来 `lig1_mask/lig2_mask` 可用于初始化对齐、界面读出和 TriFuse 条件，但不能改变冻结 DiffDock 的完整配体图。片段 expert 需要单独处理封端、原子特征和 torsion 映射，当前不支持。

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

当前两类任务均使用完整 L；端点 mask 或二元 warhead pose 确定局部 frame 的方法尚未接入。

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

DiffDock 的原始 `tr/rot` 输出不是最终 pose。生产流程按 §8 转为速度参数化，使用 dt 从同一旧状态生成两个有限刚体 proposal：`D_A` 与 `D_B`，不添加随机 SDE 项。

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

`sdpi` 必须采用 DiffDock 默认推理 YAML 的受体尺寸自适应高斯，不提供固定标准差的兼容分支。对每侧居中后的受体坐标 `p_j`，定义：

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

`initial_noise_std_proportion` 必须为上述固定正值；负值或任何请求固定绝对尺度的配置必须直接报错。文档、CLI、缓存签名和结果 JSON 均使用 `sdpi` 表示这一唯一语义，禁止同名模式随配置切换分布。

## 8. 训练—推理契约：SDPI Flow v1

生产训练与采样改为条件流匹配，`generative_process=sdpi_flow_v1`。
`t=0` 是原始晶体，`t=1` 是 SDPI 的可动自由度先验。**不投影晶体真值，
不新增内部几何自由度**。ETKDG 与晶体的刚性片段内部几何差异仍是明确的
近似，不能宣称全坐标终端分布完全相同。

### 8.1 路径与端点

令原始配体中心为 c₀，两侧受体中心为 cᵢ、SDPI 尺度为 sᵢ。
独立采样 zᵢ ~ N(0,I)、Rᵢ,₁ ~ Haar(SO(3))，并对每个共享 torsion
采样 aⱼ ~ Uniform(-π,π)。置：

- mᵢ(t) = (1-t)c₀ + t(cᵢ+sᵢzᵢ)；
- Rᵢ(t) = Exp(t Log(Rᵢ,₁))，使用主值 Log（π 切割处为零测集）；
- C(t) = 对原始配体应用 t·a 的完整 torsion 旋转，再将中心移回 c₀。

Lᵢ(t) = (C(t)-c₀) Rᵢ(t)ᵀ + mᵢ(t)。
两侧通过 `assemble_interface_pair` 解析装配；E3 固定，完整配体只有一份。
t=0 精确返回晶体；t=1 的中心为受体自适应 Gaussian、姿态为 Haar、
可旋转角为 uniform。Haar/圆周 uniform 的平移不变性消除了原生位姿和
初始 torsion 对这些终端变量分布的影响。不可旋转的内部几何不在此结论内。

### 8.2 速度目标与对齐修正

平移目标为 cᵢ+sᵢzᵢ-c₀，torsion 目标为 a。
采样器每次 torsion 后做相对上一步的 Kabsch 对齐，因而会移除刚体自旋；
训练必须把这部分自旋补入两侧 angular velocity，不能仅监督 Log(Rᵢ,₁)。

对当前中心化形状 x，先按 directed bond axis 解析计算 torsion 的原子速度 w，
减去其均值，再计算：

- J = Σ (||xₙ||² I - xₙ xₙᵀ)；
- ω_shape = J⁺ Σ (xₙ × wₙ)；
- ωᵢ = Log(Rᵢ,₁) + Rᵢ(t) ω_shape。

POI-side 目标再映射到当前 E3/global frame。
这是一阶最小二乘刚体自旋分解；原子坐标有限差分用于独立测试，不参与生成
训练目标。路径直接使用光滑 bond rotations，禁止逐状态拟合回 native，
以免长链折叠构象的全局 Kabsch chart 引入奇异速度。

### 8.3 损失、预条件与反向积分

用按受体 sᵢ 和 π 归一化的速度 MSE，监督两个界面和唯一共享 torsion，
保留两侧 torsion 辅助监督。固定条件下的平方损失最优解为
v(x,t)=E[dXₜ/dt | Xₜ=x,t,条件]。
对任意光滑测试函数 f，
dE[f(Xₜ)]/dt = E[∇f(Xₜ)·v(Xₜ,t)]，因此该速度场满足路径边缘分布的
连续性方程。这是流匹配的数学依据，不保证有限网络或有限训练达到最优解。

DiffDock 的旧 sigma/score_norm 仅保留为预训练骨干的时间特征与输出预条件：
normalized translation = -sigma_tr × raw_tr；
normalized rotation = -raw_rot / score_norm_rot；
normalized torsion = -raw_tor / sqrt(E[score_tor²])。
这些不是“旧 score 自动等于新速度”的恒等式，TriFuse/LoRA 必须重新训练。

生产采样使用确定性的反向 Lie–Euler 积分：
中心减去 dt·v_tr，旋转左乘 Exp(-dt·ω)，torsion 更新为 -dt·v_tor，
并使用同源双 proposal 解析更新 POI/body frame。
不再使用经验温度、随机 SDE 增量或旧 DSM target。
`freeze_poi` 仅是改变生成过程的诊断，不属于完整双界面契约。

checkpoint 必须为 version 17、architecture=trifuse_v13、
generative_process=sdpi_flow_v1。旧 version 16 不允许静默恢复。
初始化本身仍是 sdpi_v1；评估报告和续跑签名另记生成过程版本。

### 8.4 验证与边界

运行 `python tests/test_sdpi_flow.py`，检查晶体/先验端点、完整坐标切向量、
步长收敛、全局等变性、反向 oracle、速度归一化和训练梯度。

真实案例脚本 `scripts/validate_sdpi_flow.py` 接受 `--case-dirs`、
`--seeds`、`--out`；可另加 `--gpu-smoke-dir`（必须为不存在的目录）。
GPU smoke 检查真实 optimizer step、LoRA warmup、参数更新、保存恢复及20步采样。
benchmark smoke 产物只作诊断，不能作为正式训练起点或精度报告。

首轮 5T35_D_A_759、6BN7_B_C_RN3、7JTO_L_B_VKA 各3个seed的几何门禁通过；
3次 GPU 更新、checkpoint 恢复和3个case各20步采样通过。
这不证明 DockQ 提升，更不证明流匹配优于经过充分训练的 DSM 基线。

2026-09-07 收尾全套回归为77项通过；真实几何复查9组通过，
半步完整坐标误差最大约9.01e-7 Å（dt=5e-5，仅作局部切向量门禁）。
GPU首轮报告为 `results/sdpi_flow_validation_v1.json`；最终复跑使用
`results/sdpi_flow_validation_final.json`。诊断 checkpoint 均与正式训练隔离。

旧 native DSM 的不匹配诊断保留在 `scripts/audit_initialization_consistency.py`；
该脚本固定审计旧过程，不是新 flow 的通过/失败门禁。其不可变几何诊断仍适用：
3个case、9次初始化的不变距离差为0.35–0.74 Å，保守 indexed-atom RMSD
下界约0.03–0.06 Å；这是下界，不是误差上界或 DockQ 界。

## 9. 碰撞、口袋和 MGD 边界

两侧独立二元初始化不保证合理的蛋白质—蛋白质界面，尤其 MGD 可能依赖由另一颗蛋白和配体共同形成的协同界面。两个 DiffDock 仅提供局部先验；TriFuse 必须观察 A–B、A–L、B–L 三类关系。

初始蛋白碰撞可以作为未筛选 baseline 被记录，但不能称为可用生产先验。碰撞拒绝、口袋约束或界面筛选都会改变联合分布，应作为显式模式，在训练和评估中分别报告，禁止静默启用。

## 10. 已实现与后续工作

已实现完整 ligand 共享构象、单次 torsion、双侧受体自适应尺度、解析装配，以及 sampler/CLI/trace/缓存/续跑签名接入。固定尺度与旧局部路径已删除；当前训练目标与采样改为 §8 的 SDPI Flow，模型骨干架构保持不变。

后续实验是：同 seed、同构象配置下比较独立 DiffDock A/B、双 expert＋固定 torsion 平均、训练后 TriFuse。当前 zero-init smoke 只验证有限采样，不替代三组精度对照。

版本化 `binary_prior` sidecar 和 prior adapter 见[单独设计](双二元强先验接入方案.md)，尚未实现。只有片段 expert 被单独训练和验证后，才研究以 warhead 片段替代完整配体输入。

## 11. 验收门禁

### CPU 几何与回归

1. 随机 SE(3) 下验证逆变换 roundtrip、`RᵀR=I`、`det(R)>0`。
2. 装配前后两侧 receptor–ligand 全部 `cdist` 保持，误差不超过 `1e-5`。
3. 验证两侧共享同一内部构象、canonical torsion 映射和方向翻转。
4. 当前直接保存变换并保持原子索引；非法 atom/torsion map 必须失败。未来若增加外部匹配/Kabsch，歧义对称匹配与少于三个非共线锚点必须拒绝，不能伪造唯一解。
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

正式实验必须固定代码 commit、权重、配置、数据版本、原子映射和随机种子。`sdpi` 已是唯一默认实现，但默认启用不等于通过全部精度、碰撞或训练后模型验收。
