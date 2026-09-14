# TriCoopDiff 模型与坐标契约

更新：2026-09-14。本文合并当前模型、专家、SDPI 初始化与流公式；不是旧 TriFuse v13 的使用说明。

## 当前架构

```text
同一旧三元状态（固定 E3，一份完整 ligand）
  ├─ 冻结 DiffDock：E3–ligand ─┐
  └─ 冻结 DiffDock：POI–ligand ├─ 转到共同 E3 frame、速度预条件
三元几何与共享时间 ───────────┘
                 ↓
       ConsensusFuse：trust gates + equivariant residual
                 ↓
       两界面刚体速度 + 唯一共享 torsion 速度
                 ↓
       SDPIFlow.step：精确群组合、同步更新 POI
```

[consensus.py](../tricoopdiff/consensus.py) 的 `ConsensusCoopDiff` 使用不可训练调制的冻结专家；`ConsensusFuse` 读取专家建议后，用不变量条件预测标量系数，组合共同坐标系中的向量基。当前模型**有残差速度场**，不能继续套用旧“Fuse 不输出 rigid residual”的描述。

两个专家读取完整 ligand，不切割 warhead，不向 frozen 图注入新字段；相同基座复用不等于两个独立训练的专家。当前 baseline 不启用 LoRA、FiLM 或层间 bridge。`judge_head/state_head` 无有效排序监督且被冻结；trust gate 不是校准置信度。

## 模型版本不能混用

| 家族 | architecture | checkpoint_version | 说明 |
| --- | --- | ---: | --- |
| 旧 FiLM/LoRA | `trifuse_v13` | 17 | 遗留模型/训练代码，非本分支精修基线 |
| 标准 Consensus | `residual_consensus_v2` | 18 | `ConsensusCoopDiff` |
| factorial 实验 | `residual_consensus_factorial_v1` | 1 | 实验加载器；具体 adaptation 以配置为准 |

MGD94 当前实验用第三行的 frozen arm。检查架构、完整 runtime config、EMA、冻结专家身份和数值版本，不可只比较整数版本或把 SDPIFlow 内部的 version 17 常量当作所有模型的版本。详见 [consensus_experiment.py](../tricoopdiff/consensus_experiment.py)。

## 参考系与一步更新

E3 固定。概念状态为两个相对 SE(3) 位姿 `T_E3L, T_POL` 和共享 torsion `chi`：

```text
T_L   = T_E3L
T_POI = T_L · inverse(T_POL)
L_new   = D_A(L_old)
POI_new = D_A · inverse(D_B)(POI_old)
```

`D_A/D_B` 都必须从同一个旧 ligand 生成。用完整刚体乘法与逆，不能独立取负平移或把旋转向量相减当作 SE(3) 组合。随后 torsion 只应用一次，更新 POI body frame、crop 和 cross-edge。精确保留两侧 proposal 的界面距离只针对 torsion 前的刚体装配。

实现多用行向量，公式若用列向量必须明确转置。跨专家 frame 不直接交换 vector irreps；先旋转至共同 frame。见 [rigid.py](../tricoopdiff/rigid.py)、[sdpi_flow.py](../tricoopdiff/sdpi_flow.py)。

## SDPI 初始化

SDPI = **Shared-Ligand Dual-Pose Initialization**，模式 `sdpi`，版本 `sdpi_v1`。

从保持 canonical atom 顺序的完整 RDKit 图生成 ETKDGv3 构象（默认随机坐标策略），只采样一次完整 torsion。两侧复制同一内部构象，分别采样 Haar SO(3) 和高斯平移，再解析装配。每侧独立计算：

```text
r_rec = sqrt(mean(||p_j - mean(p)||²))
sigma_init = r_rec × 1.4601642460337794 / 1.73
translation_noise ~ Normal(0, sigma_init² I)
```

必须保存实际旋转、中心和平移，不用事后拟合猜回已知变换。`tr_sigma_max=19` 不是初始化尺度。SDPI 不调用 DiffDock 生成初态，也不保证蛋白不穿插或口袋正确。

BiTSeed 与固定候选是另外的初始化路径，见[候选协议](BiTSeed-Hybrid在线初始化规格.md)。不再声称全仓库“只支持 SDPI”；单例标准入口仍只暴露 SDPI。

## SDPI flow 的训练与采样

当前生成过程为 `sdpi_flow_v1`：clean/native 为 t=0，SDPI 可动自由度先验为 t=1，两侧共享时间。不是训练两条独立时间的 DiffDock DSM，也不是随机 SDE 精修。

设晶体 ligand 中心为 c₀，受体中心 cᵢ，受体尺度 sᵢ，采样 zᵢ、Haar Rᵢ,₁ 与共享扭转增量 a：

```text
mᵢ(t) = (1-t)c₀ + t(cᵢ+sᵢzᵢ)
Rᵢ(t) = Exp(t Log(Rᵢ,₁))
C(t)  = native 经 t·a 的键旋转，再中心化到 c₀
Lᵢ(t) = (C(t)-c₀) Rᵢ(t)ᵀ + mᵢ(t)
```

训练目标为解析路径速度。torsion 对齐会去除刚体自旋，必须把自旋补回两侧 angular target：对中心化形状 x、去平移原子速度 w，`J=Σ(||x||²I-xxᵀ)`，`ω_shape=J⁺Σ(x×w)`。各侧角速度包含旋转后的该项；POI 侧再映射到 E3 frame。有限差分只作独立测试，不生成训练 target。

DiffDock sigma/score_norm 是预训练特征与输出预条件，转换包括 `-sigma_tr*raw_tr`、`-raw_rot/score_norm_rot`、`-raw_tor/sqrt(E[score_tor²])`，不意味着原始 score 天然就是正确流速度。

采样采用反向 Lie–Euler：平移减 dt·v，旋转左乘 Exp(-dt·ω)，torsion 减 dt·v_tor；无额外温度随机项。较短 t_start 只用于显式 fixed-candidate 诊断，不能把 BiTSeed 任意解释成 SDPI 的某个时间边缘分布。

## 专家与不可变契约

Vendored 来源说明见 [diffdock/__init__.py](../tricoopdiff/diffdock/__init__.py)，加载见 [expert_factory.py](../tricoopdiff/expert_factory.py)。当前 score 参数为 ns/nv=48/10、11 层、Cα 受体图、完整 ligand 原子图；实际权重与配置必须配套。新增入口先导入 `tricoopdiff.diffdock_compat`。

检查完整 atom/torsion 映射、SE(3) round trip、等变性、同源双 proposal、单次 torsion、动态 frame、有限输出与缓存身份。不能用零值替换 NaN；activation clamp 必须恢复 checkpoint 配置。

## 尚未解决或未接入

训练 native scaffold 与推理 ETKDG 的键角/环系等不可动内部几何不同；刚体和 torsion 终端分布一致不等于全坐标支持一致。参见 [Q-012](../Q.md#q-012训练与推理的内部几何支持)。

联合训练计划在 `exp/dual-init`：实际双来源端点、正确路径与 gauge、一套共享共识模型，并比较有无来源 embedding。来源沿轨迹不变，SDPI 补位标为 SDPI；训练混合比例不等于推理配额。来源条件应接入当前 Consensus/Fuse 接口，不套用旧 v13 FiLM 图；仅新增来源 bit 不能修复不可达 target。

外部 warhead pose、口袋 sidecar、片段专家及 prior adapter 是另一个未落地方案，不是当前 BiTSeed 的必需输入，也不能宣称已具有 DeepTernary PROTAC 的同输入条件。
