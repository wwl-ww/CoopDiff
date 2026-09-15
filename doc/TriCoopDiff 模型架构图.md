# TriCoopDiff 模型与坐标契约

更新：2026-09-15。本文描述当前 `residual_consensus_v2`，不是旧 TriFuse v13 的使用说明。

## 当前架构

```text
同一三元状态（固定 E3，一份完整 ligand）
  ├─ 冻结 DiffDock：E3–ligand ─┐
  └─ 冻结 DiffDock：POI–ligand ├─ 转到共同 E3 frame、速度预条件
三元几何、共享时间与实际来源 ───┘
                 ↓
       ConsensusFuse：trust gates + equivariant residual
                 ↓
       两界面刚体速度 + 唯一共享 torsion 速度
                 ↓
       SDPIFlow.step：精确群组合、同步更新 POI
```

`ConsensusCoopDiff` 使用冻结专家；`ConsensusFuse` 读取完整三元几何、两侧 proposal、扩散状态和候选实际来源，用不变量条件预测标量系数并组合共同坐标系中的向量基。两个专家读取完整 ligand，不切割 warhead，不启用 LoRA、FiLM 或层间 bridge。

## 模型版本

| 家族 | architecture | checkpoint_version | 说明 |
| --- | --- | ---: | --- |
| 历史模型 | `trifuse_v13` | 17 | 仅用于识别旧 checkpoint |
| 标准 Consensus | `residual_consensus_v2` | 18–20 | 当前 B/C 模型；v20 记录双来源端点协议 |
| factorial 实验 | `residual_consensus_factorial_v1` | 1 | 冻结专家实验加载器 |

训练分支的 validation-loss 最佳模型为 source-conditioned C epoch 81 EMA（`3.1384998601`）；当前 PROTAC 精修工作基线为 C40 EMA，不能将模型版本与最优训练轮次混为一谈。Checkpoint v20 同时保存 raw 与 EMA TriFuse，并用空 expert-adapter 字段记录冻结契约。下游选择依据见[完整配对结果](../RANKING_PROTAC_RESEARCH.md)。

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
