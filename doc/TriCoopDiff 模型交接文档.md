# TriCoopDiff 模型交接文档

> 历史文档：本文描述旧 `trifuse_v13`/LoRA/FiLM 架构，不是当前
> `residual_consensus_v2` 交付模型。当前模型与实验结论以
> [SDPI–BiTSeed 联合训练交接](SDPI-BiTSeed联合训练交接.md)为准。

## 模型定位

TriCoopDiff 用两个带 branch-specific LoRA/FiLM 的冻结 DiffDock expert，分别生成 E3–Ligand 与 POI–Ligand 界面 proposal。模型直接在两个界面相对坐标上扩散，再通过精确 SE(3) 组合重建三元复合物；它不是两个 Ligand score 的融合器。

## 坐标状态

E3 是固定全局参考系。动态状态为：

| 状态 | 含义 | 负责模块 |
|---|---|---|
| `T_E3L` | Ligand 相对 E3 的刚体 | `sub1` DiffDock |
| `T_POL` | Ligand 相对 POI 的刚体 | `sub2` DiffDock |
| `chi` | Ligand torsion | TriFuse 逐键协调两 expert evidence |

全局坐标由 `T_L = T_E3L`、`T_POI = T_L · inverse(T_POL)` 唯一确定。任何坐标更新后都必须同步刷新 POI-local frame、receptor crop 和 cross-edge。

## 双 expert 协作

两个分支使用同一个预训练 DiffDock 基座，但 LoRA 和 FiLM 参数互相独立：

- 两个分支在同一个旧 Ligand 上分别生成 `L1`、`L2` rigid proposal；
- Ligand 采用 `L1`，POI 接收把 `L2` 对齐到 `L1` 的同一个精确刚体变换；
- torsion 是共享内部自由度，由 TriFuse 逐键预测两侧权重与 residual，并只应用一次。

禁止恢复以下旧逻辑：

- `s1 + g * (s2 - s1)` residual-gated merge；
- 将两个 expert 的刚体 score 都用于移动 Ligand；
- 让独立 TriFuse SE(3) head 从头预测 POI movement；
- 对 rotation vector 简单取负来近似逆刚体更新。

## TriFuse

TriFuse 读取三元上下文并输出：

1. E3、POI、Ligand 分别池化的类型 token；
2. E3–L、POI–L、E3–POI 的 RBF 距离分布、接触比例和尺度编码；
3. `sub1`、`sub2` 各层独立的 FiLM scale/shift；
4. 逐键 torsion coordinator；
5. scalar auxiliary heads，例如功能或置信度预测。

TriFuse 不自行输出笛卡尔 rigid score、merge gate 或 SE(3) residual。它通过 FiLM/bridge 和有界 score gain 校准专家建议，并裁决共享 torsion；刚体方向仍来自两个 DiffDock，坐标由显式群组合定义。功能/置信度 heads 当前无监督且冻结，不能用于可靠排序。

## 训练

从真实三元坐标构造通向 SDPI 终端的双界面条件流路径；随后解析重建一致的全局坐标和二体图。当前训练是 SDPI velocity matching，不是 native DiffDock DSM。

两侧共享流时间；sigma/score_norm 仅作为预训练输出的预条件。速度 target 包含 torsion 对齐所移除的自旋，采样反向 Lie–Euler 积分，不再叠加经验温度随机项。原始晶体目标不投影，详见[初始化规范 §8](初始化方案_gpt6.md)。

## 采样

一个反向步的固定顺序是：

1. 在同一个旧 Ligand 上同时得到 `sub1` 的 `L1` 与 `sub2` 的 `L2`；
2. Ligand 采用 `L1`，把 `L2 → L1` 的刚体变换同步施加给 POI；
3. 应用一次 TriFuse 协调后的共享 torsion；
4. 更新 POI local frame；
5. 重建并验证全部 frame 与图。

初始化唯一支持 [`sdpi`](初始化方案_gpt6.md)。完整 ligand 从化学图重新生成共享构象，A/B 独立采样尺寸自适应刚体先验后精确装配。旧组件级和局部初始化已删除，当前不支持二元端点 sidecar。

## Checkpoint 与恢复

当前架构标识为 `trifuse_v13`，checkpoint version 为 17（generative_process=sdpi_flow_v1）。两个专家按层同步运行：每层各执行一次卷积，再按 canonical ligand atom ID 交换 scalar irreps，生成有界、zero-init 的下一次调制。bridge 不交换不同 frame 的 vector irreps，不增加卷积调用数。旧权重不兼容。

## 最小验收

- 两个 branch 的 native/inline parity 与 finite 检查；
- 相对/全局 frame round trip 和 SE(3) 等变测试；
- 同源双 proposal 与 `L2 → L1` 跨界面距离不变量测试；
- TriFuse torsion coordinator 零初始化、逐键输出及单次应用测试；
- 三体 context 对界面几何和组分类型敏感，并对共同 SE(3) 变换保持不变；
- 两侧 expert torsion 均接受辅助监督，正式入口检查全部 TriFuse/LoRA 梯度；
- forward-only、单 optimizer step、训练/验证 smoke 全 finite；
- 非 version 17、错误 architecture 或不完整运行配置的 checkpoint 均被明确拒绝。

详细坐标公式见 [公式推导参考.md](公式推导参考.md)，实现原则见 [双DiffDock界面协同修正方案.md](双DiffDock界面协同修正方案.md)。
