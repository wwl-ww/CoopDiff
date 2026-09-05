# TriCoopDiff 模型交接文档

## 模型定位

TriCoopDiff 用两个带 branch-specific LoRA/FiLM 的冻结 DiffDock expert，分别生成 E3–Ligand 与 POI–Ligand 界面 proposal。模型直接在两个界面相对坐标上扩散，再通过精确 SE(3) 组合重建三元复合物；它不是两个 Ligand score 的融合器。

## 坐标状态

E3 是固定全局参考系。动态状态为：

| 状态 | 含义 | 负责模块 |
|---|---|---|
| `T_E3L` | Ligand 相对 E3 的刚体 | `sub1` DiffDock |
| `T_POL` | Ligand 相对 POI 的刚体 | `sub2` DiffDock |
| `chi` | Ligand torsion | 两 expert proposal 均值 |

全局坐标由 `T_L = T_E3L`、`T_POI = T_L · inverse(T_POL)` 唯一确定。任何坐标更新后都必须同步刷新 POI-local frame、receptor crop 和 cross-edge。

## 双 expert 协作

两个分支使用同一个预训练 DiffDock 基座，但 LoRA 和 FiLM 参数互相独立：

- `sub1` 在标准 E3–Ligand 二体图上运行，直接更新 Ligand rigid pose；
- `sub2` 在标准 POI–Ligand local frame 图上运行，保持 Ligand fixed，通过 proposal 的精确逆刚体变换更新 POI；
- torsion 是 Ligand 的共享内部自由度，取两分支 proposal 的算术均值并只应用一次。

禁止恢复以下旧逻辑：

- `s1 + g * (s2 - s1)` residual-gated merge；
- 将两个 expert 的刚体 score 都用于移动 Ligand；
- 让独立 TriBrain SE(3) head 从头预测 POI movement；
- 对 rotation vector 简单取负来近似逆刚体更新。

## TriBrain

TriBrain 读取三元上下文并输出：

1. `sub1`、`sub2` 各层独立的 FiLM scale/shift；
2. scalar auxiliary heads，例如功能或置信度预测。

TriBrain 不输出坐标 score、merge gate 或 SE(3) residual。坐标动力学完全由两个 DiffDock proposal 与显式刚体组合定义，避免让轻量网络重新学习二体 docking。

## 训练

从真实三元坐标提取 `T_E3L,0` 与 `T_POL,0`。分别使用 native DiffDock forward process 对两个界面加平移、旋转噪声，并对共享 torsion 加噪；随后从噪声界面状态重建一致的全局坐标和二体图。

每个 expert 必须接收与自身界面噪声一致的时间、sigma 和 target。不得独立扰动全局 POI，再把该相对噪声错误地标记为 Ligand 时间。训练损失直接监督两个界面 proposal，辅助 scalar loss 不得改变坐标 target。

## 采样

一个反向步的固定顺序是：

1. 用 `sub1` 更新 `T_E3L` 和 Ligand；
2. 重建 `sub2` 输入；
3. 用 `sub2` 更新 `T_POL`，由 `T_POI = T_L · inverse(T_POL)` 移动 POI；
4. 应用一次平均 torsion；
5. 重建并验证全部 frame 与图。

初始化模式只提供起始界面 proposal。它不得引入第三套坐标语义，也不得以 near-native pose 掩盖全局装配能力。

## Checkpoint 与恢复

当前架构标识为 `interface_coordinate_v1`，checkpoint version 为 4。加载时必须同时验证 version、architecture、LoRA 配置和模型 shape。version 3 使用旧动力学和旧参数语义，必须硬拒绝；不支持 legacy mode、自动迁移或用 v3 继续微调。

## 最小验收

- 两个 branch 的 native/inline parity 与 finite 检查；
- 相对/全局 frame round trip 和 SE(3) 等变测试；
- `sub1` 只移动 Ligand、`sub2` 只移动 POI 的行为测试；
- 两 expert torsion 均值且每步只应用一次；
- forward-only、单 optimizer step、训练/验证 smoke 全 finite；
- v3 和错误 architecture checkpoint 均被明确拒绝。

详细坐标公式见 [公式推导参考.md](公式推导参考.md)，实现原则见 [双DiffDock界面协同修正方案.md](双DiffDock界面协同修正方案.md)。
