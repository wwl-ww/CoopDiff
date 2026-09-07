# TriCoopDiff 文档索引

当前分支为 `model-v2-dual-consensus`。当前模型、数学契约、训练/推理入口和验收边界以
[DualConsensus.md](DualConsensus.md) 为准；设计动机见 [sug_gpt6.md](../sug_gpt6.md)。

## 当前实现

- E3 固定，两份配体几何独立；POI 有独立的可学习刚体速度。
- 两个二体 expert 通过 detached 标量桥接和 branch-specific FiLM 交换信息。
- 同一 ETKDG 基构象流形，双份独立 torsion/pose 先验，可达目标投影监督。
- 渐进 internal/torsion/endpoint/pose 共识，加独立 POI 和三元坐标/接触监督。
- 简单物理 collapse 输出 A 配体与预测 POI；不平均原子坐标，并报告残余分歧。
- checkpoint v18 / `dual_consensus_v1` / `dual_consensus_flow_v1`；拒绝 v17。
- 当前仍仅定义 Phase 1。没有标签的 judge/state heads 保持冻结。
- `valid` 只表示数值完成；局部结合、共识和完整 ternary DockQ 必须分别评估。

## 数据与专家参考

| 文档 | 用途 |
|---|---|
| [Phase 1 数据协议](Phase%201%20数据协议.md) | PROTAC/MGD 官方训练池、切分及 benchmark 协议 |
| [DiffDock 审计文档](DiffDock%20审计文档.md) | 冻结二体专家结构与预训练输出单位 |
| [公式推导参考](公式推导参考.md) | 旋转等基础数学；DSM 部分不是当前训练目标 |

与 DeepTernary 横向比较时，主指标应使用其完整原子 DockQ；接触 surrogate 或 Cα 近似指标不能替代它。

## 历史主线参考

以下文档记录 `phase1-run` 的 v17 共享配体模型，不是本分支的动力学或 checkpoint 契约：

- [TriCoopDiff 模型架构图](TriCoopDiff%20模型架构图.md)
- [TriFuse 架构与审计](TriFuse%20架构与审计.md)
- [TriCoopDiff 模型交接文档](TriCoopDiff%20模型交接文档.md)
- [双DiffDock界面协同修正方案](双DiffDock界面协同修正方案.md)
- [初始化方案](初始化方案_gpt6.md)
- [DeepTernary 初始化对照](DeepTernary与TriCoopDiff训练推理初始化对照.md)
- [双二元强先验接入方案](双二元强先验接入方案.md)：未实现的 pocket/prior adapter 设计。
- [旧训练入口交接](训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md)：集群环境参考；新分支使用独立输出目录。

代码和测试优先于旧研究笔记。实验产物保存在 `results/`；不要将 smoke checkpoint 当作正式训练模型。
