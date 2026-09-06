# TriCoopDiff 文档索引

- [Phase 1 数据协议](./Phase%201%20数据协议.md)：正式联合训练池、cluster
  切分/采样规则及 PROTAC/Molecular Glue benchmark 协议。

本目录只保留长期有效的设计、验证和运行文档。代码、测试与最新结果始终优先于文档；实验产物位于 `results/`，不要把临时日志或 agent 交接稿放入 `doc/`。

## 阅读顺序

| 文档 | 用途 | 何时阅读 |
|---|---|---|
| [TriCoopDiff 模型架构图.md](TriCoopDiff%20模型架构图.md) | 完整 Mermaid 模型图、模块职责与单步坐标更新 | 快速查看当前整体架构时 |
| [TriFuse 架构与审计.md](TriFuse%20架构与审计.md) | `trifuse_v13` 数据流、参考系、训练契约与已知边界 | 快速理解或审计当前模型时 |
| [TriCoopDiff 模型交接文档.md](TriCoopDiff%20模型交接文档.md) | `trifuse_v13` 架构、FiLM、训练和采样 | 修改核心模型前 |
| [双DiffDock界面协同修正方案.md](双DiffDock界面协同修正方案.md) | 双 expert 的界面职责与刚体更新顺序 | 修改坐标动力学时 |
| [双二元强先验接入方案.md](双二元强先验接入方案.md) | 双端点、双口袋与局部 frame 的输入契约，以及继续训练/从头训练两条路线 | 对齐 DeepTernary 的预测先验时 |
| [DeepTernary与TriCoopDiff训练推理初始化对照.md](DeepTernary与TriCoopDiff训练推理初始化对照.md) | 当前初始化与 DeepTernary 不同入口的输入条件差异 | 设计公平对照实验时 |
| [初始化方案_gpt6.md](初始化方案_gpt6.md) | 唯一 `sdpi` 初始化、共享构象、精确装配与验收 | 修改采样或初始化时 |
| [训练入口 SLURM 与 Phase1 smoke 交接文档.md](训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md) | 环境预检、SLURM、缓存和训练入口 | 提交训练前 |
| [DiffDock 审计文档.md](DiffDock%20审计文档.md) | 冻结二体专家的结构和扩散实现 | 修改 expert 集成时 |
| [公式推导参考.md](公式推导参考.md) | SO(3)、DSM、Tweedie 和物理项推导 | 核对数学约定时 |

## 当前权威结论

- 历史 [DeepTernary 审计产物](../results/audit_sota_metrics.json) 记录 mean oracle-best DockQ 0.6588（22 cases × 40 seeds）；不是当前模型结果或 model-ranked top-1，比较前须核对入口与输入先验。
- 与 DeepTernary 横向比较时，主指标必须调用其完整原子 `DockQ/dockq_util.py::cal_dockq`；Cα 或接触恢复近似值只能作为诊断指标。
- E3 固定；动态状态是 E3–Ligand 与 POI–Ligand 两个相对刚体界面以及共享 Ligand torsion。
- 推理唯一支持 `sdpi`：从完整化学图生成一次构象并采样一次 torsion，分别使用两侧受体 RMS radius 计算平移尺度，再精确逆装配；旧 `native`、`coupled5a`、`anchor_local` 和固定 Å 初始化均已删除。
- 两个 rigid proposal 从同一个旧 Ligand 生成；Ligand 采用 `sub1` proposal，POI 接收将 `sub2` proposal 对齐到 `sub1` proposal 的精确刚体变换。
- TriFuse 使用组分类型、三组跨分子距离分布和接触统计，负责 branch-specific FiLM 与逐键 torsion 协调；它不输出刚体 merge gate 或独立 SE(3) score。
- 训练分别对两个界面应用 native DiffDock forward noise，再重建唯一一致的全局坐标。
- 该训练终端与 `sdpi_v1` **不是同一分布**；平移中心/尺度、有限角噪声和
  配体不可变内部几何均需处理。数学审计与真实样例见
  [初始化规范 §8](初始化方案_gpt6.md#8-训练推理一致性)，不能把装配一致当作分布一致。
- 新 checkpoint 必须是 version 16 且 `architecture=trifuse_v13`；旧版本必须拒绝。
- version 16 记录 activation clamp、torsion expert 辅助损失与 LoRA warmup；训练与评估必须恢复一致配置。
- 当前只定义 Phase 1；`judge_head/state_head` 缺少监督标签，因此保持冻结，不能用无损失的“Phase 2”假装完成微调。
- Phase 1 使用 PROTAC/Molecular Glue 官方训练成员的去 benchmark 同 PDB
  交集（7,198 cases、653 merged clusters），并按 cluster 留出内部 validation。
- `binary_prior` sidecar 和 prior adapter 仅有设计，尚未实现；不属于 `sdpi`。已合并重复的三分子初始化文档，以初始化规范为唯一入口。

## 维护规则

每份主题只保留一个权威文档。新结论直接更新对应文档，不再创建“交接 v2”“最终版”或一次性任务 brief。命令必须从仓库根目录运行；结果路径只作示例，不作为永久证据。删除过时文档前，将仍有效的结论合并到上表对应文件。
