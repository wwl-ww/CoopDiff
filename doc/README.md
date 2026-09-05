# TriCoopDiff 文档索引

本目录只保留长期有效的设计、验证和运行文档。代码、测试与最新结果始终优先于文档；实验产物位于 `results/`，不要把临时日志或 agent 交接稿放入 `doc/`。

## 阅读顺序

| 文档 | 用途 | 何时阅读 |
|---|---|---|
| [TriCoopDiff 模型交接文档.md](TriCoopDiff%20模型交接文档.md) | `trifuse_v7` 架构、FiLM、训练和采样 | 修改核心模型前 |
| [双DiffDock界面协同修正方案.md](双DiffDock界面协同修正方案.md) | 双 expert 的界面职责与刚体更新顺序 | 修改坐标动力学时 |
| [三分子初始化策略方案.md](三分子初始化策略方案.md) | `native` 与 `coupled5a` 初始化、参考系和验收 | 修改采样或初始化时 |
| [训练入口 SLURM 与 Phase1 smoke 交接文档.md](训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md) | 环境预检、SLURM、缓存和训练入口 | 提交训练前 |
| [DiffDock 审计文档.md](DiffDock%20审计文档.md) | 冻结二体专家的结构和扩散实现 | 修改 expert 集成时 |
| [公式推导参考.md](公式推导参考.md) | SO(3)、DSM、Tweedie 和物理项推导 | 核对数学约定时 |

## 当前权威结论

- DeepTernary 可复现实验基线为 **mean best DockQ 0.6588**（22 cases × 40 seeds）。
- 与 DeepTernary 横向比较时，主指标必须调用其完整原子 `DockQ/dockq_util.py::cal_dockq`；Cα 或接触恢复近似值只能作为诊断指标。
- E3 固定；动态状态是 E3–Ligand 与 POI–Ligand 两个相对刚体界面以及共享 Ligand torsion。
- 两个 rigid proposal 从同一个旧 Ligand 生成；Ligand 采用 `sub1` proposal，POI 接收将 `sub2` proposal 对齐到 `sub1` proposal 的精确刚体变换。
- TriFuse 使用组分类型、三组跨分子距离分布和接触统计，负责 branch-specific FiLM 与逐键 torsion 协调；它不输出刚体 merge gate 或独立 SE(3) score。
- 训练分别对两个界面应用 native DiffDock forward noise，再重建唯一一致的全局坐标。
- 新 checkpoint 必须是 version 10 且 `architecture=trifuse_v7`；旧版本必须拒绝。
- version 10 记录 activation clamp 与 torsion expert 辅助损失权重；训练与评估必须恢复一致配置。
- Phase 1 使用官方 DeepTernary MGD cluster representatives 训练，并保持 train/validation/test 三个索引互不重叠。

## 维护规则

每份主题只保留一个权威文档。新结论直接更新对应文档，不再创建“交接 v2”“最终版”或一次性任务 brief。命令必须从仓库根目录运行；结果路径只作示例，不作为永久证据。删除过时文档前，将仍有效的结论合并到上表对应文件。
