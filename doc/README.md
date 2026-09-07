# TriCoopDiff 文档索引

- [Phase 1 数据协议](./Phase%201%20数据协议.md)：正式联合训练池、cluster
  切分/采样规则及 PROTAC/Molecular Glue benchmark 协议。

本目录只保留长期有效的设计、验证和运行文档。代码、测试与最新结果始终优先于文档；实验产物位于 `results/`，不要把临时日志或 agent 交接稿放入 `doc/`。

## 阅读顺序

[夜间实验编排](夜间实验编排.md)：十小时顺序执行器、验收报告和故障停止契约。
默认 `overnight_v2.json` 已接入单状态 v2 实际验收、限时训练和评估；
双构象历史模板仍拒绝启动，不属于本分支的实验职责。

独立二元对照的实测结果与 DockQ 小分子指标定义见
[独立 DiffDock 小分子评估](独立DiffDock小分子评估.md)。

当前 `model-v2` 入口使用冻结专家后的残差共识模型，设计动机见
[`sug_gpt6.md`](../sug_gpt6.md)。下表中 `trifuse_v13`、FiLM 和旧训练
交接文档是历史参考，不是 v2 的运行配置；旧 SLURM 的 LoRA 参数仍需迁移。

### 当前代码导航

- `tricoopdiff/consensus.py`：冻结专家调用、门控与残差速度场。
- `tricoopdiff/geometry_encoders.py`：配体化学特征、逐键上下文和三元几何编码。
- `tricoopdiff/consensus_layers.py`：token 协商、共享 torsion 与预留输出头。
- `tricoopdiff/expert_time.py`：独立二元对照与协同模型共用的 DiffDock 时间输入。
- `tricoopdiff/consensus_train.py`：残差正则、EMA 和 v2 checkpoint；目前复用
  `train.py` 的训练循环。`model.py`、`trifuse.py` 保留旧 FiLM 模型及兼容导出。
- `scripts/evaluation_runner.py`：训练权重的统一 EMA 评估；
  `evaluate_entry.py --allow-zero-init` 仅是未训练共识诊断，不是原生 DiffDock 对照。

公共层拆分不更改 `state_dict` 键或数值计算。回归入口为
`python tests/test_model_build.py`、`python tests/test_consensus.py` 和
`python tests/test_module_boundaries.py`；最后一项检查兼容导出及新旧模块隔离。

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
- v2 在冻结专家完成推理后生成门控先验与等变残差，协调共享 torsion；
  不向专家内部注入 FiLM、LoRA 或逐层 bridge。历史 `trifuse_v13` 才使用 FiLM。
- 生产训练使用 `SDPIFlow` 条件速度匹配，采样反向积分同一速度场；不再使用
  native DSM target 或经验温度 SDE。可动自由度终端与 SDPI 对齐，内部几何差异
  仍是已知近似；原始晶体监督不投影。数学契约见[初始化规范 §8](初始化方案_gpt6.md)。
- v2 checkpoint 必须是 version 18、`architecture=residual_consensus_v2`、
  `generative_process=sdpi_flow_v1`；评估使用 EMA。version 17 仅供旧模型参考，
  不能作为 v2 续训权重。运行配置记录 activation clamp、残差正则、EMA decay
  和 torus score norm 版本，并校验冻结专家身份。
- v2 已有构建、梯度、EMA 恢复及短轨迹检查，尚无充分训练后的质量结论。
- 当前只定义 Phase 1；`judge_head/state_head` 缺少监督标签，因此保持冻结，不能用无损失的“Phase 2”假装完成微调。
- Phase 1 使用 PROTAC/Molecular Glue 官方训练成员的去 benchmark 同 PDB
  交集（7,198 cases、653 merged clusters），并按 cluster 留出内部 validation。
- `binary_prior` sidecar 和 prior adapter 仅有设计，尚未实现；不属于 `sdpi`。已合并重复的三分子初始化文档，以初始化规范为唯一入口。

## 维护规则

每份主题只保留一个权威文档。新结论直接更新对应文档，不再创建“交接 v2”“最终版”或一次性任务 brief。命令必须从仓库根目录运行；结果路径只作示例，不作为永久证据。删除过时文档前，将仍有效的结论合并到上表对应文件。
