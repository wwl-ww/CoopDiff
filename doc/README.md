# TriCoopDiff 文档索引

代码、测试和 `results/` 中的完整报告优先于文档。本文只负责路由，不重复保存实验
流水账；一次性运行状态见仓库根目录 [`todo.md`](../todo.md)。

## 当前阅读顺序

1. [低 DockQ 根因排查交接](低DockQ根因排查交接.md)：model-v2 已完成的根因证据、
   MGD94 oracle、双专家 anchor 门禁，以及下一分支的在线初始化方向。
2. [初始化方案](初始化方案_gpt6.md)：现有 SDPI 初始化、共享 ligand 构象和精确刚体
   装配契约。该文档描述当前基线；下一分支需补充确定性在线 anchor 生成。
3. [TriCoopDiff 模型架构图](TriCoopDiff%20模型架构图.md)：当前模块和坐标更新关系。
4. [Phase 1 数据协议](Phase%201%20数据协议.md)：训练池、cluster split 和 benchmark
   防泄漏规则。
5. [训练入口与 smoke](训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md)：环境、
   缓存、SLURM 和训练入口。

## 专题参考

| 文档 | 用途 | 状态 |
|---|---|---|
| [DiffDock 审计](DiffDock%20审计文档.md) | frozen 二元专家、扩散和 checkpoint | 当前参考 |
| [公式推导](公式推导参考.md) | SO(3)、DSM、Tweedie 与物理项 | 当前参考 |
| [TriFuse 架构与审计](TriFuse%20架构与审计.md) | 旧 `trifuse_v13`/FiLM 数据流 | 历史参考 |
| [TriCoopDiff 模型交接](TriCoopDiff%20模型交接文档.md) | 旧模型训练与采样契约 | 历史参考 |
| [双 DiffDock 界面协同](双DiffDock界面协同修正方案.md) | 双专家职责与刚体更新 | 设计背景 |
| [双二元强先验](双二元强先验接入方案.md) | 双端点和口袋先验设想 | 未实现设计 |
| [DeepTernary 初始化对照](DeepTernary与TriCoopDiff训练推理初始化对照.md) | 不同输入条件的公平性 | 对照参考 |
| [独立 DiffDock 评估](独立DiffDock小分子评估.md) | 二元对照和指标定义 | 对照参考 |
| [夜间实验编排](夜间实验编排.md) | 已结束的夜间执行协议 | 历史记录 |

## 当前事实边界

- MGD 是当前主目标；PROTAC 后续会加入 warhead/口袋信息。
- 完整原子 DeepTernary DockQ 是主指标；近似 DockQ 只作诊断。
- model-v2 的核心瓶颈是联合位置与朝向的搜索覆盖，不是单一实现错误或小幅 loss。
- `sdpi` 是生产基线；`expert_anchor` 是已通过 20-case 门禁的实验初始化，不能静默
  替换生产协议。
- 下一分支必须从原始输入在线生成 anchor，禁止候选路径读取 native 或加载可人工
  编辑的 pose cache；MGD94 不参与 selector 训练。
- oracle-best、native-reading oracle 和 model-ranked top-1 必须分别报告。

## 代码导航

- `tricoopdiff/initialization.py`、`tricoopdiff/sampler.py`：基线初始化与采样。
- `tricoopdiff/pose_anchors.py`：统一 proper SE(3) 拟合和表面 anchor 原语。
- `tricoopdiff/expert_anchors.py`：双专家 geometry/consensus anchor。
- `tricoopdiff/consensus.py`：冻结专家与残差共识模型。
- `scripts/evaluation_runner.py`、`scripts/evaluate_index.py`：可恢复评估和协议清单。

## 维护规则

每个主题只保留一个权威入口。新结论替换旧状态，不创建“最终版”“交接 v2”或一次性
brief。删除历史文档前，先把仍有效的契约合并到当前文档；不得把生成结果提交为源码。
