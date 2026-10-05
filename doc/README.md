# 文档索引

更新：2026-09-16。当前代码定义实现，实验 manifest 与逐例结果定义已运行事实；提案和历史 smoke 不能替代二者。

## 阅读顺序

| 文档 | 唯一负责的主题 |
| --- | --- |
| [项目概览](../README.md) | 项目定位、目录与快速检查 |
| [当前工作](../todo.md) | 当前分支状态与下一步 |
| [联合训练交接](SDPI-BiTSeed联合训练交接.md) | SDPI/BiTSeed 路径、B/C 结果、最终 checkpoint 和 LoRA 结论 |
| [BiTSeed-Hybrid 协议](BiTSeed-Hybrid在线初始化规格.md) | 候选生成、fallback、selector、当前实验榜单 |
| [模型与坐标契约](TriCoopDiff%20模型架构图.md) | 当前 Consensus、SDPI flow、source conditioning、版本与不变量 |
| [数据与指标协议](Phase%201%20数据协议.md) | 训练/验证划分、MGD/PROTAC 输入和评分边界 |
| [运行指南](训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md) | 环境、入口、复跑与验收；SLURM 仅为历史记录 |
| [问题账本](../Q.md) | 已修复缺陷、未解决限制和证据路径 |
| [独立二元评估](独立DiffDock小分子评估.md) | 历史两案例对照、小分子 LRMSD/DockQ 口径 |
| [Ranking与PROTAC条件调研](../RANKING_PROTAC_RESEARCH.md) | 2026-09-15研究提案、C40/C81完整对照、原始资料与后续实验；不是已实现协议 |

`AGENTS.md` 为贡献约定。`hh.md` 是用户原始讨论稿，非现行实现规格。

## 当前事实边界

- B/C 均已完成 100 epochs；训练分支按 validation loss 交付 C81 EMA（`3.1384998601`），本分支 PROTAC 精修工作基线仍为 C40 EMA。完整22例 Final/Union差异见[配对结果](../RANKING_PROTAC_RESEARCH.md)。
- 正式模型使用冻结 DiffDock experts。LoRA 从头训练与 best-checkpoint 微调均未超过父模型，已停止并删除可训练实现。
- 联合训练 BiTSeed 固定 lazy `C=1, M=1`；两个二元专家各运行一次，任一失败时回退 SDPI。
- MGD94与PROTAC22按各自输入协议分别报告；当前还在研究PROTAC额外条件，不能混比两种数据。完整原子 DeepTernary DockQ 是主指标，近似 DockQ 只作诊断。
- 历史 POI–配体接触指标曾在原子不对应时混入接触密度比例；PROTAC22 单候选面板 17/22 例受影响。新评估已拆分字段，旧档案须经对应性审计，不能直接称“双界面都好”。
- BiTSeed 本身不包含 oracle、selector 或 ranker。
- model-v3 将ranking拆为全池初态256→64 pre-selector与已精修状态post-ranker；两者标签、checkpoint和指标不可互换。PROTAC条件/全缺失40-epoch配对训练及PROTAC22三组结构审计均已完成：混合条件训练的无条件推理有平均收益，但pocket-only输入的直接边际为负，故不列为默认。完整数值与边界见`todo.md`。

## 维护规则

每个主题一个主文档，不新增“最终版/交接 v2”。新协议同时更新实现与对应文档；尚未实现的能力必须明确标注。不要把静态 GPU 占用、一次性临时环境或已删除脚本写成可直接执行的当前入口。
