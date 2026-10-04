# TriCoopDiff / BiTSeed

TriCoopDiff 研究蛋白–配体–蛋白三元装配：两个冻结 DiffDock 专家提供二元建议，残差共识模型协调双界面运动。BiTSeed 是在线初始化模块，不是精修模型或排序器。

## 当前研究状态

候选池工作配置为 **192 BiTSeed + 64 SDPI**，从中选择 64 个初态交给 TriCoopDiff。该配置是对照基线，不是已证明的全局最优配置；selector 尚未冻结。候选生成、失败补位和 MGD94 审计见[初始化与实验协议](doc/BiTSeed-Hybrid在线初始化规格.md)。

当前双线为 ranking 与 PROTAC 条件建模。已有学习型 pre-selector 显示计算分配信号，但尚无可部署的 native-free top-1 ranker；pocket-only 输入也未产生稳定正收益，不列为默认。完整数值、当前 checkpoint 边界和后续门禁统一见 [todo.md](todo.md) 与[Ranking/PROTAC 调研](RANKING_PROTAC_RESEARCH.md)；联合训练的实现与 LoRA 结论见[联合训练交接](doc/SDPI-BiTSeed联合训练交接.md)。

## 目录

- `tricoopdiff/models/`：生产模型，仅包含模型本体（编码、交互、运动输出）和冻结专家两个实现文件。
- `tricoopdiff/legacy/`：历史 FiLM 模型；不由生产模型导入。
- `tricoopdiff/experiments/`：诊断实验模型与训练目标；不由生产模型导入。
- `tricoopdiff/conditioning/`、`tricoopdiff/ranking/`：可选条件建模与候选排序。
- `tricoopdiff/inputs/`：蛋白嵌入和图构造。
- `tricoopdiff/geometry/`：刚体变换、等变数值保护和物理几何检查。
- `tricoopdiff/initialization/`：初始化分布、双专家候选、终点投影与缓存、分布审计。
- `tricoopdiff/metrics/`：完整原子 DockQ 和固定坐标系集合分析。
- `tricoopdiff/` 根目录：仅保留包入口、训练、采样、流、评估、运行支持和 DiffDock 兼容补丁，共 7 个 Python 文件；`diffdock/` 为 vendored 专家。
- `scripts/`：当前特征化、训练、评估与通用审计入口；model-v3 已移除退役运行 wrapper。
- `tests/`：断言式回归检查。
- `doc/`：主题文档；从[文档索引](doc/README.md)进入。
- `data/`、`checkpoints/`、`results/`、`logs/`：本地资产/产物。

训练实现统一在 `tricoopdiff/train.py`；双初始化的投影、缓存契约与在线生成统一在 `tricoopdiff/initialization/endpoints.py`；刚体变换与拟合统一在 `tricoopdiff/geometry/rigid.py`。已删除被合并的旧模块，仓库入口使用新路径，不保留转发空壳。

生产运输使用 `SDPIFlow`，不实例化旧扩散过程。BiTSeed 的二元专家扩散采样保留在 `tricoopdiff/legacy/diffusion.py`；SO(3) 数值表直接复用 vendored 专家实现，不再维护第二份缓存与预计算代码。

生产路径为 `models/consensus.py` → `sdpi_flow.py` → `train.py` / `sampler.py` → `evaluate.py`。共享文件发布和资产哈希集中在 `runtime.py`。三类训练器复用优化状态保存/恢复，但保持各自的 checkpoint 协议校验；不支持通过恢复权重静默更换监督目标。

历史 FiLM 和诊断实验保留用于复核已有结果，不由生产模型导入。评估的主指标仍是蛋白–蛋白完整原子 DockQ。集合多样性已统一为固定 E3 坐标系下的 RMSD，报告以 `fixed_e3_v1` 标记，不能与旧的对齐后多样性直接比较。

## 环境与检查

从仓库根目录运行；代码未作为 Python 包安装。仓库有 [environment.yml](environment.yml)，但它是环境导出，不保证不同 CUDA/驱动机器可直接复建。

```bash
python scripts/run_cpu_tests.py
python tests/test_model_build.py
python tests/test_training_contract.py
python tests/test_dual_init.py
bash scripts/gpu_status.sh
```

本项目在本地 GPU 环境直接运行，不使用 SLURM。模型任务需要冻结专家权重、ESM 缓存、数据与匹配的 checkpoint。

`run_cpu_tests.py` 在离线 CPU 子进程中逐个执行全部测试函数，不依赖 pytest；需要环境依赖和现有 DiffDock 数值缓存，不自动下载或生成缓存。

## 接手与贡献

先读 [todo.md](todo.md) 确认当前边界，再读 [Q.md](Q.md) 的已知限制。贡献约定见 [AGENTS.md](AGENTS.md)；模型版本、数据和运行参数以当前代码及对应实验 manifest 为准。

报告必须区分初始化 oracle、精修 oracle、初末态并集 oracle 和 native-free top-1；不要以测试标签选择候选或以小样本结果宣称泛化。
