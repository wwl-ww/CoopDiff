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

新生产训练使用 v24 的 `real_initialization_reachable_endpoint_v2`：从推理同款 ETKDG-SDPI 或原始 BiTSeed 起点执行完整轨迹，不再把训练输入投影回晶体内部几何。配体监督目标投影到该起点的刚体＋torsion 可达空间；蛋白和蛋白界面仍监督真实终点。目标投影是可达坐标替代指标，不等价于恢复精确晶体键长；投影未收敛时明确报错。

BiTSeed 生产训练按样本 occurrence 重采，验证固定 occurrence 0；缓存按 occurrence 分文件保存，RAM 每个 case 仅保留最新记录。缓存包含专家配置和实例数值策略，新协议不可复用旧缓存。重采会增加二元专家计算与磁盘占用，不再享有每 case 只生成一次的旧成本。

目标投影统一使用保留原子对应关系的 NumPy Kabsch 拟合，包括共线和单原子退化情形；目标缓存生成身份为 `raw_target_projection_v3`，拒绝复用旧拟合标签。checkpoint 恢复先验证权重、优化器动量、调度器、进度和 RNG，再加载实时状态；非法恢复不会留下半加载模型。

候选与流更新器要求 body frame 是合法 SO(3) 旋转，拒绝缩放、剪切和反射矩阵。可微刚体拟合 RMSD 在精确拟合零点保持有限梯度；验证和采样共用上下文恢复各子模块模式及 RNG。EMA 临时权重下禁止优化和保存/恢复训练状态，EMA 加载失败也会回滚原始权重。

专家 clamp 属于实例，不受其他 trainer/runner 覆盖；专家 sigma 配置必须匹配生产 DiffDock-L 预条件器。v24 checkpoint 保存 Python、NumPy、Torch CPU/CUDA RNG 和专家运行身份；恢复要求相同 CUDA 拓扑。生产训练只保留此终点协议；旧 checkpoint 仅用于推理或作为新训练的权重来源，不再支持恢复旧生产训练。历史速度实验显式使用基础速度训练器；历史条件训练仅读取已有投影归档，不再在线生成旧协议缓存。

采样保留调用者 RNG。请求 rank features 时会额外读取最终构象，使用 `t=0, dt=0`，标记为 `final_pose_t0_dt0_v1`；不能与旧的最后一步更新前特征混用。

训练验证采用 `case_fixed_real_initialization_protein_fullatom_v4`：SDPI 起点按 case 固定，不随遍历顺序或子集变化；验证与特征处理保留所有已初始化 GPU 的 RNG。恢复不同验证协议时，保留权重和优化器/LR，但重置最佳 loss、DockQ 和调度器的旧验证历史。完整 DockQ 续跑必须有匹配的逐 walker 完整评分计数。

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
