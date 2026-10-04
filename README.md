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

历史 FiLM 编排和诊断实验保留用于复核已有结果，不由生产模型导入。v5 直接在原生专家 forward 中实施新的条件 FiLM，不复制历史 forward。评估的主指标仍是蛋白–蛋白完整原子 DockQ。集合多样性已统一为固定 E3 坐标系下的 RMSD，报告以 `fixed_e3_v1` 标记，不能与旧的对齐后多样性直接比较。

`model-v4` 分支的新默认架构为 `ternary_endpoint_v6`，checkpoint v30：两次冻结二元专家读取互以另一蛋白–配体当前结构为条件，全局软对应产生逐节点条件，并在专家最后三层对 even-scalar 通道实施有界 FiLM。距离、局部方向投影与有符号几何直接进入条件消息，伙伴特征相同也能传递构象变化。FiLM 输出宽度按专家实际 even-scalar 宽度配置（当前 48），与读取的全部标量特征宽度（58）分开。FiLM 零初始化保持原生专家输出；专家参数仍冻结，条件网络与三元网络联合训练，不是 LoRA，也不声称两个条件模型自动等价于完整联合分布。两分支在同一更新前状态读取条件，不串行改动坐标。条件目前是结构与界面表征，不是同步交换专家每层隐藏状态。

三元网络保留独立的完整刚体／torsion 输出；条件专家运动经单位及 POI 质心约定转换、有界化和可学习门控后作为附加先验，三元场不是二元输出的凸组合。v6 的三元上下文和融合门控直接读取专家运动强度、相互内积及其与当前质心方向的投影，共 14 个旋转不变量。torsion 在每条可旋转键上分别学习两个专家的权重，替代全局门控下的等权平均。v1–v5 按保存的配置保持历史行为，拒绝将旧权重静默解释成 v6。代码与反传验收不构成 DockQ 提升的证据。

v6 三元主干使用每个蛋白 16 个可学习软锚点和全部配体原子：连续的软最远点选择结合特征查询，使锚点覆盖不同空间区域；不使用会在近等距状态跳变的硬选点。完全对称且特征相同的输入仍遵守对称性，不人为选择世界坐标轴。主干持续维护带类型、几何及三角更新的成对关系；关系通过软对应回传到完整残基／原子节点，没有接触半径限制。E3–POI 关系拥有独立 token，但与两条蛋白–配体界面共享关系更新。该表示是资源受限的压缩，不等同于保存全部残基对的稠密特征，也不跨采样步缓存旧关系。配体节点消息直接读取四维键类型，torsion 仍读取实际键与片段相位。

v6 在全局锚点之外直接传递蛋白–蛋白局部残基对消息，20 Å 支持边界的权重及一阶导数连续归零；三角关系按 E3、POI、配体分别归一化后学习混合，不再由节点数量决定组分贡献。运动头同时学习节点贡献权重，平移和正则化惯性投影使用一致的加权读出。零初始化保持三元运动为零，条件专家先验可独立提供初始运动；两者在输出中分别记录。全局等变性来自实际坐标差、标量关系与刚体投影。退化刚体的不可观测旋转轴采用有界数值正则。v5（checkpoint v29）保留原来的特征锚点、条件及门控，已有隔离训练不受源码升级影响。

v6 训练采用 `endpoint_state_recovery_v1`：完整起点始终执行 20 步；辅助入口按八样本块交替采用近目标状态和真实 rollout 的中间状态。在 t≈0.75/0.5/0.25 的实际状态上施加合法的小幅刚体／torsion 扰动，断开入口之前的辅助梯度，仅执行剩余步数并监督真实终点。主路径仍完整反传，辅助权重为 0.25；验证只计算主路径。训练和推理共用同一个状态推进器。

生产 batch 使用变长图打包：每个时间步将多个样本的二元图送入两次原生专家 forward，FiLM、节点特征、torsion 按样本边界拆回；三元读出和刚体更新按各自拓扑执行。`batch_size>1` 在 v6 中不再是串行逐样本反传，批量平均 loss 只进行一次 backward/optimizer/EMA 更新；显存随 batch 增长。不同辅助时钟允许部分轨迹先结束。

上一代模型 `ternary_endpoint_v4`（checkpoint v28）及其后台隔离训练仍保留：两个冻结专家提供逐节点标量特征和运动证据，三元场通过全局 attention、连续局部几何和内部形状／全局方向基独立决定刚体运动；torsion 读出包含化学键、两侧片段和按邻居特征加权的一至三阶有符号相位，避免把几何对称误当成化学对称；近共线相位采用连续置信度。v4/v5 的 `continuous_horizontal_v1` torsion 更新保持真实键旋转，并以局部水平速度撤销平移／spin，不再对整步折叠构象求可能跳变的 Kabsch gauge。训练与采样共享这一一阶流更新器。旧 `ternary_endpoint_v1/v2/v3` 仍按原始参数、解码与行为构造，不允许静默升级加载；v2/25、v3/26 不作为新生产训练的续训来源。

主监督保持 `real_initialization_reachable_endpoint_v2`：从推理同款 ETKDG-SDPI 或原始 BiTSeed 起点执行完整20步，不把训练输入投影回晶体内部几何。配体监督目标投影到该起点的刚体＋torsion 可达空间；蛋白和蛋白界面仍监督真实终点。v3/v4 以八样本周期分开安排来源、零／小扰动和辅助入口时刻，分别从 t=1／0.2 执行20／4步，以0.25权重复用终点损失；平均增加12步。验证拒绝训练辅助入口，日志分开报告主终点 loss 与辅助 loss。目标投影是可达坐标替代指标，不等价于恢复精确晶体键长；投影未收敛时明确报错。当前 batch_size 是串行微批的平均梯度累积，不是真正图并行。

BiTSeed 生产训练按样本 occurrence 重采，验证固定 occurrence 0；缓存按 occurrence 分文件保存，RAM 每个 case 仅保留最新记录。缓存包含专家配置和实例数值策略，新协议不可复用旧缓存。重采会增加二元专家计算与磁盘占用，不再享有每 case 只生成一次的旧成本。

目标投影统一使用保留原子对应关系的 NumPy Kabsch 拟合，包括共线和单原子退化情形；torsion 用对应原子的有符号相位初始化并按周期角优化，避免零初始化和 ±π 人工边界制造错误局部解。目标缓存生成身份为 `raw_target_projection_v4`，拒绝复用旧拟合标签。checkpoint 恢复先验证权重、优化器动量、调度器、进度和 RNG，再加载实时状态；非法恢复不会留下半加载模型。v28 优化器状态按参数 ID 与完整名称校验，防止同形参数动量串线；旧归档无参数名时警告并依赖原始注册顺序，不承诺检测旧归档中的同形串线。

候选与流更新器要求 body frame 是合法 SO(3) 旋转，拒绝缩放、剪切和反射矩阵。共享旋转指数采用稳定半角 Rodrigues 表达，避免大角度矩阵指数引入缩放／剪切；有界方向编码避免大有限输入的范数溢出。可微刚体拟合 RMSD 在精确拟合零点保持有限梯度；验证和采样共用上下文恢复各子模块模式及 RNG。EMA 临时权重下禁止优化和保存/恢复训练状态，EMA 加载失败也会回滚原始权重。

专家 clamp 属于实例，不受其他 trainer/runner 覆盖；专家 sigma 配置必须匹配生产 DiffDock-L 预条件器。checkpoint 保存 Python、NumPy、Torch CPU/CUDA RNG 和专家运行身份；恢复要求相同 CUDA 拓扑。v25+ 保存训练 recipe 与 EMA 更新计数，v2/v3 EMA 按成功 optimizer 更新 warmup，验证默认使用 EMA 并明确权重口径。旧 v24/v1 的终点训练可按原架构、固定 EMA 和无辅助项续训；更早的历史监督不作为生产续训协议。历史速度实验显式使用基础速度训练器；历史条件训练仅读取已有投影归档，不再在线生成旧协议缓存。

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
