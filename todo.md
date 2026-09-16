# 当前工作与交接

更新：2026-09-15。这里仅记录执行状态，不重复协议或历史流水账。

## model-v3 起点与清理范围

本分支从已推送的 `master` 合并提交 `92cf1f0` 创建；清理已提交为 `75a16e2`。
正在实现独立质量排序器与 PROTAC 条件资产审计；尚未训练新的 model-v3 精修模型。
第一步只清理退役入口，不改模型或实验数据：

- 移除旧 GPU3 夜间执行/接管、factorial/rollout 启动与固定路径 `local_*` 诊断链。
- 移除已由 lazy C1M1 替代的全池训练端点缓存生成/合并/启动脚本，以及5个 SLURM wrapper、旧 Phase1 readiness 和一次性交接/rigid-oracle 汇总。
- 删除只测试上述退役流程的测试；旧 v2 测试里的验证集整簇排除逻辑迁到 `prepare_validation_index.py` / `test_validation_index.py`。它只写新索引，不自动修订现有训练集。
- 保留标准训练、B/C 启动、全池精修、时间步审计、二元基线、数据处理、ranker 原型及核心回归。保留库内 factorial checkpoint 加载与几何/轨迹不变量测试，避免破坏旧基线复评。
- 全量 CPU 检查统一为 `python scripts/run_cpu_tests.py`；默认不启动任何 GPU 作业。

清理验收：移除36个退役脚本/配置与4个专用测试，另迁移1个 v2 测试文件并保留其数据划分检查。
剩余44个测试文件共228项 CPU 检查全部通过；130个一方 Python 文件语法与删除入口引用扫描通过，
保留的本地 shell 启动器语法检查通过。模型库、数据、权重和历史结果未修改，未启动 GPU 实验。

历史文件均可从 `92cf1f0` 恢复。完整清理前源码可在独立 worktree 查看，例如
`git worktree add --detach /mnt/disk/wl/MoRE-tri/CoopDiff-v2-review 92cf1f0`（目标需不存在）。
Git 不携带被忽略的数据/权重；复跑旧实验仍需原 manifest 和本地资产。

## model-v3 第一批：R1 数据链与 P0 条件审计（2026-09-15）

用户授权共享物理 GPU5、7，每卡一个 worker，不停止其他任务。后台 tmux 会话
`model_v3_ranking`，输出根目录 `results/model_v3_ranking_20260915/`。
运行代码已封存在该目录的 `runtime_v1/`，后续修改工作区不影响本批任务。

| 阶段 | 配置与边界 |
| --- | --- |
| 固定生成器 | C40 EMA；192 BiTSeed +64 SDPI，C=4、M=8，二元20步、补采上限2、连续失败上限3；精修20步，seed31000 |
| 训练采集 | 内部训练集16个组各取1例；每例按实际来源随机平衡抽32条，来源不足时回填并记录纳入概率；共512条 |
| 验证采集 | 内部验证集4个组各取1例；全池256条，不先按质量筛选；共1024条 |
| 独立评分 | 采集入口只接收蛋白、配体和分组信息；独立 CPU 进程加载 native，计算初末态完整原子 DockQ |
| 排序对照 | 采集和评分完整通过后，各卡训练一个100轮小型质量头：仅几何 vs 冻结特征＋几何；不更新 C40 或 DiffDock |

两卡各串行完成自己的训练/验证分片；case 失败记录后继续后项。任何缺失或评分失败都会阻止自动训练排序器，
不把剩余成功例冒充完整样本。查看 `pipeline_status.json`、`train_archives/worker_*.json`、
`val_archives/worker_*.json` 和 `*_gpu5.log` / `*_gpu7.log`；关闭终端不影响后台运行。

数据边界：原训练/验证索引有83个相同 PDB 重叠，本批验证剔除全部原训练 PDB，且两边均剔除官方
PROTAC/MGD benchmark PDB。组内按 case 名取首例，再按固定 seed 抽组，不依据 DockQ 选例。
这仍只是16/4组的流程验证：验证数据曾参与 C40 模型选择，也未保证蛋白家族/配体独立，不能声称 untouched holdout。
索引、排除名单与源文件指纹在 `indexes/audit.json`。

实现入口：`prepare_ranking_smoke.py` → `collect_ranking_archive.py` →
`score_ranking_archive.py` → `train_state_ranker.py`，后台编排为 `run_ranking_pipeline.py`。
初末态探针均在对应已保存坐标上以 t=0.05、dt=0 重新前向，并恢复随机状态；不复用旧采样器最后一步更新前的特征。
当前质量头使用连续 DockQ 与有序0.23/0.49阈值监督，报告 Initial / Final / Union 的模型 top1、top5召回和 oracle；
尚未实现未来价值 pre-ranker、跨 walker 交互或可用于正式 benchmark 的冻结排序器。

验收：两卡共2例、4条20步烟测及独立评分通过；第一轮全量 CPU 回归47个文件236项通过。
增补标签隔离/缺失末态检查后，全量为237通过、1失败（仅单状态时 listwise loss 不适用）；
已修复工作区质量损失的单状态边界，受影响的7项排序测试定向重跑全部通过，合计238项验收。
该边界修复未改写运行中的快照；当前每例至少32个初态，不触发单状态路径。
训练16/16例、验证4/4例均已完成采集，分别512/512与1024/1024条精修有效；
独立评分、两种排序头训练及CPU比较全部成功结束，`pipeline_status.json` 为 `complete`。
初步排序效果不理想，完整数值和适用边界见下节；原采集/训练GPU任务已结束。
首次烟测遇到 vendored 导入改变 CWD，已修正并用新快照重跑通过；旧日志保留。

P0 初审输出 `protac_condition_audit.json`：22例均完成审计，但严格化学子图映射未得到唯一匹配，
可直接用于条件推理的 case 为0。PDB键级信息、封端/片段化学和二元结构来源仍需核实；这不表示22例生物学上无匹配。
当前只产出局部口袋和映射诊断，不读取 ternary native，不把未经来源验证的二元资产接入生成器。

### 继续实现：独立推理、R0/R1 汇总与映射诊断

- 新增 `rank_pose_archive.py`：在已保存的配对初末态上输出 native-free top1/top5，
  不把未精修、未提取特征的全池初态偷偷加入 Union。无权重时使用预先固定的 R0 Cα 规则：
  `min(双侧6Å接触比例) − 双侧2Å冲突比例之和`；这是新的便宜诊断基线，**不是历史 Geometry-64**，也不是全原子物理评分。
- 学习排序按质量 logit 降序；并列按 pool index、初态优先确定性打破，不使用标签。
  推理权重先导出为仅含权重、训练归一化和生成器合同的 `inference.pt`；排序进程不接受含验证指标的训练 `best.pt`。
  训练/推理共享特征白名单，不使用来源、阶段、候选ID或真实DockQ作显式模型输入。
- 新增 `audit_state_ranking.py`：预测落盘后才加载独立标签，校验 archive hash 和状态覆盖。
  输出 all/BiTSeed/SDPI × Initial/Final/Union 九组统计，包括模型 top1、top5-best、oracle、regret、
  逐例 retention、≥0.23/0.49 命中和 case-macro Brier。缺失来源或末态保留在 case 分母；
  oracle=0 的 retention 记未定义并单独计数。随机对照使用同一状态集合均匀无放回抽样的解析期望，非新增轨迹。
- `compare_state_rankers.py` 在上游完整成功后，以纯 CPU 串行比较固定 R0、仅几何 MLP、冻结表示＋几何 MLP。
  后台会话 `model_v3_rank_audit`，封存代码 `runtime_audit_v1/`，日志 `ranking_audit.log`；最多等待12小时，
  上游失败则退出，不用成功子集出表。产物 `validation_ranking_audit/comparison.{json,md}`。
  不新增 GPU worker、不修改正在运行的 `runtime_v1/`。这仍是内部4组全池 post-ranking 验证，
  **不是新的256→64 selector消融，也不是独立holdout成绩**。
- 验收：真实2例烟测档案已完成 R0 推理与独立审计（`smoke_r0_ranks/`、`smoke_r0_audit.json`），
  仅检查输出链，不据此宣称效果。全量 CPU 回归48文件245项通过；追加完整三方法 CLI 对照测试后，
  受影响14项定向重跑通过，当前共246项验收。测试覆盖无结构/native/标签文件的独立推理、权重导出、
  标签隔离、失败末态、缺失来源、零oracle、随机top5解析概率和生成器身份校验。

PROTAC 增量报告 `protac_condition_diagnostics_v2.json`：44侧中29侧能匹配元素连接关系，
但忽略立体化学后仍为0侧严格化学匹配；其余15侧连元素连接关系也未匹配。
例如5T35_D_A_759的两侧片段在PDB读取后全是单键，而完整SDF含芳香键和双键。
下一步先核实片段的可信化学模板、封端及二元来源，不能把宽松图匹配当作有效原子映射；
`relaxed_mapping_usable_for_inference` 与条件推理就绪标记仍全部为 false。

### 继续实现：CCD 化学恢复与独立条件编码器

新增 `scripts/prepare_protac_binary_chemistry.py`，复用已有 `restore_ligand_chemistry.py`：
按 CCD 原子名称/别名恢复键级和电荷，保持原 PDB 的重原子坐标及顺序；不补缺失原子、不自动删封端。
程序只接受显式的离线 `--ccd-dir`，不在运行时联网。2026-09-15 单独获取了 RCSB 的9个公开 CCD 文件：
3JF、0S6、LVY、JQ1、FX5、TKY、8E8、3CQ、35Q，存于 `ccd_binary_components/`。
来源说明见 [wwPDB CCD](https://www.wwpdb.org/data/ccd)，下载路径为
`https://files.rcsb.org/ligands/download/<component>.cif`；实际文件 SHA256 在恢复报告中。

当前产物 `binary_chemistry_v2/report.json`（以上相对路径均基于本批结果根目录）：

- 22例44侧，34侧完成化学恢复且坐标不变，10侧因 CCD 重原子缺失而拒绝完整组分恢复。
- 恢复的34侧中15侧可严格化学匹配，但各有12个对称映射；这些映射落在同一个原子集合，
  因而片段角色集合已明确，**逐原子坐标对应仍不能当作唯一解**。其余19侧仍未严格匹配。
- 多解及截断标记保留；只有枚举未截断且所有匹配的原子集合相同时，才报告 `unambiguous_role_atoms`。
- CCD化学恢复不证明二元结构来源，`ready_for_conditioned_inference` 仍为0；未用这些资产重跑或改动旧C40实验。

新增 `tricoopdiff/condition_features.py` 的 `SideCondition` / `build_condition_features` /
`ProtacConditionEncoder`；最初作为独立输入适配与可训练编码器验收，现已接入下述实验性融合类。
生产 sampler、训练入口和 C40 checkpoint 仍未启用这些通道。

| 输入 | 当前合同 |
| --- | --- |
| 片段/角色 | 显式化学图及立体匹配的原子映射；E3端、POI端、linker、未知；linker必须显式注释，不把所有剩余原子都算作linker |
| 口袋 | 已对齐当前受体节点顺序的概率及可用性；未知口袋与已知全负口袋分开 |
| 二元几何 | 两侧各自的蛋白/片段坐标；当前只编码侧内距离与可用性，不利用任意跨侧绝对布局 |
| 来源/可靠度 | provenance/evidence ID保留在manifest审计，confidence进入模型；不复用候选source的0/1；拒绝native/unverified部署类型。通用训练无法校准多类provenance，v2不把其one-hot送入模型 |

支持单侧/无条件、通用三元配体的双侧角色重叠；PROTAC模式拒绝端点重叠。无条件编码输出严格为零；
新增的C40 EMA合成输入迁移回归见下文。当前二元距离摘要不等于完整known-binary姿态约束：
后续仍需实现坐标对应的局部frame对齐、受约束初始化、条件训练和生产checkpoint协议。

针对条件、对称映射和CCD恢复的15项检查通过；本轮全量 CPU 回归49个文件253项全部通过，`git diff --check`通过。
下一步：核实10侧裁剪/封端定义、19侧完整片段与PROTAC核心的关系，以及真实二元来源；
独立验证通过后再接入新的条件精修模型，不把尚未训练的输入通道直接塞给C40。

### 继续实现：条件融合原型与 C40 EMA 迁移烟测

新增 `tricoopdiff/conditional_consensus.py` 的 `ConditionedConsensusFuse` 和
`ConditionedConsensusCoopDiff`。架构单列为 `conditioned_residual_consensus_v3_prototype`；
旧版本18/19/20及生产加载器未升级，不把原型冒充已训练的v3精修checkpoint。

- 条件残差加入 `TernaryGeometryEncoder` **类别编码之后**的节点表示，并影响协商context和运动gate；
  不改原始化学类别、不调制DiffDock专家输入或输出。
- 无条件走旧计算路径；显式全缺失条件产生零残差。新增参数使用独立随机状态初始化，
  迁移保留旧共同权重。条件专属gate支路非零初始化，避免旧零初始化运动头阻断编码器首步梯度。
- 旧融合类拒绝条件输入，旧模型wrapper拒绝条件融合类；实验模型通过 `conditions=` 显式传入已校验的特征。
  节点数量/顺序须与当前受体和ligand一致；当前检查形状，序列/原子索引来源对齐仍由输入构建负责。
- 原型导出使用 `conditioned_fuse_prototype_v1`，严格记录父checkpoint SHA256与条件合同；
  不携带专家权重，也不能直接由现有生产训练/评估入口加载。

新增6项回归及原11项Consensus检查通过，覆盖无条件精确回归、参数/RNG迁移、首步编码器梯度、
后续节点投影梯度、非零残差下SE(3)等变、source独立性、专家输入/梯度隔离和原型权重往返。

实际烟测入口：`python scripts/smoke_conditioned_fuse.py --checkpoint <C40.pt> --out <新目录>`（CPU-only）。
产物 `results/model_v3_ranking_20260915/conditioned_c40_cpu_smoke/`：严格加载真实C40 EMA，
使用合成图/二元建议，在source=0/1下分别验证无条件速度与rank_features精确一致，条件速度变化及编码器梯度非零。
没有optimizer更新；`prototype_untrained.pt` 是**未训练的融合模块原型**，不作DockQ性能或真实条件精修效果结论。
该烟测期间GPU5/7执行原封存的R1采集；未切换到条件原型，现原R1任务已结束。
本轮全量 CPU 回归50个文件259项全部通过，`git diff --check`通过。

### 继续实现：条件缺失、扰动和类别覆盖审计

新增 `tricoopdiff/condition_augmentation.py`，默认所有概率/噪声为0，即恒等变换，
**不是已冻结的训练增强配比**。可显式指定整例/单侧/口袋/二元姿态缺失，以及口袋概率噪声、
二元片段的局部刚体平移/旋转扰动；返回seed、参数和事件记录用于复放。

- 整例丢弃同时去掉linker角色，确保真的回到无条件；丢弃口袋记为unknown，而非已知全负口袋。
- 片段刚体扰动不改变内部距离、原子对应或受体坐标；局部轴从本侧蛋白几何构造，
  同seed下两侧frame独立变换仍满足协变性。共线/退化蛋白frame明确报错，不回退到任意世界坐标轴。
- 不消耗全局Python/NumPy/Torch随机状态，左右侧随机决策独立；不修改输入对象、采样状态、
  原速度监督或候选source，因此没有偷偷改变flow路径后继续使用旧导数的问题。
- 扰动后的条件带synthetic标记，可显式降低confidence；目前只提供机制，尚未证明模型能识别错误条件或自动降权。

`scripts/smoke_condition_robustness.py` 在真实C40 EMA上执行12种条件状态×2种source的CPU合成压力检查。
产物 `results/model_v3_ranking_20260915/condition_robustness_cpu.json`：24项均有限；
无条件输出精确保持原样、有条件梯度非零，optimizer更新数为0。它证明数值/接线合同，不是学到鲁棒性或DockQ提升。

另新增 `scripts/audit_ranking_label_balance.py`，要求完整档案与标签匹配后才输出分来源、分阶段的类别覆盖，
同时报告state数量和贡献正例的case数量。后续新运行的排序训练报告也会包含类别覆盖与缺失类别提醒。
完整评分：16个训练case共1024个初末态，仅6个≥0.23（来自3例）、0个≥0.49；
4个验证case共2048个初末态，仅6个≥0.23（来自2例）、0个≥0.49。
因此当前16/4组应继续定位为流程pilot，不凭低loss、低Brier或大量相关姿态宣称训练数据充分。
该提醒不改变已封存本批候选/训练配置，不按已见标签挑换case。

验收：新增6项增强测试通过；全量CPU回归51文件265项通过。
随后追加类别case计数检查及训练报告检查，受影响的12项排序测试定向通过，累计266项验收。

### 首轮 R0/R1 完整结果与后续诊断

以下是**内部4例、每例256条配对轨迹的后排序**，不是PROTAC22成绩，也不是256→64筛选消融。
Initial / Final 分别在初态/末态中选top1，Union在512个初末态中统一选top1，再跨case取均值。

| 方法 | Initial top1 | Final top1 | Union top1 | ≥0.23 初/末/Union |
| --- | ---: | ---: | ---: | --- |
| 固定R0 Cα规则 | 0.0736 | 0.0193 | 0.0182 | 1/4、0/4、0/4 |
| 仅几何MLP | 0.0253 | 0.0369 | 0.0386 | 0/4、0/4、0/4 |
| 冻结表示＋几何MLP | 0.0261 | 0.0305 | 0.0305 | 0/4、0/4、0/4 |
| 均匀随机top1解析期望 | 0.0233 | 0.0287 | 0.0260 | 非实际抽样命中数 |
| 全池oracle（使用真值） | 0.2852 | 0.2549 | 0.2852 | 2/4、2/4、2/4 |

两种MLP的top5也均未命中≥0.23；所有方法及oracle均无≥0.49。几何MLP/冻结表示MLP分别由
内部validation composite loss选中epoch4/3。冻结表示在本轮未优于仅几何；不能据4例判定普遍优劣。
这轮证明采集—独立评分—训练—native-free推理链能工作，**尚未得到有用的可部署ranker**。
训练/验证共3072个初末态的128维冻结探针均有效，无探针失败补零，不能将本轮差表现归因于该缺失路径。
标签稀缺是明确限制，但尚不能认定它是唯一原因，需区分训练拟合、特征质量与泛化问题。

证据均位于 `results/model_v3_ranking_20260915/`：
`validation_ranking_audit/comparison.{json,md}`、`train_label_support.json`、`val_label_support.json`。
已完成训练曲线诊断：原16/4划分、原特征/损失/seed、100轮不变，额外记录每轮train/val榜单和
最后一轮表现；仍按原validation loss选模型，不用最后一轮或训练集成绩重新挑模型，不改候选池。
暂不把现有pilot权重接入正式selector，也不依据已见DockQ筛选扩大训练集。

产物为 `fit_diagnostics_{geometry,frozen_geometry}/`，代码封存在 `runtime_fit_diagnostics_v1/`；
在GPU5/7各一个后台worker执行，均已结束。两个best checkpoint的state_dict与首轮逐张量完全一致。

| 特征 | 第100轮 Train Union top1 | Train ≥0.23 | 第100轮 Val Union top1 | Val ≥0.23 |
| --- | ---: | ---: | ---: | ---: |
| 仅几何 | 0.0923 | 2/16 | 0.0291 | 0/4 |
| 冻结表示＋几何 | 0.1009 | 3/16 | 0.0166 | 0/4 |
| 各自集合oracle | 0.1157 | 3/16 | 0.2852 | 2/4 |

两头的训练loss持续下降，而验证loss在epoch4/3之后总体变差；不是简单的“没有梯度/完全学不会”。
本批更支持拟合与泛化脱节，不能用增加epoch替代新增训练case与质量覆盖。
训练只抽32条、验证全池256条，oracle预算不同；listwise交叉熵也随集合大小变化，
**不能直接比较train/val loss绝对值**，应看各自随epoch变化和相同评估集合上的top1。
下一批应事先固定不依赖DockQ的抽组规则与独立验证划分；正例增强若引入native扰动，只能用于
训练数据构建并明确标注synthetic，不能混进推理候选或用benchmark真值做训练。

本轮验收：新增训练诊断不改变权重/RNG的检查，全量51文件267项通过。
随后修复条件模块CPU隔离区调用全局 `torch.manual_seed` 会重置CUDA随机流的问题：
改为仅设置CPU默认generator，合成烟测fixture同步修复；新增禁止CUDA重设种子的回归，
条件模块7项定向重跑通过，累计268项验收，`git diff --check`通过。已封存runtime不改写。

### 第二批：固定抽组扩展32/8（2026-09-15）

沿用用户授权共享物理GPU5/7、每卡一个worker。后台会话 `model_v3_ranking_expand32`，
结果目录 `results/model_v3_ranking_expand32_20260915/`，入口 `launch.sh`，代码封存为 `runtime_v1/`。

| 项目 | 固定配置 |
| --- | --- |
| 训练 | 32组各1例，每例64条，实际来源随机平衡抽样并记录纳入概率；2048条 |
| 内部验证 | 8组各1例，每例完整256条；2048条 |
| 生成/精修 | 原C40 EMA、192/64池、C4 M8、二元20步、精修20步、proposal seed31000；不改冻结专家 |
| 抽组 | 原seed7315、整簇排除benchmark、验证排除全部原训练PDB；32/8分别是原固定序列的较长前缀 |
| 后处理 | 完整独立评分 → 类别/身份审计 → 两种100轮质量头及训练曲线 → CPU R0/R1九组榜单 |

原16/4例与本批索引前缀已逐项一致性检查；**不是新独立holdout**。增加训练样本数与每例轨迹数，
同时扩验证面板，因此不能直接把两批总体均值差归因于某个单因素。若做旧/新ranker比较，应在同一
封存评估集合上比较，先核对生成器合同；旧runtime哈希与新runtime不同，不能强行绕过检查加载。
本批不依据已见DockQ替换低质量case；初末态仍相关，增加state数量不能替代独立case覆盖。
为保持整批档案manifest一致，全部重新采集，不拼接修改旧档案；旧结果完整保留。

新增预算参数 `run_ranking_pipeline.py --train-walkers 64 --val-walkers 256`，case数从实际索引推导，
启动前核对schema、split、唯一case、PDB交集及抽组审计，状态记录计划4096条，不再硬编码16/4。
评分成功后额外运行 `audit_ranking_label_balance.py`；任何采集、评分或身份审计失败都阻止自动拟合。
训练入口现在也独立核对label manifest、完成回执、内嵌manifest、case/split及精确状态覆盖，
拒绝重复标签，不仅依赖编排器保证正确。首批20例通过新版严格加载；受影响15项定向检查通过。

启动时两卡均有其他作业，未停止任何他人进程。初步以首批墙钟按轨迹数线性放大约3.4小时作
粗略参考，case大小/二元生成开销/共卡负载均会改变吞吐，不能当截止承诺；以worker进度为准。
查看 `pipeline_status.json`、`train_archives/worker_{0,1}.json`、`pipeline.log`。

另新增 `scripts/audit_ranking_references.py --expect-native-frame`：只对从bound native拆出的内部
输入做评分正对照，不适用于unbound或独立二元姿态，不产生候选/训练标签。40例原始构象重建均得到
完整原子DockQ=1.0；规范化路径参数后复跑报告为本批目录的 `reference_metric_control_v2.json`，
带输入和评分代码指纹。这只能排除原始模板重建的明显
错配，不能替代采样坐标变换、化学有效性或泛化验证，也不能将这些满分构象加入验证候选池。

本轮全量回归52文件271项通过；之后补评分正对照失败/非有限值测试，编排与正对照4项定向通过，
累计272项验收。中途测试代码作用域错误已修复并包含在完整重跑中；生产后台使用封存代码未受影响。
启动验收时两卡各完成首例64条；`2H61_D_E_PG4` 的256个池初态hash与首批完全一致，
32条共同抽中候选的末态hash也全部一致。这是单例生成回归证据，不代替本批最终完整性/收益审计。

### 已排队：约10小时夜间串行任务（2026-09-15）

用户明确要求安排约10小时后台串行任务。继续只共享GPU5/7，每卡最多一个本项目GPU worker，
不占用0–4/6、不停止其他人的作业。后台会话 `model_v3_ranking_night160` 已启动，
根目录 `results/model_v3_ranking_night160_20260915/`，入口 `launch.sh`，状态 `queue_status.json`。

| 顺序 | 工作 | 固定预算/复用 |
| --- | --- | --- |
| 1 | 等待当前expand32全部完成，包括其最终R0/R1报告 | 不另起GPU worker；上游失败则停止依赖队列 |
| 2 | 新增固定序列第33–160个训练case，128例×64条 | 8192条新精修；原192/64池、C40 EMA、20步、原seed |
| 3 | 两个CPU评分分片＋完整标签/类别审计 | 标签与生成进程隔离；任何缺失阻止后续训练 |
| 4 | 32/80/160例训练量×两种特征×三个seed | 18个质量头；100轮、seed7315/7316/7317，不训练精修器或DiffDock |
| 5 | 同一8例验证集，native-free排序后再独立审计 | 三榜、all/BiTSeed/SDPI九组、≥0.23/0.49、top5、oracle、校准与训练拟合诊断 |

复用expand32的32例训练档案和全部8例验证档案/标签，不重跑旧轨迹，不拼改任何原manifest。
新增采集**仍调用expand32封存的生成runtime**，确保生成器指纹相同；夜间新runtime只负责
队列、分片评分、排序训练/审计。多档案训练核对生成器、采样协议、PDB唯一性、标签完整性，
再按已声明的case顺序取32/80/160前缀；归一化只由该前缀训练数据计算。
新增名单来自原seed7315序列，不读取DockQ选例；8例验证是已反复使用的内部验证，不是新holdout。

本实验固定epoch，不固定optimizer updates：分别3200/8000/16000次更新，报告显式记录。
因此是数据与相应计算量的学习曲线，不能单独宣称纯数据量的因果收益。
三个seed全部保留，汇总均值与样本标准差/命中范围，不用验证结果挑最好的seed代替整体表现。

总预算以当前任务剩余约3小时＋新增采集/评分约6–7小时＋质量头/审计约20–40分钟估计，
**目标约10小时，不是硬截止或满卡承诺**；共卡负载、case大小和失败会改变时间。
不在10小时强杀进行中的case。等待上游最多12小时，超时记录失败且不启动新的GPU工作。
新增case采集失败后仍扫后续case，但不使用成功子集冒充160例完整训练；队列不会自动重启或扩大预算。

明早主要读取：`summary.md`、`summary.json`、`extra_train_label_support.json`，以及
`rankers/{geometry,frozen_geometry}_n{32,80,160}_s{7315,7316,7317}/report.json`、`audit.json`。
只有18项全部成功才产出总表；中途可查看`queue.log`、`extra_train_archives/worker_{0,1}.json`和逐项日志。
验收：多档案/前缀归一化、队列入场/上游失败测试通过；真实2例4轨迹双分片评分与旧标签逐项完全一致。
最终全量CPU回归53文件276项全部通过，`git diff --check`通过。队列验收状态为
`waiting_upstream`，不与expand32增加并发GPU worker；夜间代码与启动文件已封存，关闭终端不影响运行。

## 本分支已完成

- [x] 在线 BiTSeed、短补采、部分成功保留和 SDPI 缺额补位。
- [x] 确定性 256→32/64 选择：几何排序、固定 E3 坐标系去重、来源保护与回填。
- [x] PROTAC22 固定 256 候选的比例 oracle 消融。
- [x] MGD94 全池 24,064 条精修；同一轨迹用于各 selector 的反事实比较。
- [x] 分来源统计与 Initial / Final / Union 三榜；结论见[协议](doc/BiTSeed-Hybrid在线初始化规格.md)。
- [x] `scripts/gpu_status.sh` 一次查看全部 GPU。
- [x] 合并旧架构、公式、初始化及运行交接，删除过期重复文档。
- [x] `exp/dual-init` 的双来源训练、source条件、缓存与冻结专家实现已合入（`6e26621`）。
- [x] B40/C40及C81 best各完成PROTAC22全池评估，三组均22例、5632条有效轨迹。

## 当前决定

候选池沿用 192 BiTSeed + 64 SDPI；不做 DiffDock LoRA。C组为后续主线，当前PROTAC精修工作基线保留C40 EMA；C81是训练validation-loss最佳，而非已证明的下游最优。Geometry、Diversity、Hybrid-48/16和Hybrid-56/8继续对照，尚不冻结selector；旧MGD factorial模型下对56/8的负面判断不外推到新C组。

下一步设计见[Ranking与PROTAC条件调研](RANKING_PROTAC_RESEARCH.md)。下面是后续工作清单，不构成自动启动 GPU 作业或重新训练的授权。

## 待办

### 已完成：B/C epoch40 推理对照（历史启动配置）

用户最终指定 B、C 均固定 `checkpoint_epoch40.pt`，不用各自的 best epoch；主评估统一加载 `ema_trifuse_state_dict`（decay=0.999）。权重在独立工作树的 `results/dual_init/arm_{B,C}_online_c1m1_seed73/`；复制/封存时记录 SHA256，不覆盖训练目录。1:1混合并不意味着40轮与旧模型20轮等计算预算，需同时报告更新次数。

原主计划为 MGD94；2026-09-14 用户授权先将 PROTAC22 unbound 全量推理与现有训练共享 GPU5/7 后台运行。协议为192 BiTSeed +64 SDPI、seed31000、20步精修；保留 Geometry-64、Diversity-64、Hybrid-48/16、Hybrid-56/8 与 SDPI-64，即两个新模型×五个选择方案共10组，对实际 BiTSeed/SDPI 来源分别报告 Initial/Final/Union。全池反事实审计每个模型精修全部256候选，共22×256=5632条轨迹；两个全池作业即可汇总10组，不是10次重复精修。尚未启动MGD全量或新的旧模型基线。只有输入、候选坐标/ID、数值协议与评分合同一致时才能复用旧结果，否则需另行安排配对基线。

C 按实际 candidate source 输入0=SDPI、1=BiTSeed；fallback候选为0，标志在轨迹中不变。B不使用来源条件。新权重是v20，必须使用支持该版本的联合训练分支接口，不可将其版本号改成旧格式或丢弃source参数。

准备检查（2026-09-14）：此前CPU检查通过B50/C40 EMA严格加载、合成输入前向有限、来源条件与fallback路由。最终B40/C40烟测按用户授权与训练共享GPU5/7，CPU线程限制为2、每组使用私有特征缓存；不停止训练、不占用AEGIS的卡。烟测使用6BN7完整256池，各取第一个真实BiTSeed与SDPI候选做20步精修，逐次model调用核对来源并记录候选hash。入口为 `scripts/evaluate_selected_refinement.py --runtime-root <dual-init-worktree> --smoke-one-per-source`；日志/报告位于 `results/dual_init_refinement_smoke_20260914/{b40_gpu5_v1,c40_gpu7_v1}/`。烟测不是64-walker成绩。

全量启动记录：输出根目录 `results/protac22_dual_init_e40_20260914/`，GPU5为`b40_gpu5/`、GPU7为`c40_gpu7/`；tmux会话分别为`protac22_b40_20260914`、`protac22_c40_20260914`，关闭终端不影响运行。代码与epoch40权重已复制封存，特征缓存各自独立；启动检查补齐了快照缺失的DiffDock数值缓存，重启的仅为这两个新评估进程。C=4、M=8、binary_steps=20、retry_limit=2、consecutive_failure_limit=3；启用`--refine-full-pool`。每组以`manifest.json`记录协议和源文件hash，`progress.json`每16条精修更新，逐case落盘，结束生成`summary.json`及`failures.json`。三榜均为case-best oracle，不是可部署top1。共卡预留约8–12小时，实际以完成22例及有效轨迹核验为准，不保证截止时间。

### 已完成：C best 配对复评（2026-09-15）

B40/C40 PROTAC22 全池评估均已完成22/22例、5632/5632条有效轨迹，无case失败；分别耗时6.70/7.09小时，22池及5632个初态hash、精修seed一致。C40五组最终均值均高于B40，但并非逐例多数胜出，不能据此断言来源条件的普遍因果优势。

用户选择C组作为后续主线，并授权原版（非LoRA）C best复评。封存权重为epoch81、global_step47547、best_val_loss3.1384998601、source_conditioning=True、EMA decay0.999；SHA256为`e637d8a7d8569f57a19aa136d99bc9e943ebbc82df88a385d03f6aa5b670a5be`。输出`results/protac22_cbest_e81_20260915/`，复跑入口`launch.sh`；历史运行使用GPU5和同名tmux会话，现已结束。复用C40封存runtime及相同PROTAC22 index、seed和全部生成/精修配置，使用复制的私有特征缓存；22×256条20步精修汇总五种selector和三榜。不得混用另一边的LoRA训练权重；完成情况及配对核对见下文。

上述启动说明为历史记录：C81现已22/22例完成，5632条有效，无case失败，耗时7.13小时；与C40的22个池、5632个初态及seed均核对一致。五组Final均值C40更高，部分64候选Union均值C81更高。完整对照见[研究文档第2节](RANKING_PROTAC_RESEARCH.md)，不再将C81作业描述为仍在运行。

### 后续验收

本轮GPU烟测已通过：B40/C40在6BN7上均完成两来源各20步，4/4轨迹有限、无记录失败；两个256候选池hash及选中初态hash完全一致，实际来源均为192/64。模型调用的source trace分别为20个1（BiTSeed）和20个0（SDPI）。含候选生成的case耗时B=132.29秒、C=130.25秒，单条20步精修约3.49–3.55秒（不含事后评分）；这是单例烟测，不作为准确率比较或全PROTAC吞吐保证。训练进程保持运行。启动阶段发现并修复了runtime导入切换CWD造成的相对路径错误，旧报错保留在同一run.log中，最终结果以完成JSON和空failures.json为准。

- [x] `exp/dual-init` 已交付共享模型、训练配置、checkpoint身份和双来源验证结果；代码已合入，权重不随Git迁移。
- [ ] 在内部验证集比较旧模型、无来源条件的混合训练、有来源条件的混合训练；使用相同初态和预算。
- [ ] 新权重下重新审计三种 selector 的最终 retention、来源收益与 SDPI 能力保持。
- [ ] 分别检查初态、末态的全原子几何；验证好构象召回后，才决定哪些规则可 hard reject。
- [ ] 在内部验证集冻结 selector；MGD94 仅作已使用过的开发评估，不冒充未触碰 holdout。
- [ ] 部署 top-1 仍需独立验证的 native-free 选择规则；Union oracle 不替代排序器。

## 联合训练分支边界

分支：`exp/dual-init`。工作树：`/mnt/disk/wl/MoRE-tri/CoopDiff-bitseed-mixed-path-repro`（目录保留旧名）。

目标是训练**同一个**兼容两种初始化的模型，不是只复现混合路径失败，也不是拆成两个模型。实现与交接已在`6e26621`合入；原工作树、checkpoint及缓存独立保留。不要从本分支清理或改写另一工作树。

用户提供的 `hh.md` 保留原文，属于设计意见，不是代码契约或实验结论。已采纳的主张是“共享模型 + 真实双来源训练路径 + 来源条件消融”，不是“只加一个 bit 就解决分布失配”。

## 产物与恢复

- 2026-09-15 审计补救：新评估入口使用 `bitseed_hybrid_selected_refinement_v3`，续跑前检查所有结构/评分参考、专家资产与评分代码的内容指纹；缺指纹的旧结果或配置不匹配时必须换新 `--out-dir`，`--no-resume` 也不覆盖已有 case。Union 在精修失败时保留有效初始化，但不冒充有效 Final。旧 C40/C81 结果和封存入口原样保留，未重新计算。
- 补救验收：48 个测试文件逐函数 CPU 检查共 245 项通过（补齐测试搜索路径、预载现有 DiffDock 数值缓存）；160 个一方 Python 文件语法检查通过，C40/C81 EMA 严格加载通过。新增回归入口为 `python tests/test_selected_refinement_contract.py`；未启动 GPU 实验。
- 最新 MGD94：`results/mgd_full_pool_refinement_20260913/summary_v1.json`，逐例文件位置由其中 `integrity.case_files` 指定。
- 早期 proposal 消融：`results/proposal_oracle_ablation_20260912/`；只比较同协议目录，不混用 v4 前后的失败处理。
- 旧 SDPI 对照：`results/mgd_followup/mgd94_x64_seed31000/`、`mgd94_x64_seed41000/`；它们不是当前统一池的同一批初态。
- 精修 checkpoint 与复跑命令见[运行指南](doc/训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md)。
- 已清理的历史文档/分析脚本可从提交 `3070922` / `a097801` 读取；恢复到独立目录，不覆盖当前源码与结果。

GPU 状态、PID 和 ETA 不写成长期规范。需要时运行 `bash scripts/gpu_status.sh`，再查具体日志与输出更新时间。

## Model-v3 双线执行更新（2026-09-16）

### Ranking

`model_v3_ranking_night160`的128个新增case采集、精确DockQ和18项post-ranker网格已经完整结束。8例内部验证上没有形成可靠学习曲线：纯几何从32增至160例的union top1没有提升，冻结特征也未稳定胜过纯几何；所有配置的union ≥0.23均为0，仅`frozen+geometry, n=160`有一个seed达到1/8。该结果说明当前特征/标签不够支持部署top1，不允许挑单seed上线。按case配对审计见`results/model_v3_ranking_night160_20260915/paired_audit/paired.md`。

新增`model_v3_initial_selector`后继队列，固定在上述post-ranker完整结束后运行。它对32个原训练case、128个扩展case和8个验证case的全部256个初态打标签，再训练3种数据规模×3个seed的geometry-only 256→64 pre-selector。主指标是selected-best、逐例oracle retention、≥0.23命中和全池oracle ID是否保留；这不是post-ranker，也不是预测C40精修后价值。结果目录为`results/model_v3_initial_selector_20260916/`。

新增C40 future-value selector对照，避免把“当前初始化好”误当成“值得花精修预算”。`scripts/train_future_selector.py`只读初态14维几何，分别训练预测final与union价值；训练用160例中预先随机且按来源平衡抽取的64条，验证强制8例×完整256条。主审计除final/union oracle retention外，还统计初态<0.23、末态≥0.23的救回候选是否被选64保留。后台队列在current-quality selector完整结束后运行三seed×两target，写入`results/model_v3_future_selector_20260916/`。

为定位post-ranker失败是否源于特征过弱，新增native-free expanded-geometry对照。原14维Cα摘要之外，`scripts/extract_expanded_ranking_features.py`从输入蛋白PDB提取重原子，以Cα刚体拟合到每个候选frame，计算双侧配体距离/覆盖及双向蛋白界面37维统计；输入index显式禁止label/native字段，特征产物绑定archive hash。`scripts/train_expanded_geometry_ranker.py`使用完全相同的标签、loss、n=160和三seed，仅替换为51维几何，因此可与旧geometry n160直接比较。后台结果目录为`results/model_v3_expanded_ranker_20260916/`。

full-pool selector完成后另算精确随机基线，不靠有限次随机seed：uniform-64以及按来源32/32、48/16、56/8分别由组合概率得到expected-best、oracle retention和≥0.23期望命中数。报告目录为`results/model_v3_selector_baselines_20260916/random_baselines/`。

新增学习分数后的native-free去冗余对照，不修改或重训selector。选择规则为`q_hat - λ × max RBF similarity`，相似度只使用按训练统计标准化的14维初态几何；预声明`λ={0,.025,.05,.1,.2}`，其中0必须逐项复现原稳定top-64。current-quality审计等待9个selector完成后写入`results/model_v3_selector_diversity_20260916/`；future-value审计分别覆盖Final/Union目标及rescued basin，写入`results/model_v3_future_selector_diversity_20260916/`。这些λ共用内部8例，只是诊断，不能挑最佳值后称为独立验证。

全链结束后，`model_v3_ranking_summary`要求current selector、精确随机配额、future selector、51维重原子post-ranker和两类去冗余审计全部成功，才生成`results/model_v3_ranking_summary_20260916/report/summary.{json,md}`；任一上游失败均拒绝汇总成功子集。总表明确区分“所选64条中的oracle-best”（计算分配指标）与“ranker实际top-1”（部署指标），二者不得互换。新增代码通过67个逐文件测试入口；后台runtime均记录对应Git提交。

旧post-ranker的160例监督共20,480个状态，但仅78个`0.23≤DockQ<0.49`、1个`≥0.49`，因此另加固定loss对照以区分特征不足与稀疏分类项拖累：复用同一51维特征、网络、160/8划分和三seed，只训练同case listwise目标，温度固定为0.05与0.01。队列`model_v3_expanded_objective_v2`等待expanded特征完成后运行6个CPU fit，写入`results/model_v3_expanded_objective_v2_20260916/`；首个无状态文件的等待器未开始fit即停止并由v2取代。总汇也由`model_v3_ranking_summary_v2`取代旧等待器，并把composite与两种ranking-only目标并列，不按seed或温度挑最好结果。

### PROTAC条件模型

已实现独立条件架构、训练checkpoint、训练入口和部署入口：

- 父模型固定为C组checkpoint的EMA权重，并记录父文件SHA256；旧v20加载器拒绝条件checkpoint。
- native训练坐标只生成6 Å离散双侧contact/pocket mask与本侧最近距离，不保存或返回坐标；默认condition dropout分别覆盖整侧、口袋、角色与binary局部几何缺失。
- 新增条件注入严格零初始化门，未训练模型对有/无条件均精确复现父模型；第一步学习门强度，之后条件编码器获得梯度。
- validation为SDPI/BiTSeed端点×full/absent四格；同case共享时间、同来源full/absent共享coupling，回退单列。
- 部署manifest严格绑定case、ligand canonical isomeric SMILES、atom count、片段子图映射和蛋白节点顺序，且没有native三元坐标字段。
- 优化器明确分为父模型`1e-4`与condition新参数`1e-3`两组并写入checkpoint；此前同用`1e-4`的20-epoch pilot显示条件门过小，只作为负面对照。

真实单case/单epoch烟测已完成训练、EMA验证及checkpoint往返。配对后的full/absent数值在两来源上相同到浮点误差，证明迁移合同成立；它不证明条件带来精度收益。8训练/4验证的20-epoch pilot显示condition与父模型同用`1e-4`时门过小；改为condition组`1e-3`后门稳定打开且未数值失稳，但小样本full/absent差仍很小。

首轮formal作业在epoch1结束前停止：审计发现训练条件只出现`training_native_supervision`，推理manifest却会激活从未训练的`template` provenance输入列。该目录保留为aborted诊断，不产出模型结论。v2把provenance/evidence ID仅保留在manifest与报告，不进入条件张量；网络只看role/pocket/binary可用性及confidence。

第二次启动又发现两臂各自扩展endpoint cache时会产生微小GPU非确定性，因此也在epoch1结束前停止，不作结果。v3尝试用default臂预领先共享cache，但epoch1验证后两臂重新同步；cache仅395例，无法保证后续未缓存case不会并发生成。v3只保留完整epoch1及其checkpoint，已产生的epoch2部分不采用。

v4实现了严格的单写者cache，但复查发现训练中“双侧仅pocket可用”的自然概率约万分之一，而PROTAC22首轮部署正是pocket-only；v4的epoch2部分因此停止且不作最终结论。

当前formal作业为`model_v3_conditioned_v5_default/absent`，从各自v3 epoch1 checkpoint续跑epoch2–40，物理GPU5/7；结果写入`results/model_v3_conditioned_full40_v5_{default,absent}_20260916/`，runtime为`results/model_v3_conditioned_full40_v5_20260916_runtime/`。default以0.2概率显式保留双侧pocket、同时移除role/binary；验证扩展为`SDPI/BiTSeed × full/pocket-only/absent`六格。default仍是feature/endpoint cache唯一写者，absent带`--paired-cache-follower`且始终全缺失。两者仍为587训练簇、65验证簇、C40 EMA父权重、相同seed与1:1 SDPI/BiTSeed协议；64个测试文件通过。formal训练序列由v3 epoch1加v5 epoch2–40组成。

v5封存runtime写入`config.json`的`checkpoint_selection`说明字符串沿用了旧四格措辞，但实际`validate_grid`、逐epochmetrics和best-checkpoint判断从启动起均为上述六格算术平均；这是元数据文字错误，不是运行逻辑变化。主分支入口已修正为明确的六格说明，不回写或替换在途封存runtime/配置，最终审计以metrics中的六个`by_cell`及实际selection loss为准。

训练loss自动审计会话为`model_v3_conditioned_compare`：等待两侧`final_model.pt`后核对除dropout/输出路径外的配置完全一致，输出逐epoch四格loss、full−absent差和raw/EMA condition gate；报告明确不把loss差写成DockQ收益。

部署资产新增pocket-only安全降级协议。旧严格化学恢复在PROTAC22中有34/44侧可恢复，但0/22例能同时得到双侧严格warhead映射，故第一轮不猜测atom map。`scripts/make_protac_pocket_manifests.py`只读取每侧unbound protein与同局部frame的`unbound_lig{1,2}.pdb`，按与训练一致的6 Å阈值生成Cα节点口袋mask；22/22例均成功，双侧分别为6–12和6–23个口袋节点。产物为`results/model_v3_protac22_pocket_manifests_v2_20260916/{pocket,absent}/`，报告明确：不读取native ternary、不使用cross-protein geometry、不使用warhead映射，来源仅称provided template。

`evaluate_selected_refinement.py`的新v4协议支持显式conditioned checkpoint/manifest及指定selector arm。训练结束后的结构主对照固定Hybrid-56/8、64 walker、seed31000、20步：

| 对照 | 训练条件 | 推理条件 | 解释 |
| --- | --- | --- | --- |
| default-pocket | 默认缺失增强 | 双侧pocket-only | 完整系统第一轮 |
| default-absent | 默认缺失增强 | 全条件缺失 | 同权重内条件增量 |
| control-absent | 训练时始终缺失 | 全条件缺失 | 条件训练本身的代价/收益 |

三组必须核对逐case proposal pool hash及初态hash；任一不一致不能作配对结论。结构评估报告Initial/Final/Union及≥0.23/0.49，仍需另接post-ranker才是部署top1。后台接管入口为`scripts/run_conditioned_protac_queue.py`，每张物理卡串行执行任务，不与训练同时增加GPU worker。
GPU5队列在训练完成后运行`default-pocket`，GPU7串行运行`default-absent`与`control-absent`；当前会话分别为`model_v3_conditioned_v5_eval_gpu5/7`。`model_v3_conditioned_v5_audit`等待三组全部22例结束后才汇总逐例配对差、胜/平/负及pool/初态/seed一致性，不接纳成功子集。
