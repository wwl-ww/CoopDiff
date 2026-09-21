# Ranking 模型架构

## 2026-09-20：数据规模与更新次数对照

三卡界面消融已全部完成20轮：pooled/contact-independent/contact-interaction的
val Union Top1为0.046876/0.040274/0.048188，均>=0.23命中0/8。
交互组最终略高但不构成可靠泛化突破；不再以该8例内部验证选择更多新模块。

扩展train128目前119例完整归档，0例标签；原5/7队列将在各自补采完成后评分。
新后台队列`results/ranking_data_scale_20260920/`复用授权的0/1/2卡：

| GPU | 训练组 | 总轮数 | 状态前提 |
| --- | --- | ---: | --- |
| 0 | 原train32、无新增界面分支 | 100 | 原数据齐全，预检后启动 |
| 1 | train32＋额外128、原模型 | 20 | 全160例完整归档及标签齐全 |
| 2 | 同train160、两层局部消息传递＋pooled | 20 | 同上 |

保持C40 EMA初始化、seed7315、lr3e-4、decision/top_weight1、辅助/future/budget全0，
val8与256候选全池不变，每case一次更新。预先指定两种比较：
train32 epoch20 vs train160 epoch4（640更新）；train32 epoch100 vs train160 epoch20
（3200更新）。同时报告同20轮结果，但不能把它当等计算量的数据消融。
GPU2选择较简单的既有消息传递作为结构对照，不宣称它是验证集最优。

每组冻结runtime并记录plan/launch；等待数据时不占GPU，齐全后先运行完整checksum、
标签覆盖、PDB/group隔离预检，再确认目标GPU无计算进程。预检失败则停止并记录失败，
不自动删case或换数据。每组持有非阻塞文件锁防重复启动。队列基础测试通过。
这仍是内部验证；未新增独立外部测试或完成家族同源性审计。

## 2026-09-20：三卡界面架构消融（已后台启动）

用户新增授权三张卡，检查空闲后使用物理GPU0/1/2，不改变5/7现有任务。
输出`results/ranking_contact_architecture32_20260920/`，各组源码冻结、命令与PID
见launch.json。统一train32/val8全池、seed7315、C40 EMA编码器初始化、h128、
lr3e-4、20epochs、chunk4、decision全排序＋top_weight1；所有辅助/future/budget
权重为0。主比较固定epoch20，不在反复使用的val8上挑best宣布成功。

| GPU | 读出 | 作用 |
| --- | --- | --- |
| 0 | pooled | 两层局部消息传递后，按组分汇总的既有界面基线 |
| 1 | contacts-independent | 相同消息传递；保留局部接触token，query注意力汇总 |
| 2 | contacts | 与GPU1相同，但汇总前增加一层接触token自注意力 |

每个E3-L/POI-L/E3-POI界面最多取32个最近的无向接触（12Å内kNN图），包含
两端化学表示、关系类型、距离RBF；ligand键参加消息传递但不作为界面token。
方向按组分类型规范化；距离边界精确平局仍有离散选边歧义。两种contact组
共同参数逐项初始化一致，independent组跳过且冻结交互层，dropout均为0。
新分支仍作为原global context的门控残差，不是全原子网络，也不是彻底移除
原编码器的替代模型。蛋白仍为Cα；不能据此宣称解决侧链化学信息缺失。

相关10项测试通过；真实256候选/512状态反向烟测17.34秒，PyTorch峰值allocated
约167MiB（非整卡显存），报告`results/contact_readout_smoke_20260920.json`。
这仅证明可运行，不证明泛化收益。train160仍待补采和完整标签，不混入本轮。

此前两组epoch20拟合审计已完成：纯排序train/val Union Top1为0.188723/0.041407；
三分量辅助为0.188396/0.027391。两者训练集都接近oracle0.189428，继续支持
泛化失败而非不能拟合；不能据此排除数据分布因素。

## 2026-09-20：十小时实验结果与当前判断

GPU5/7队列分别完成10.049/10.055小时，两组均完成20epochs，之后继续补采。
固定epoch20、同一train32/val8、同一seed，不能拿各自best代替这个对照：

| 质量监督集合 | future选64后的Union oracle均值 | 选择后Final Top1 | 选择后Union Top1 | Union Top1>=0.23 |
| --- | ---: | ---: | ---: | ---: |
| 全池（all） | 0.106981 | 0.031448 | 0.029427 | 0/8 |
| selector选64（future-top64） | 0.119683 | 0.026497 | 0.025055 | 0/8 |

均匀随机选64的Union retained-oracle期望为0.139167，两组均未达到；全池Union
oracle均值0.197744，>=0.23机会为2/8。这里随机基线只用于候选保留质量，
不是使用真实标签来部署选择。选64参与训练略改善保留质量但降低最终Top1，
**未证明这次训练方式改动有效，不能推广为默认最佳**。验证集只有8例且反复
使用，不是独立测试；也不能把它与旧train160实验当成单变量对照。
证据：`results/ranking_night_fullpool_20260917/gpu5_all/history.jsonl`、
`gpu7_future-top64/history.jsonl`的epoch20，以及保存的`epoch_020.pt`。

截至本次启动继续任务前，train32全32例均有256条有效轨迹及标签；额外train128
已有52例完成归档、13312条均有效，尚有部分case中间状态。已在GPU5/7恢复
原补采/标签队列，不重新生成候选、不重复已有pair，日志为
`results/ranking_extension_resume_20260920_gpu5.log`和gpu7对应文件。
同时启动固定epoch20的train32/val8拟合审计（只评估、不更新权重），入口
`scripts/audit_fullpool_ranker.py`，结果目标为夜间运行目录下
`audit_epoch20_all.json`及`audit_epoch20_future-top64.json`。
拟合审计已完成：两组训练集全池Union Top1分别0.185030/0.185893，oracle为
0.189428；selector保留oracle均为0.189428，>=0.23机会均命中11/11。
验证集则全池排序和选64后的排序都失败，因此当前主要证据指向泛化不足，
不是模型在训练集上无法学习质量。训练样本仅32个独立case，不能直接锁定
是数据量、分布还是表示问题。该审计自身未更新模型；后续对照见下文。

后续先在相同训练case上比较纯排序监督与明确几何质量辅助监督，保持数据量
和结构编码器可比，再决定是否增加生成器内部特征。不是根据这8例反复使用
的验证集直接选定更大网络。

本地DeepTernary代码核查：`predict_TL12186.py:214`按`p2_rmsd_pred`升序取首项，
`predict_cpu.py:823`把该首项DockQ与全池best DockQ分开报告。质量分数来自
生成器三元节点特征经TwoWayTransformer后的专用query及三层MLP，不是独立
读取PDB的全排序网络。`deepternary/models/losses.py:362`虽命名rmsd，实际标签
为P2预测/目标坐标的L1误差，再以L1监督预测标量；不能误称标准RMSD。该标签
表达式未detach，不应不加分析地照搬其梯度耦合。借鉴点是共享交互表示和具体
质量监督，而不是认定单点回归必然优于全排序。

下一步数据准备核查：现有full-atom标签已包含dockq、fnat、irms、lrms和method。
可以在不重跑生成器/评分的前提下，监督fnat以及两个有界误差分量
`1/(1+(irms/1.5)^2)`、`1/(1+(lrms/8.5)^2)`；三者均值恰好对应本地
DockQ实现（`DeepTernary/DockQ/dockq_util.py:363`）。它们是同一目标的可解释
分解，不是额外独立真值或新信息来源；价值需通过相同数据/编码器对照验证。
这些标签并不等于ligand两端的结合质量，后者尚需单独定义。

### 几何分量辅助监督：实现与配对实验

9月20日已实现可选三输出sigmoid辅助头，读取与quality排序头相同的结构表示，
以三个有界分量的MSE监督编码器；标签detach且不进入结构输入。部署排序仍使用
原有score，不把辅助分数未经验证地替代排序。默认component_weight=0，旧checkpoint
保持兼容。部分标签、非法值及分量均值与DockQ不一致均拒绝训练。

train32/val8共20480个初末态标签通过分解一致性检查。真实256候选、512状态
反向烟测通过，chunk4约19.45秒，PyTorch峰值allocated约140MiB（不是整卡占用）。
证据：`results/ranking_component_smoke_20260920.json`。

已后台启动同数据、同seed7315、同C40 EMA编码器初始化的两组20-epoch对照：

| GPU | 实验 | component_weight |
| --- | --- | ---: |
| 5 | 纯全排序基线 | 0 |
| 7 | 全排序＋几何分量监督 | 1 |

其余统一：hidden_dim128、interface_layers0、lr3e-4、decision目标、top_weight1、
quality_weight0、archived-pairs、quality_walker_scope=all、chunk4。
两组future_weight与budget_weight均为0，隔离质量学习问题；不与上一轮future联合
训练直接当作单变量对照。共享权重初始化和每epoch样本顺序保持一致。
主比较预定固定epoch20的Initial/Final/Union Top1、>=0.23/0.49命中、oracle与差距；
辅助MAE只作诊断。不能根据各组best或这8例反复使用的内部验证宣布泛化成功。

输出：`results/ranking_components32_20260920/{baseline,components}`；
完整命令/PID/源码哈希在`launch.json`，源码冻结在`runtime/`。
原GPU5/7补采队列继续运行，不停止、不修改其运行快照。本次对照尚无效果结论。
更新：上述两组已完整完成20epochs。固定epoch20的val8结果如下：

| 目标 | Initial Top1 | Final Top1 | Union Top1 | Union >=0.23 |
| --- | ---: | ---: | ---: | ---: |
| 纯排序 | 0.026210 | 0.043175 | 0.041407 | 0/8 |
| 排序＋三分量辅助 | 0.023703 | 0.024923 | 0.027391 | 0/8 |

本轮不支持采用三分量辅助监督为默认，不能把训练loss下降视为成功。
两组固定epoch20的train32/val8复评在CPU后台运行，输出
`results/ranking_components_fit_audit_20260920/{baseline,components}.json`。
评估入口已修复为按checkpoint实际监督记录决定是否评估future：本轮两组
future都未训练，不再把其随机头当作selector报告。该修复有专门回归测试；
此前夜间两组future确实受监督，原报告不受此问题影响。
本次相关CPU回归23个文件、132项通过，包括全池训练、精确梯度回放、分量标签、
旧checkpoint兼容性和结构几何测试；不等同于已重跑全仓所有测试。

### 并行泛化诊断

入口 `scripts/audit_ranker_generalization.py`，固定已完成夜间all组epoch20，不读取
正在变化的best模型。train32/val8数据seal与checkpoint核对，运行前检查PDB/group隔离。
结果目录 `results/ranking_generalization_audit_20260920/`，CPU两进程并行，逐case写入，
只有顶层complete=true才表示全部完成。

- support：分来源/阶段统计标签均值、oracle、>=0.23/0.49结构数；记录配体原子数。
  用15项候选几何特征拟合固定alpha=1的ridge，仅训练集标准化/拟合，每case等总权重；
  同时报告固定接触覆盖率规则。蛋白采用Cα，不能将距离代理称为全原子clash或能量。
- intervention：验证8例原始分数与来源交换、统一SDPI、来源embedding置零对照，
  报各来源Top64占比、来源内准确率和选64后Union oracle；只改变标志，不改变结构。
  这是分布外敏感性实验，不等价于重新训练无source模型。
- 几何扰动：每case每阶段/来源取规范顺序首个结构，不按真值挑选；比较整体刚体
  变换与POI/ligand单独移远100Å的分数变化。扰动后未重算DockQ，不能沿用原标签
  声称准确率或严格单调质量关系。

新增3项CPU测试通过。家族同源性、等架构纯DockQ回归及更细粒度表示探针仍待实施；
当前诊断不覆盖这些问题，也不据8例重复验证宣称独立泛化有效。

support任务已完成40例：简单ridge的验证Initial/Final/Union Top1为
0.022620/0.020784/0.021536，固定接触规则为0.035966/0.029345/0.036472，
两者Union >=0.23均0/8。未发现简单几何基线能解决当前问题。
train32中SDPI有12/4096、BiTSeed有55/12288个初末态结构达到0.23，分别覆盖3、11例；
val8中分别为3/1024、3/3072，分别覆盖1、1例。初末态相关，不能当成独立样本数；
这一分布差异需要结合完整来源干预再判断，不能只凭Top64来源比例断言捷径。

来源/几何干预已完成8例（固定夜间all组epoch20）：原始、交换来源、统一SDPI、
来源embedding置零的Union Top1分别0.038516/0.037974/0.039003/0.039003，
全部>=0.23命中0/8。按初态quality选64的Union retained oracle分别为
0.112245/0.127744/0.133546/0.133546；这不是future选择器指标。
32个几何探针的整体刚体变换最大分数误差4.77e-6；POI、ligand分别移远100Å时
各32/32探针降分，均值变化-5.4568/-4.8328。只证明对明显破坏敏感，不证明
能区分近邻basin或细微的高质量结构。干预结果不支持仅靠删除来源标志修复泛化。

### 等编码器纯回归对照（已启动）

新增`--objective regression --top-weight 0`，直接以`sigmoid(score)`与DockQ的MSE
训练原scalar head和编码器，不新增头；推理仍按原score排序（sigmoid单调）。
强制archived-pairs、all、全池有效末态齐全，且quality/component/future/budget权重
全为0。在本次每例256初态＋256末态的条件下，两阶段等权、每case等权。
不宣称回归值已校准；仍不开放未监督的q_hat/p23/p49。

使用同train32/val8、C40 EMA初始化、seed7315、lr3e-4、20epochs与chunk4；这是
监督目标对照，不做验证调参。MSE与排序loss尺度不同，不能直接比较loss数值。
队列目录`results/ranking_regression32_20260920/`，GPU5当前baseline进程退出且
确认epoch20历史记录及checkpoint存在后启动。进程start-time防止PID复用误判；
若前置实验不完整则退出并记录blocked，不自动抢卡。源码已冻结；队列已确认
前置baseline完成20epochs，并自动启动GPU5纯回归训练，目前进入第6轮。
新增回归梯度/训练推理2项与既有decision/replay5项测试通过。

### 同来源、同阶段的局部质量审计（已完成）

`audit_ranker_generalization.py --mode local`已在CPU后台启动，固定夜间all组epoch20，
同时检查train32和val8。输出`results/ranking_local_quality_20260920/local.json`，
只有complete=true才表示全40例完成。不修改训练或原始结构。

每个case内按stage/source分组，以固定E3参考系中的POI Cα和ligand原子坐标找
最近邻；距离为两种组分均方位移等权平均后开方，不做native对齐。邻居只由坐标
决定，去重无向对后比较真实DockQ差与分数差；DockQ差<=1e-4视为不可比较，
分数平局记0.5。报告实际距离，不能将稀疏池中的“最近”自动称为相近basin。
另报告每组>=0.23结构与低于阈值结构的成对判别率；没有正/负样本时为null，
不是零分。比较应分别报告距离区间和case级平均，不能把相关候选对当独立样本。
4项审计测试通过，包含邻居选择不依赖标签以及来源分组隔离。

全40例已完成。以下先在每case内汇总可比近邻对，再对有支持的case等权平均：

| 几何距离 | train准确率（case数/对数） | val准确率（case数/对数） |
| --- | --- | --- |
| <2Å | 0.4583（4/31） | 0.6923（1/13） |
| 2–5Å | 0.6944（10/141） | 0.5376（4/36） |
| 5–10Å | 0.6056（28/1547） | 0.5108（7/404） |
| >=10Å | 0.6679（32/10256） | 0.6153（8/2567） |

大多数最近邻仍>=10Å，不能据此声称已充分覆盖局部basin。仅看DockQ差>.01时，
val的<5Å区间只剩1例6对、准确率0.5，细粒度结论尤其不稳。
同stage/source的>=0.23正负判别率，先组内再case等权，train为0.950474（11例），
val为0.433418（2例）。4I0P的BiTSeed初态/终态分别0.1728/0.3053；2BVR的SDPI
分别0.6048/0.6508。支持“好结构辨别泛化差”，但只有2个阳性验证case，不能
将几千候选对当作几千独立样本，或锁定唯一架构原因。

### 冻结表示探针（已完成）

`--mode probe`在CPU后台读取同一epoch20模型，截取readout之前的global/E3-L/POI-L/
ligand四个结构token，明确排除stage/source embedding。编码器和所有原模型权重冻结，
仅训练集标准化后拟合固定alpha=1的ridge，按原Initial/Final/Union口径评估。
它检验当前表示能否由另一简单读出方式利用，不是等架构重训练，也不能用负结果证明
表示中绝无有效信息。输出`results/ranking_frozen_probe_20260920/probe.json`；新增
测试确认相同结构不同stage/source得到完全相同的探针特征、无梯度且hook正确清理。

探针全40例已完成：训练Initial/Final/Union Top1为0.155004/0.157381/0.185073，
Union >=0.23为10/32；验证为0.022210/0.024208/0.022717，命中0/8。
重新拟合简单读出仍表现出明显训练/验证差距，不支持“只替换读出头即可修复”。
这不等于证明编码器是唯一原因；样本量、训练分布及监督目标仍需区分。
全仓CPU回归完成84个文件、395项通过、0失败，日志
`results/rank_regression_cpu_20260920.log`（包括未提交的历史原型）；随后新增
future评估开关测试另有1项通过。

## 2026-09-17：决策目标修复（代码已实现，效果待验证）

新训练默认 `--objective decision --top-weight 1`。quality 共享同一个结构编码器和
排序头，但分别计算全 Initial、有效 Final、实际 walker 可选 Union 的全排序
loss 与 best-tier loss，再按决策集合等权平均。未入选初态只参加 Initial；不能
挤占 Union 的头部监督。不存在另一个 Union 网络。旧目标可通过
`--objective legacy --top-weight 0` 显式复现；现有后台实验和历史结果不变。

future 明确支持 `--future-target final|union`，默认 final 以兼容旧协议。
Final 标签为末态 DockQ；Union 标签为同一 walker 的 max(初态,末态)。输入始终
只有初态，目标写入 checkpoint 和推理报告。失败、未精修候选仍是未知结果，
不能伪造为零；历史选择造成的训练偏差尚未解决。

默认 `--checkpoint-task quality` 仍按质量任务选 best.pt。训练初始化 selector
时显式使用 `--checkpoint-task future --future-weight 1`，按所声明目标的
Top64 retained oracle 选 checkpoint，而不是借用 quality Union Top1。
该模式要求验证集每例至少 64 个候选且全池终态齐全，训练前检查；失败缺失
不能被包装为完整验证。保留率用于计算分配，不等于最终 Top1 收益。

本次没有宣称解决多样性、失败风险或结构表达问题，也没有重启旧训练。
辅助 q_hat/p23/p49 仍须单独监督；quality_weight=0 的输出不具备预测含义。

### 预算选择监督与数据前提

新增可选 `--budget-weight 1`，只作用于 future 头，保留原有全排序监督。
`budget_retention_loss` 比较最高分的真实最佳层候选与第 64 高分的非最佳层
候选，用 softplus 差值推动至少一个最佳层候选进入 Top64。它不要求该候选
成为 Top1；最佳层有多个候选时不要求全部保留。非最佳候选不足 64 时已保证
保留，返回可微零。该目标是分段可微的保留率代理，不是精确期望 oracle、
多样性目标或最终 Top1 目标；分数并列的部署 tie-break 仍由推理接口定义。

默认权重为 0。启用时逐例要求候选池大于 64 且所有候选终态标签齐全，
否则训练前报错，不生成实验目录。2026-09-17 核查现有
`structural_future160_20260917/ema/audit.json`：160 例全部仅有 64 条终态。
因此该训练集不能用于真实 256→64 预算监督，需要训练 split 的全池精修标签；
不能挪用验证集或给未精修候选填零。现阶段仅完成受控梯度/优化验证，未启动
或宣称完成新的预算训练实验。

### 补齐训练轨迹

`scripts/extend_ranking_archive.py` 从原始归档读取完整候选池，不运行二元生成，
只精修没有 pair 记录的候选。失败记录也保留、不默认重试。原始候选及已有
pair 内容不变，扩展结果写入独立目录，每条轨迹原子保存，可中断续跑；只有
遍历完整池后才写 complete receipt。全池协议为 `complete_existing_pool_v1`，
原始 pair 的 inclusion_probability 保留历史含义，新补 pair 为 1。

必须使用与原归档匹配的 `--runtime-root`，当前已找到
`results/model_v3_ranking_expand32_20260915/runtime_v1`。程序核对 checkpoint、
expert 资产及关键精修代码哈希，使用原 proposal_seed + 2000000 + pool_index。
当前工作树 sampler 已不同，不能直接混入旧轨迹。`--audit-only` 不创建输出；
`--max-new-pairs 1` 仅做一条补采烟测，部分归档不能进入正式标签训练。
支持 `--num-shards 2 --shard-index 0|1` 按 case 分工，每例带排他锁。

32 例真实归档审计已通过，缺 6144 条；全 160 例共缺 30720 条。
2026-09-17 GPU5 单条烟测及一次重启续跑均通过：首例由64→65→66条，新增
终态有效，旧pair及候选未变，部分归档没有complete receipt。
最初启动 `scripts/run_ranking_extension_queue.py` 两个后台 worker（GPU5/7各一个），
计划32例补采→标签→128例补采→标签。晚间改为下述10小时对照队列：仅停止
旧队列父进程，保留两条32例补采子进程原地运行；128例补采顺延，尚未启动。
日志为 `results/ranking_extension_gpu5.log`、`results/ranking_extension_gpu7.log`。
输出为 `results/ranking_fullpool_train32_20260917`、
`results/ranking_fullpool_train128_20260917`，标签目录分别追加 `_labels`。
这是进行中，不是已完成；补采可断点续跑。训练队列状态与重启要求见下节。
最终仍用独立的 `score_ranking_archive.py` 产生标签，原生参考不进入补采过程。

### 全池标签进入训练的契约

补齐后的训练归档可以直接使用 `--candidate-scope archived-pairs`：每个候选都有
pair，因而已经包含全部初态和有效终态。不需要把旧的 initial-label 文件改绑
到新 archive hash，也不需要额外 joint 数据合并；旧标签绑定仍保持严格。

训练与验证的 collection policy 名称只有一种受限兼容：
`complete_existing_pool_v1` 与历史 balanced-random 名称可以不同，但双方每例
必须都拥有全池有效末态，checkpoint、steps、pool、expert assets、seed规则及
五个关键精修代码文件必须一致。缺失、失败、其他 selector 策略或未知代码
版本仍拒绝。原始 train/validation collection policy 均写入 checkpoint 元数据。
真实旧训练/验证归档的这五个关键代码哈希已核对一致。

下一次预算训练应先 audit-only，再开启训练；关键参数为
`--objective decision --candidate-scope archived-pairs --future-weight 1
--future-target union --budget-weight 1 --checkpoint-task future`。
这里 union 明确表示允许保留初态。若只允许最终精修结构，应另行指定 final，
不能把两个目标混报。全池训练尚未启动；3项协议测试、14项数据测试及6项
future测试通过，不代表新的验证集指标已经改善。

全池集成测试 `python tests/test_fullpool_training.py` 已通过：使用256个小型合成
候选（512个初末态）调用真实训练入口，开启 decision、future union、budget
及 future checkpoint 选择，确认 quality/future 两个头和结构编码器均更新，
且 best.pt 使用 Top64 Union retained oracle 指标。训练记录新增独立的
`budget_train_loss`，不能仅凭总loss下降声称预算项有效。本测试预算项为
0.693561；只是接线与梯度验证，不是生物学效果，也不能替代真实蛋白规模的
GPU显存测试。候选跨集合的多样性建模仍未实现。

### 保留完整排序的低激活显存反向传播

`--backward-chunk-size 4` 启用两遍计算：第一遍不保留结构激活，获得完整列表
输出，对整个列表计算loss及输出梯度；第二遍按4个结构重算，用完整列表的
输出梯度累积模型参数梯度。所有候选仍在同一个排序目标中竞争，不是小批
局部排序；不改变标签、预算或优化器更新频率。默认0保留普通autograd。
这会增加前向计算时间，不宣称加速。适用于当前确定性、逐结构独立打分模型，
不支持带训练dropout/BatchNorm或候选集合交互的网络。重算输出不一致时拒绝。

`tests/test_ranking_backward.py` 在两种chunk大小、带/不带界面消息模块、quality
辅助loss及future加权累积下逐参数验证梯度等价；256候选集成训练也通过。
独立真实规模检查入口为 `scripts/smoke_fullpool_ranker.py`：仅对一个训练case
做一次双任务反向和更新，记录耗时、峰值显存及梯度，不导出部署checkpoint。
2026-09-17真实case `3WU2_A_C_SQD` 的256候选/512初末态烟测通过，耗时31.318秒，
PyTorch峰值allocated约140MiB、reserved约170MiB（不含CUDA上下文、其他进程，
不是整卡显存）。两头与编码器梯度非零有限，完成一次优化更新。
报告为 `results/ranking_fullpool_real_smoke_20260917.json`。这是单例、无界面消息
分支的资源检查，不能外推所有蛋白尺寸或验证效果；也不是下面新模式的真实训练。

### Selector参与训练时的实际walker集合

新增可选 `--quality-walker-scope future-top64`，默认all保留全池基线。要求训练
和验证都有全池有效终态，开启decision目标及future头。每一步在参数更新前，
用当前future头对初态预测，按分数和pool-index稳定tie-break选64个walker：

- Initial：仍对全池256个初态全排序。
- Final：只对入选64个末态全排序。
- Union：只对入选walker的128个初末态全排序。
- future：仍用全池最终效用标签监督，并可叠加Top64保留loss。

选择不读取DockQ，不反传离散Top-K梯度；shared encoder仍接受quality和future
各自loss。未入选终态不能改变本步Final/Union目标。若按quality任务保存best，
使用future选64之后的实际Union Top1，而非全池Union榜首；按future任务保存则
继续用对应Top64 retained oracle。两类指标不混用。
3项选择/梯度/保存指标测试以及开启此模式的256候选合成集成训练通过。
该模式尚未做真实训练效果对照；早期selector较弱、选择集合随训练变化是已知
风险，保留all基线，不宣称本模式已是默认最佳。

### 2026-09-17晚间：GPU5/7各10小时软预算

后台入口 `scripts/run_ranking_night.py`。运行目录及冻结训练代码为
`results/ranking_night_fullpool_20260917/`（runtime子目录）；队列日志为
`gpu5_queue.log` / `gpu7_queue.log`，状态为对应 `gpu*_status.json`。

| 阶段 | GPU5 | GPU7 |
| --- | --- | --- |
| 数据前置 | 等原32例分片补采结束，给本分片打标签 | 同左 |
| 屏障 | 等32例标签全部齐全、该卡旧训练结束 | 同左，并等待已有固定epoch审计退出 |
| 主要训练 | quality-walker-scope=all | quality-walker-scope=future-top64 |

两组共同配置：完整train32、原val8全池，C40 EMA复制encoder、hidden128、
interface_layers=0、seed7315、lr3e-4、decision loss、top_weight=1、future_weight=1、
future_target=union、budget_weight=1、quality_weight=0、chunk4、checkpoint-task=future。
唯一研究变量是质量排序监督是否限定selector选出的walker。最多20 epochs；
先前等待/数据阶段消耗夜间预算，训练使用剩余时间。每个epoch结束保存last、
独立epoch_NNN.pt及指标，达到软预算后退出，不中途强杀epoch，所以总时长
可能超过10小时一个epoch及初始化开销。若数据屏障耗尽预算，明确失败，不假装训练。
用户追加要求至少安排足十小时后，队列已更新：若20epochs提前完成，自动继续
剩余train128全池补采，每批最多64条新轨迹，逐条原子保存，到十小时之后的
批次边界收尾；不重复已完成候选。若该分片全部补完，则独立计算其完整标签。
这是训练所需的数据扩充，不用空等或重复实验凑时长。测试覆盖训练提前完成
时进入补采、训练超过预算时不重复启动任务。运行异常仍明确失败，不能保证
故障情况下持续占卡；不会为了时长忽略错误。

比较必须使用两组共同完成的固定epoch，报告future选择的Top64 retention、
选择后Final/Union Top1及>=0.23命中，不能各挑best后当作严格对照。
每卡新任务串行；已有训练和补采会先正常收尾，不新增训练与其争卡。
停止的旧队列父PID为3422735/3422736，保留的采样子PID为3422737/3422738。
不修改已有轨迹、不扩展其他GPU、不git提交。新训练目录不得覆盖，若需续训
用last.pt和新输出路径显式恢复；不要直接重启整条队列覆盖训练输出。

## 当前决策

模型服务两个决策，输入显式提供阶段和实际初始化来源：

| 决策 | 可见结构 | 排序目标 |
| --- | --- | --- |
| 最终结构排序 quality | 初态或有效末态 | 当前真实DockQ的完整顺序 |
| 精修前价值排序 future | 仅初态 | 固定精修协议下的末态效用顺序 |

stage为initial/final，source为sdpi/bitseed，来源编码与现有Consensus约定相同。末态保留轨迹来源；专家失败回退按实际SDPI标记。来源及阶段embedding与结构表示一起输入读出，不将候选ID作为输入。

Final与Union共享quality排序头：Final筛出末态再排序，Union对有效初末态一起排序。它们没有独立头。初态当前质量也可由quality头排序，但该榜单不等于future计算分配榜单。

## 复用已有结构能力

结构编码复用geometry_encoders.py::TernaryGeometryEncoder，包括配体分类embedding、E3–ligand/POI–ligand/E3–POI三类界面、逐配体原子双侧接触以及有符号手性信息。已有graphs.py::build_trifuse_graph负责结构输入。此前重复新写的图构造已经撤回。

ranker拥有独立可训练的编码器实例，可以deepcopy父模型pooling权重初始化。质量排序梯度进入该副本，不更新TriCoopDiff生成器。它直接读取global、e3l、pol和可学习汇聚的ligand_tokens，不必重新运行DiffDock，也不依赖混入扩散时钟/proposal的冻结rank_features。

训练入口现已暴露--encoder-checkpoint与--encoder-weights model|ema，必须同时指定。只复制trifuse_state_dict或ema_trifuse_state_dict里的pooling.*，先验证全部键、形状和有限值，再严格加载到ranker私有编码器；排序头不变，缺EMA不回退主权重。不加载DiffDock运行时，不与生成器共享可更新参数。默认仍为random，没有把预训练初始化提升为已验证更优配置。初始化选择与checkpoint哈希写入audit/checkpoint；audit-only也检查初始化兼容性，失败时不创建成功输出目录。已用真实C40主/EMA参数验证加载，并检查EMA初始化在真实结构上的排序梯度。

已有近邻裁剪、并列距离索引依赖、蛋白角色敏感性仍是明确限制。新增可选界面消息分支补充键拓扑传播（见末节），但尚未证明泛化改善，也未解决全部表示限制。

## 两个排序头与辅助质量输出

结构表示与阶段/来源embedding融合后：

- rank_head输出无界的当前结构排序分数；
- future_head输出无界的精修价值排序分数，只接受initial；
- quality_head为当前质量任务提供三个区间概率及各区间的条件均值。

排序分数不是DockQ，也不是成功概率。q_hat、p23、p49是辅助质量估计，不能与排序分数混称；排名默认始终按rank_head或future_head的score降序，不在推理时临时切换为q_hat。

辅助质量区间为低于0.23、[0.23,0.49)、≥0.49。q_hat=Σπμ，p23=π1+π2，p49=π2。标签所在区间接受条件均值监督，分类监督区间概率。sigmoid均值只能逼近端点；这些概率尚未验证校准。future模式只输出score，拒绝把未来标签用于当前质量辅助头。

两个任务共享结构主干和条件融合，任务头独立。联合训练可能有梯度冲突，默认权重尚未冻结；现已实现可选 `--future-weight` 联合训练，具体固定对照与限制见末节。不能将仅训练quality的历史checkpoint用于future推理。

## 完整列表监督

full_list_ranking_loss针对同一case的一张候选列表，采用处理并列标签的逐级列表似然：

1. 按真实DockQ或未来效用降序分档；
2. 每个候选与所有严格较低档候选竞争；
3. 对所有有更低档对手的候选汇总损失；档内不规定顺序；
4. 标签唯一时等价于按有效后缀平均的ListMLE。

排序相邻档的容差显式记录，默认1e-4。分档以当前档最高标签为锚点，避免近似相等关系的链式扩张。全体标签相同或单候选时排序梯度为零，可选质量辅助损失仍可学习绝对质量。

这不是只优化赢家或将回归值事后排序：中下部错序也影响损失。训练目标可使用完整有效候选列表，但全排序不等于每个训练case已拥有256条精修标签；未生成末态的候选不能伪造标签。

structural_ranking_loss以列表损失为主，quality_weight控制辅助质量监督，默认0。训练器逐case优化，每例贡献一次更新，禁止把不同complex拼成一张列表。当前用同一case的所有已标注初态及有效末态构成一张质量列表。

### 可选的最佳档监督（默认关闭）

full-list目标本身监督Top-1。解析检查发现：分数全部相同、N个不同标签且不落入同一容差档时，按N−1个有效后缀平均的损失对最佳候选logit梯度为−1/N。这个缩放不是实现bug，也不能据此断言实际训练失败原因，但完整顺序与首选质量应分别验收。

best_tier_loss定义为logsumexp(全部score)−logsumexp(最佳标签档score)，即负对数最佳档概率总质量。最佳档按max(label)−label≤tie_tolerance确定，允许并列最佳，不人为指定唯一赢家。全部同档或单候选时损失及梯度为0；辅助项本身不监督其余顺序，必须与原全排序共同使用。

组合为L_full + top_weight·L_best + quality_weight·L_quality。训练CLI新增--top-weight，默认0，所有此前真实烟测仍是原目标。此项没有新模型头，不改变推理接口/分数/标签，不代表校准成功概率，也不利用DockQ差值大小。需要将来预先固定对照协议检验，未按8例验证调系数或重训。tests/test_structural_ranking_objective.py覆盖不同N的解析梯度、并列最佳、单候选/全并列、极端分数稳定性、分数平移不变性以及组合目标保留中后部监督。

## 数据适配与训练入口

structural_ranking_data.py读取已封存归档：验证manifest绑定、归档文件哈希、初末态哈希、实际来源、候选身份和标签一一覆盖。构图复用build_trifuse_graph，用归档候选坐标覆盖POI/ligand模板，不将标签或native参考交给模型。失败末态缺席，但保留初态；未精修的池内初态不自动加入本训练协议。

scripts/train_structural_ranker.py提供quality完整列表训练入口，接收train/val的archives与labels目录及新out目录；--audit-only仅审计，不训练。启动前检查checkpoint/steps/pool/selection一致性及case/PDB/group跨split隔离；这不替代蛋白同源性审计。当前随机初始化ranker，冻结未训练的future_head。quality_weight=0时辅助质量输出同样没有获得专门监督，不能当作可用概率。

逐epoch报告Initial/Final/Union的Top-1、oracle、≥0.23/≥0.49命中及完整预测顺序。另报逐case成对顺序准确率，以及Top-8/32/64集合内最佳真实DockQ（Top-K oracle，不是可直接部署的Top-1）。成对指标排除真实标签差≤1e-4的候选对，预测同分计0.5；无可比对时记缺失。该诊断采用两两容差，与训练的锚定分档不完全相同，不用于挑选checkpoint。Final缺失病例单独计数，均值注明available分母；成功率使用全部病例分母。按验证Union平均Top-1选择best.pt，last.pt另外保存可续训状态。

### 断点续训

--resume /path/to/last.pt按完整epoch边界恢复；--epochs是目标总轮数，必须大于已完成轮数，--out必须是新目录。last.pt原子保存当前参数、优化器、Python/Torch及当前训练CUDA设备RNG、已完成轮数和历史best快照。best.pt用于推理，不作为完整续训文件；旧烟测best仍可推理，但缺恢复状态时拒绝冒充续训。中断发生在epoch中间时，会重跑该轮，暂不支持case级恢复。

恢复前核对数据/标签封存、协议、初始化、模型配置、优化目标、关键实现文件哈希与PyTorch版本。除了输出路径、目标轮数、resume路径和audit-only，参数必须一致；原预训练初始化checkpoint也须仍可读取。原目录保持不变，新目录写resume.json记录父checkpoint哈希，history.jsonl只记新增轮次；继承历史best，避免续训后把较差模型冒充全程最佳。CPU合同测试验证连续两轮与一轮后续训一轮的参数、优化器和验证结果逐项一致；这不承诺跨硬件/CUDA算法的位级复现。

列表逐结构前向且保留完整计算图。2026-09-17在物理GPU 7对hidden_dim=128进行真实完整列表前向+反向烟测（quality_weight=0.1、无optimizer step）：3WU2_A_C_SQD训练case的128状态用时1.65秒、峰值allocated 1423 MiB；4WNB_A_B_4BN验证case的512状态用时5.05秒、3251 MiB。loss及梯度有限，结构编码器梯度非零。这是两个case的计算检查，不是模型效果、全数据最坏显存或整轮训练耗时估计。合成数据另覆盖一轮CLI训练、checkpoint保存和严格加载。

2026-09-17只读检查results/model_v3_ranking_expand32_20260915：32个训练case共2048对（每例64），8个验证case共2048对（每例256），封存与PDB/group隔离检查通过。训练与验证标签覆盖不同，不能宣称训练使用256条全池标签。新增数据合同测试包括候选坐标覆盖、一次合成数据优化更新、失败末态、重复标签/哈希/来源错误及split/protocol泄漏拒绝。

数据审计后锁定每个case的archive与label文件哈希，后续epoch若文件内容变动则拒绝继续；checkpoint元数据记录train/val逐case封存值及label manifest哈希，避免同一目录在训练途中被更新却不留痕。

### 可复用的160例训练归档

32例是管线烟测子集，不是仓库全部可用数据。现已支持--train-archives和--train-labels按相同顺序传入多目录。StructuralArchiveCollection保持分片及case原顺序，逐片核对split与生成器协议、拒绝重复PDB，原manifest不改写；corpus摘要由有序分片哈希派生，checkpoint另记录各原始分片身份。跨训练/验证的group和PDB隔离仍由完整审计检查。

2026-09-17联合审计expand32/train_archives与night160/extra_train_archives：32+128=160例、10240对精修归档，配套20480个初末态标签。全部封存/标签/隔离检查通过，case顺序与night160/indexes/audit.json中的train_cases及val_cases逐项一致。审计产物results/structural_ranker_corpus160_audit_20260917/audit.json。未重采样，未合并改写原文件，未启动160例结构ranker训练。

已有类别支持记录合计79个训练状态≥0.23（24/160 case）、其中1个≥0.49（1/160 case）。因此“没有≥0.49标签”仅适用于32例烟测子集。新增128例已在历史特征ranker实验中用过；同一8例内部验证也已多次用于比较，不能重新宣称是untouched holdout。更大训练集不自动构成更强排序证据。

### 全池初态完整列表监督

训练CLI现支持--candidate-scope initial-pool，沿用同一个结构ranker和quality头，但只监督全部初态。此模式的train/val-labels必须指向exact_initial_pool_labels_v1标签（现有initial_selector目录），不能误用初末态pair标签。适配器核对每个pool_index、candidate_id、实际source、state hash、case及完整覆盖；忽略标签文件里旧14维geometry摘要，模型仍从归档的结构与化学特征构图。数据准备不读取pairs末态。

验证只报告Initial排序和256→64选择，不制造Final/Union榜单。主checkpoint选择指标为Top-64集合内的平均最佳真实DockQ，另报保留率、≥0.23/0.49命中、canonical oracle ID保留及选中SDPI/BiTSeed数量；这只是预算内oracle，不等于可部署Top-1，也不是未来精修价值。全排序损失未改变；测试小池采用min(64,N)，实际归档N=256。初態专用checkpoint标记supervised_stages=[initial]，推理拒绝用其排Final/Union。默认archived-pairs仍是初末态共享头训练；独立initial-pool模式是实验入口，不是决定部署两个ranker。

2026-09-17完整CPU审计：160例训练40960个初态，8例内部验证2048个初态，全部身份/封存/隔离通过。训练标签分布为40875个<0.23、83个[0.23,0.49)、2个≥0.49，分别有36/160例存在≥0.23、2/160例存在≥0.49；验证有3个≥0.23、0个≥0.49，覆盖2/8例。产物results/structural_ranker_initial160_audit_20260917/audit.json。没有重采样/重评分或启动真实全池训练，仅运行合成端到端训练及推理边界测试。

### 单模型联合训练与核心对照

--candidate-scope joint在一张训练列表中放入全部初态及有效归档末态，仍由同一quality头输出同一种当前质量score。--train-labels/--val-labels提供原pair标签，新增--train-initial-labels/--val-initial-labels提供全池初态标签；每个初态仅出现一次，交集初态的DockQ必须在1e-6内一致，两次读取归档哈希也必须一致。典型训练case为256初态+64末态，共320状态；失败末态不补造标签。这不是future价值预测，也不是部署两个模型。

Initial榜单与Top-64保留审计看全部256初态；Final看有效末态；Union由独立eligibility mask限定在归档walker初末态。未精修初态虽然参与训练和Initial评估，不进入Union。Union的结构打乱对照也按eligibility分层，避免把全池候选分布偷渡进原协议。训练仍按一张完整列表平均，初态数量多于末态可能造成任务权重不平衡，本轮不额外调权。

2026-09-17已启动核心对照，目录results/structural_joint160_20260917/：GPU 5随机编码器初始化，GPU 7复制C40 EMA几何编码器；两组均为160例、seed7315、hidden_dim128、lr3e-4、20 epochs、纯全排序（top_weight=quality_weight=0），独立ranker可训练，生成器和DiffDock不更新。除编码器初始化外设置一致，验证仍是8例内部集，不是新holdout。每轮同时报告Initial Top-64与Final/Union Top-1；按预定Union平均Top-1保留best，另存last用于恢复。

启动入口scripts/launch_structural_joint_local.py将训练代码及tricoopdiff复制到独立runtime再启动两个脱离终端的进程；launch.json记录完整命令、PID、卡号和日志路径。代码快照避免后续编辑改变正在跑的实验。正式启动前真实320状态前向/反向烟测通过，峰值allocated约3466 MiB；另有联合列表、标签冲突、Union/打乱边界测试。运行结果以各arm的history.jsonl为准，启动不等于实验已完成或方案有效。

## 推理与选择

forward_list接收同一case的一组结构张量及对应stage/source，逐结构编码后拼接score，保留完整梯度。当前逐结构执行，未宣称已经有高效批处理或跨walker attention。标签不进入forward接口。

rank_states返回全部有效状态的降序索引，支持top_k及initial/final/union集合过滤；select_state是其Top-1封装。失败末态不参与排序，初态仍可用；没有有效末态时Final结果为空。精确同分保留输入顺序，调用者以稳定候选身份排序输入。

Union只改变可返回集合，不重算另一个任务的分数。历史协议只包含选中walker的初末态；允许返回未精修的其余初态属于另外的协议。

scripts/rank_structural_archive.py现提供无标签CLI：--checkpoint指定本训练器的best.pt，--archive指定已封存case.pt，--out指定新JSON，可选--top-k和--device。入口只读checkpoint、归档、manifest及complete receipt，不读取label目录或native文件；校验文件封存、case、生成器checkpoint/steps/pool/selection一致性并严格加载模型。输出三种scope的pool_index/stage/source/score及输入哈希，不把未训练辅助概率暴露为可信指标。adapt_case不带labels时只构图、有标签时再附加训练目标，两条路径结构张量一致。合成闭环测试在移走label目录后验证完整排名、Top-K及协议不匹配拒绝。尚无真实训练效果达标的checkpoint，CLI可运行不等于已可部署。

新checkpoint记录optimizer_steps与head_supervision，区分rank、quality_aux、future是否配置了监督；这是训练来源声明，不是效果保证。quality_weight=0时辅助头冻结，future头仍冻结。默认只输出排名；显式--include-quality-estimates要求可核对的辅助监督记录，只在uncalibrated_quality_estimates下输出q_hat/p23/p49，排名仍按score。没有校准实验，probabilities_calibrated固定false。旧quality checkpoint仅允许legacy_quality_task_only排序兼容，不因存在头权重就开放辅助输出；零更新/监督记录矛盾会拒绝。现有一轮烟测checkpoint已验证保持排序兼容。

候选范围现显式由--candidate-scope控制：默认archived-pairs维持上述三张榜单；initial-pool读取全部生成初态，仅输出initial榜单，不读取pairs中的精修成败/末态，也不把其余初态加入Union。adapt_initial_pool独立校验候选身份唯一性、实际来源及状态哈希，复用同一个state_graph构图函数。即使输入完全没有pairs字段也能构图；CLI仍针对已封存归档做离线回放，在线walker分配尚未接线。

initial-pool使用当前质量头，--top-k 64表示按初态当前质量取64，不等于future精修价值选择，也没有额外多样性配额。训练标签范围仍是归档对，推理扩展到全池不会补出未精修标签。2026-09-17在3WU2_A_C_SQD真实训练归档完成256初态→Top-64接口烟测：所有分数有限、索引无重复、抽查初态与archived-pairs路径分数一致。该检查使用随机hidden_dim=16模型、CPU耗时约5.51秒，无质量效果意义。合成测试同时验证未精修候选可被纳入全池，但不进入默认归档Union。

未来价值排序目前只提供排序头及初态限定。末态效用如何纳入失败、是否考虑已有初态保留价值，必须在训练数据合同中固定。Top-64可由排序结果取得，但它本身不是已经解决冗余的预算策略。

## 信息边界与验证

### 防止标记捷径掩盖结构排序

保留用户要求的stage/source条件，但不能把跨组排序提升直接解释为学会了几何。训练入口新增within_stage_source：仅比较同case、同阶段、同实际来源且真实DockQ差>1e-4的候选对，先逐case计算准确率再平均。仅按标记打分的模型在有可比对的组内恒为0.5（预测同分），无可比对的case单独记缺失。此项当前为诊断，不增加新loss，也不据此反复选择验证集模型。

2026-09-17仅审计32个训练case：initial/final × bitseed/sdpi四个组各1024状态；同组可比对62704，占全部可比对257306的24.37%。这是两两比较的数量比例，不是ListMLE梯度权重，但说明只报全局排序容易遮住组内问题。

tests/test_structural_ranker_geometry.py增加三个受控检查：固定标记时几何能改变分数；固定标记且冻结其embedding时，四个合成结构经过60步可学会人为指定的非单调全顺序；标记捷径反例在全局准确率>0.8时组内仍只有0.5。人工目标只用于容量/可训练性检查，不是物理质量标签，不证明真实DockQ泛化，也不证明训练后模型不会依赖捷径。后续真实训练必须查看同组指标，并另做固定标记内打乱结构的诊断。

训练评估现已接入该打乱诊断：--shuffle-repeats默认32（0关闭），每个case使用由case名和seed确定的独立随机数流，在相同stage/source内均匀置换、允许固定点，不移动标签。报告三张榜单Top-1及组内准确率相对置换基线的差值；逐case记录置换Top-1均值/标准差。它保留组间先验和组内分数分布，只破坏结构分数与真实身份的关联。正差值只是条件关联证据，不是因果证明、置信区间或显著性检验；诊断不参与checkpoint选择。单元素组无法打乱，movable_states显式报告。

当前独立逐结构确定性打分下，同组置换分数严格等价于置换结构后重算，已有模型前向测试验证，因此无需额外GPU前向。若未来加入跨候选交互、候选ID输入或随机推理，必须重新验证这个等价关系，不能直接沿用。本轮另测置换不跨组、不改变全局RNG、仅标记模型的诊断差值为0。

### 真实结构单case容量诊断（不是泛化结果）

scripts/smoke_structural_ranking.py只允许读取train归档；固定case、stage/source，冻结标记embedding与非排序头，随机hidden_dim=128模型用AdamW(lr=3e-4)学习完整组内顺序40步。仅导出JSON，不导出模型。2026-09-17固定manifest第一例3WU2_A_C_SQD、seed=7315，对四组各32结构依次运行，无验证集参与、无按结果挑case或调参。报告位于results/structural_ranker_capacity_20260917/。

| 组别 | loss前→后 | 组内准确率前→后 | 打乱后准确率 | 选中DockQ / oracle |
| --- | --- | --- | --- | --- |
| SDPI初态 | 2.602→0.321 | 0.533→1.000 | 0.503 | 0.0134 / 0.0134 |
| BiTSeed初态 | 2.609→0.603 | 0.546→0.998 | 0.495 | 0.0203 / 0.0203 |
| SDPI末态 | 2.614→0.390 | 0.473→0.998 | 0.504 | 0.0154 / 0.0315 |
| BiTSeed末态 | 2.674→0.472 | 0.534→0.992 | 0.494 | 0.0233 / 0.0233 |

四组各约9–10秒。这证明模型能拟合真实结构的组内顺序，不证明筛选成功或泛化：本case所有oracle均低于0.23。SDPI末态也说明接近完美的成对准确率不保证Top-1正确。评估因此同时报告mean_top1_regret_available（oracle减Top-1）、oracle_opportunities阈值机会数以及hit_rate_given_oracle（池内存在≥0.23/0.49构象的case中实际命中比例）；没有机会时为null，不记成0。未依据这一个case修改列表损失。

### 完整真实入口烟测（1 epoch，不是正式效果结论）

2026-09-17在GPU 7完成32个训练case各一次更新与8例验证，hidden_dim=128、seed=7315、lr=3e-4、quality_weight=0、shuffle_repeats=8，其余为当前默认。结果与模型仅作管线诊断，目录results/structural_ranker_pipeline_smoke_20260917/，audit.json记录完整配置与逐case封存，history.jsonl记录逐例预测顺序。best.pt是该唯一epoch的产物，不代表充分收敛或有效部署模型。随后用此checkpoint完成无标签initial-pool Top-64 CLI回放。

| 榜单 | 平均Top-1 DockQ | 平均oracle | ≥0.23命中 | oracle≥0.23病例数 |
| --- | --- | --- | --- | --- |
| Initial | 0.02765 | 0.19309 | 0/8 | 2/8 |
| Final | 0.02844 | 0.17794 | 0/8 | 2/8 |
| Union | 0.02844 | 0.19774 | 0/8 | 2/8 |

训练平均列表loss=3.88213，验证同stage/source组内成对准确率=0.59873；相对8次置换基线的平均差为+0.09594。此少量置换不是显著性检验。整体排名学到一些关联并不等于Top-1选择有效，未基于这8例调参或追加轮数。

已有label_support审计显示：训练4096状态中17个≥0.23（覆盖6/32 case），验证4096状态中6个≥0.23（覆盖2/8 case），两边≥0.49均为0。现有数据可检查训练机制，不能支撑“高质量选择/≥0.49概率已学好”的结论。初末态关联，状态数不能当独立样本数。扩大训练覆盖、冻结独立评估与正式泛化实验仍待推进。

候选坐标覆盖input_graphs中的模板POI/ligand坐标，化学与节点索引保持一致。数据适配必须验证实际来源、case、状态身份和初末态配对，不能把native参考、DockQ或文件路径作为输入。模型无法从数值判断一份标为initial的张量是否真的为初态，归档适配层仍需严格审计。

tests/test_structural_ranker.py与tests/test_structural_full_ranking.py共10项合同测试覆盖：

- 刚体与节点排列一致性、私有编码器参数/梯度隔离；
- 完整列表梯度、低排名错序惩罚、并列标签处理；
- 阶段与来源embedding获得排序监督；
- 候选重排下分数和损失一致；
- Final/Union共享分数、完整排序和Top-K、失败排除；
- future拒绝final输入及当前质量辅助监督；
- 质量阈值边界与类别合法性。

上述合同检查、单case容量诊断和一轮真实管线烟测不证明可部署排序效果；目前没有已验证可部署checkpoint。旧setwise/trajectory脚本保留为待处置原型，不是本设计的正式入口。

### 稀疏界面消息传递结构对照（2026-09-17）

核心假设：独立接触统计再池化可能不足以识别局部化学连接与双侧界面的组合关系。新增 `--interface-layers 2`，不重建原子编码器，而直接复用 `TernaryGeometryEncoder` 已编码的节点特征，在池化前传播局部信息。默认0保持原架构与旧checkpoint兼容。

- 四类边：配体共价键、E3–配体、POI–配体、E3–POI。跨组件边采用12Å半径内双向各4近邻，显式对称化并去重；不为远离的蛋白强行造边。
- 两层残差消息传递读取源/目标节点、关系embedding和16维距离RBF，按目标度数归一化，再按组件池化。经可学习sigmoid门（初值0.1）注入原global context。
- 原编码器的局部接触、化学身份和手性特征保留。此距离消息分支自身不区分镜像；蛋白仍为Cα节点，配体边尚无键级属性，不宣称全原子物理模型。
- 保留同一个current-quality排序头、stage/source输入与完整列表损失；Final/Union只改变候选资格，不增加独立head。不引入跨walker注意力，也不启用future预测头。
- 为完整320状态列表使用非重入activation checkpoint，反向重算局部消息，既不拆分排序列表，也不截断梯度。直接计算与重算的输出/参数梯度一致性已有测试。kNN等距边界仍存在索引歧义。

架构对照固定C40 EMA编码器初始化、160例训练/8例内部验证、seed7315、hidden128、20epoch、AdamW 3e-4、纯全排序监督，仅改变interface_layers。基线正在 `results/structural_joint160_20260917/ema/` 运行，代码已快照封存，不会被当前修改影响。比较完整学习曲线、Initial/Final/Union Top-1、≥0.23机会命中率、Top-64保留与组内排序，而非仅比较训练loss。重复使用的8例验证不构成独立泛化证据；不得按少数病例涨跌声称架构有效。

结构分支已在GPU 7后台启动，输出 `results/structural_interface160_20260917/ema/`，日志为同级 `ema.log`；运行命令/PID与独立源代码快照见 `launch.json`、`runtime/`。真实训练首例3WU2_A_C_SQD的320状态前向/反向/AdamW烟测通过，loss=4.78037，消息层、门控、原蛋白投影均有有限非零梯度。重算前后loss及所检查梯度相同；峰值分配显存14375→3598 MiB，耗时6.29→8.39秒（共享GPU条件下单次测量，不是性能基准）。新增5项结构测试通过，包括拓扑敏感性、不变性与重算梯度一致性；原38项排序检查和模型构建检查通过。训练尚无结构分支效果结论。

### 核心失败定位与决策回放（2026-09-17，阶段性）

固定第7轮作诊断，不按最佳轮次挑选：EMA基线组内成对准确率0.64885、Union Top-1平均0.05282，≥0.23命中0/8。4I0P_D_H_PGE的Union oracle=0.45304，模型排第40；2BVR_H_I_TYS的oracle=0.36843，排第231。整体顺序相关不能代替首选成功。

新增 `ranking_selection.py::refinement_selection_metrics` 回放真正的两阶段决策：初态分数选择walker，再以同一个质量分数在所选walker的末态/初末态中选Final/Union。标签只在决策后计量，关联依赖pool identity，不依赖数组位置。只有已观测有效配对可纳入，缺失/失败末态不伪造；明确报告配对覆盖率。它衡量固定归档精修轨迹的离线结果，不证明有跨walker交互时仍然等价。

下表由EMA第7轮原始 `history.jsonl` 中的共同Union全顺序回放；8例均覆盖256个初态及其256个有效末态，先验证覆盖完整再以顺序构造等价序数分数。这里不重新调用模型、不利用DockQ选walker。

| 初态选取预算 | 保留初态best均值 | 保留末态best均值 | Final实际Top-1均值 | Union实际Top-1均值 | 保留≥0.23末态病例数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.05949 | 0.06517 | 0.02815 | 0.02857 | 0/8 |
| 32 | 0.13224 | 0.10454 | 0.04891 | 0.05313 | 0/8 |
| 64 | 0.13725 | 0.11169 | 0.04902 | 0.05282 | 0/8 |

完整末态oracle均值0.17794，Union oracle均值0.19774，末态≥0.23机会2/8。当前质量初筛丢失了两例高质量末态对应的walker；因此不能把初态保留率称作精修收益保留率。此事实支持继续区分current-quality与future-utility目标，但尚不能证明增加future头一定解决问题。后续训练评估入口已接入同样回放；已启动任务的源代码快照不修改。

另在GPU 5启动 `results/structural_top160_20260917/ema/`，20epoch、与EMA基线同数据/初始化/seed/学习率，唯一模型训练目标变化为 `top_weight=1`（原全排序+最佳档负对数概率质量，各权重1）。不与新结构同时组合、不扫权重；默认仍为0。本实验用于区分首选监督不足与结构表示不足，不提前断言原因。总loss尺度不同，不能与纯全排序loss直接比大小。

训练正例容量诊断采用base32 manifest中按固定stage/source顺序找到的第一个≥0.23组：6S53_B_A_MPD、initial/bitseed、32状态、同一来源/阶段标记冻结、100步、EMA初始化。原架构与interface_layers=2均达到成对准确率1.0并选中oracle=0.32271；loss分别2.63578→0.03863与2.63575→0.04152。报告 `results/structural_positive_capacity_20260917/layers{0,2}.json`。只证明这一个正例组可拟合，不支持泛化、容量完全足够或新架构更优的结论。

### 已接通的联合质量／精修收益训练

`train_structural_ranker.py --future-weight 1` 在每个case上累计两条损失的梯度，然后只做一次optimizer更新：

1. quality：完整初态＋有效末态列表，标签是各自当前DockQ；
2. future：只输入已配对walker的初态图，标签严格是其固定协议精修后的真实DockQ；不是DockQ增量，也不是max(initial, final)。

两条路径共享结构编码器、阶段／来源表示，使用独立rank/future头。quality反向完成后再计算future，避免同时保留两套结构激活。默认future_weight=0保持旧行为。无末态或失败轨迹的效用未知，不作为0标签、不以初态标签替代；训练入口拒绝完全没有非平凡末态顺序的收益任务。典型训练case为320状态质量列表＋64初态收益列表。64条轨迹来自原归档筛选，存在选择偏差，不能冒称对所有256候选都有收益监督。

评估以同一个联合模型比较quality初筛和future初筛：分别选8/32/64个walker，再由quality头选Final或Union，报告精修后保留oracle和实际Top-1。验证8例具备全256配对，可离线回放；不把任何末态坐标作为future输入。`best.pt`仍按原定Union质量Top-1选择，避免悄悄改变已有模型选择标准；future曲线逐epoch记录，需与固定轮次比较，不能把该best称为最佳future checkpoint。

推理入口支持 `rank_structural_archive.py --task future --candidate-scope initial-pool`，只能加载显式记录收益监督的checkpoint，模型读初态图及标记，不读标签。未监督future头、末态输入或把quality概率当future概率均拒绝。Final/Union始终共享quality分数；收益分数只决定精修计算分配。

真实首例320＋64双任务反向与AdamW通过，quality loss=4.78041、future loss=3.23895，两个头及原编码器均有有限非零梯度，峰值3466MiB、单次约6.94秒（共享GPU烟测）。50项排序相关检查及模型构建检查通过，包含末态坐标修改不改变future预测、配对身份关联、联合训练checkpoint到无标签推理的回放。

固定对照后台目录 `results/structural_future160_20260917/ema/`，代码快照/命令/PID见父目录runtime与launch.json：GPU5、C40 EMA、160train/8内部val、seed7315、hidden128、20epoch、lr3e-4、interface_layers=0、top_weight=0、quality_weight=0、future_weight=1。此时尚无收益头有效性的结论；不自动提升为默认模型。

### 优化问题与计算诊断

冻结读取EMA基线第11轮last.pt，在已见base32训练子集回放：组内成对准确率0.76003；Initial Top-1均值0.06095、阈值机会9例命中1例，初态Top-64保留9/9机会；Final Top-1均值0.06050、机会6例命中0例；Union Top-1均值0.05676、机会6例命中0例。每例只观测原归档64条末态，不能把该子集末态oracle冒充256全池oracle。这说明首选决策在训练集上也未充分学好，暂不能将全部失败归因于跨case泛化。

工作区的全排序损失现用一次反向logcumsumexp计算所有严格较低档后缀归一化，替代逐档重复logsumexp；并列档语义与全部位置监督不变。新增与旧公式的双精度值/梯度一致性检查（含单元素、并列、320/512列表、大幅值score）。GPU7共享条件下320项、20次单独loss反向测量约38.07→1.63ms，float32 loss差0、最大梯度差4.47e-8；不是整个训练的23倍加速。该优化发生在五组源代码快照之后，不改变进行中的实验。

最佳档监督组第5轮在相同base32训练子集：Initial Top-1均值0.13394、阈值机会9例命中5例；Final 0.06544、6例机会命中2例；Union 0.07547、6例机会命中1例；组内成对准确率0.67273。与上面的EMA第11轮轮次不同，不构成严格同轮优势结论，但显示首选与整体排序确有权衡。验证集仍无阈值命中，不据训练结果选择部署模型。

### 固定轮次后台验收与随机预算基线

`scripts/evaluate_structural_runs.py` 等待各组第20轮last.pt，依次评估同一base32训练诊断子集与全部内部val8。重核归档/标签seal与训练时一致，不读取“各组最好的一轮”混出主表。记录完整逐case输出、配置及checkpoint哈希；超过目标轮次且目标权重未留存则拒绝替代。脚本已在随机编码器第13轮上完成真实40case回放烟测（`results/structural_fixed13_evaluator_smoke_20260917/`）。

五组第20轮验收已后台启动，目录 `results/structural_fixed20_audit_20260917/`，封存runtime，GPU7串行评估，最多等待12小时。`reports/progress.json`列出完成及待完成项，逐组JSON含训练诊断与验证、quality和已监督future预算回放。关闭终端不会中断；不自动提交git、不改训练快照。

预算回放新增无放回均匀随机k选n解析期望：分数第j名成为赢家的概率为C(n-j-1,k-1)/C(n,k)，用递推计算；分别报告保留末态/Union oracle、按同一quality头终选的期望和阈值概率。它不是模型输出的概率，也不是按标签选择walker。小集合穷举验证解析期望；终选同分按pool identity固定，不随初筛顺序改变。

同一val8的随机64保留末态oracle期望均值0.12213，对比EMA第7轮quality初筛0.11169；目前没有超过随机预算的证据。历史固定R0（双侧Cα接触/冲突规则，不是Geometry-64）回放8/32/64预算保留末态oracle均值0.07729/0.09787/0.10028，Final Top-1分别0.02586/0.01657/0.01658，阈值命中均0/8。R0没有在这些case上调参。随机基线是解析期望，学习模型数字是一个固定checkpoint的实际结果；不能据8例均值差宣称统计显著。

高质量末态并非都来自同一种初始化：4I0P_D_H_PGE的pool20、148均为BiTSeed，初态→末态DockQ分别0.06767→0.23506、0.16127→0.37152；2BVR_H_I_TYS的pool232为SDPI，0.36843→0.34536。因此不能把本次失败简单归为“只要偏向SDPI即可”。收益任务需要识别跨精修后的顺序变化；现有生成器采样器在固定候选后为确定性reverse flow，不额外加入温度噪声，但收益仍严格绑定生成器checkpoint与精修协议。

### 首个内部验证命中与组合对照

最佳档监督组第11、12轮连续在2BVR_H_I_TYS选中pool232：Initial/Final/Union均命中，DockQ为0.36843/0.34536/0.36843。纯全排序随机/EMA两组完成20轮，全程均未有Union阈值命中。此为复用内部val8上的阶段性现象，不是独立泛化或稳定性证明。

固定取第12轮完整配对回放，预算8/32/64的保留末态oracle均值0.10174/0.12207/0.14047，Final实际Top-1为0.05880/0.06240/0.05767；各预算均保住并选中2BVR的高质量末态。64预算的保留均值超过均匀随机解析期望0.12213，但未经置信区间检验。4I0P仍失败：初态oracle排第23并进入64，两个高质量末态对应的walker却被初筛排除；其最终质量排名为第53和230。因此初态质量、未来价值、最终质量三者不能用一项指标替代。

在原两个训练因素的基础上补齐组合项 `results/structural_future_top160_20260917/ema/`：GPU7，C40 EMA、同160/8数据、seed7315、20epoch、hidden128、lr3e-4、interface_layers=0；future_weight=1，top_weight=1（quality及future各自全排序＋最佳档监督），quality_weight=0。与quality-only/top、dual-head/no-top共同区分两个训练目标因素，而不是扫描权重或更换网络宽度。新组合使用数学等价的后缀扫描损失实现，前述旧组仍用封存实现；不声称浮点逐位一致。尚无组合组效果结论。

### Setwise预算筛选与trajectory终排对照（2026-09-21）

在不重跑生成器的前提下，使用完整train32/val8归档验证两个跨walker原型。setwise
selector以全部256个初态几何及实际来源为上下文，直接监督固定精修协议下的Union
future value，并选择64个walker。三组seed 7315/7316/7317的验证selected-best分别为
0.16641、0.17179、0.16976；完整池oracle均值0.19774，对应case-macro oracle
retention为0.8747、0.8867、0.8586。三组均只保留2个DockQ>=0.23机会中的1个。
这比旧train32质量初筛约0.1329的selected-best高，但checkpoint仍按同一val8选择，
只能作为内部机制证据，不能宣称独立泛化或部署效果。

trajectory set ranker同时读取配对初态、有效末态、位移特征、来源及失败标记，再预测
Final/Union顺序和Union回退阶段。为避免混用旧64-pair特征，重新从当前full-pool
archive离线提取并绑定manifest哈希的train32特征。50轮、seed7315的最佳内部验证
Union Top1为0.03251，Final Top1为0.03068，二者均0/8命中DockQ>=0.23；Union
oracle为0.19774。该原型没有解决终排问题，不作为后续默认方向。

结果目录为`results/ranking_setwise32_20260921/`、
`results/ranking_trajectory_features_20260921/`和
`results/ranking_trajectory_set32_20260921/`。下一步仅扩展有正信号的setwise预算筛选；
train128标签manifest按合同不重复保存case列表；128个标签文件已由archive case列表、
archive manifest哈希和逐文件哈希校验，可直接进入train160与冻结checkpoint评估。
