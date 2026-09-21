# PROTAC 信息建模交接

## 独立参考与目标可达性（2026-09-21，审计实现）

新增 `independent_training_reference.py`：复用部署 BinaryPoseInitializer 与
reference bank；删除输入整分子 conformer，不传 native ligand coords，
在目标拟合之前随机选参考，显式区分真实独立二元证据与 synthetic native-local
二元对。当前通用训练数据只使用后者，不能宣称已有独立实验二元训练材料。

新增 `chart_target_projection.py`：仅优化已有 torsion/offset 状态，不改变参考
键长/键角、不做整体对齐；返回 projected endpoint 和与 native 的坐标残差。
它是有误差记录的局部最小二乘拟合，不保证全局最优，不替换现有精确 target
编码，也尚未接入正式训练。固定参考键长误差另给出最大原子误差的下界。

`audit_independent_training_charts.py` 在三例训练材料上使用部署的 24 候选
构建规则、seed 7315、0.5 Å/0.2 rad 边界；三例均不精确可达。
拟合后 ligand/POI RMSD（Å）：6BD1_A_B_1PE 为 4.3268/11.9385，
6W1O_F_J_BCR 为 0.4040/0.00020，1NVM_A_C_MPD 为 0.1243/0.000028。
前者的大残差也可能含局部优化失败，不能直接当作架构不可达下界。
结果位于 `results/three_source_seed7315_20260921/independent_chart_audit_explicit_provenance.json`。

多起点版本 `fixed_deployment_chart_supervision_multistart_projection_v2` 已完成：
8 个确定性起点、每起点最多 100 次 LBFGS、保存每个起点的误差。完整验证集
39 例中 22 例无合格 chemical-cut，17 例有 binary chart；随机参考下 0/17
精确可达。残差分位范围（ligand RMSD）为 0.051–6.470 Å，POI RMSD 为
0.000073–17.033 Å，尚未冻结接受阈值。

对 6BD1、6W1O、1NVM 审计全部合法 reference bank 后，最佳参考分别为
0.605/0.148、0.404/0.00020、0.039/0.00019 Å（ligand/POI RMSD）。
因此 6BD1 的高残差不是单一起点导致；当前 0.5 Å/0.2 rad bounded chart
与其 native 监督存在真实不匹配。不能靠继续增加 LBFGS 起点解决，也不能把
投影结果当作 native。记录见 `independent_chart_audit_allrefs_v2.json`。

这把下一步收窄为架构/协议选择：若保留 bounded BiAnchor，必须拒绝高残差
材料并接受覆盖下降；若要覆盖这些样本，应让 BiAnchor 使用显式 assembly-capable
chart，并同步部署边界。两者都要先做小规模可达性和梯度检查，暂不长训。

下一步仍需解决高残差样本的拟合/可达性，并制定显式监督合同，才接入三来源
训练；不能将投影目标冒称 native，不能放宽精确编码阈值掩盖误差。本阶段没有
新增动作头、没有长训、没有新 oracle 成绩。

## 三来源联合训练接线（2026-09-21，尚非新的质量结果）

用户要求训练同时覆盖 **SDPI、BiTSeed、BiAnchor**。已扩展现有
`AnchoredTrainer` 与 `train_anchored_entry.py`，新增 `--three-source`
及 `--bitseed-cache-dir <独立缓存>`，不增加模型模块：

- SDPI/BiT 复用原 SDPIFlow 轨迹和速度损失；BiT 为冻结专家 C=1/M=1
  实际生成的 endpoint。BiAnchor 保留双锚流及联合动作。三者共享同一个
  模型、optimizer 和 EMA，不把自由位姿轨迹强塞进双锚运动约束。
- 按实际成功更新数优先补足较少的来源；BiT 失败回退必须标为 SDPI，
  不计入 BiT。连续 128 次未得到所请求来源的更新直接报错，不能假装覆盖。
- v12 checkpoint 保存三来源合同、成功计数及 fallback 数；显式 warm-start
  才能改变训练混合，resume 保持 optimizer/EMA/RNG/数据游标及来源计数。
- 验证分别报告三类损失及覆盖数；BiT fallback 不混入 BiT 均值。固定三类
  cohort 指纹，继续按 BiAnchor 开发损失选 PROTAC 候选，不直接平均异尺度损失。

真实 GPU 前台短检：初次 6 次更新为 `2/2/2`，恢复后第 9 步为 `3/3/3`；
每次对应来源 embedding 行有非零梯度，其他行梯度为零，专家无梯度。
记录在 `results/three_source_seed7315_20260921/{preflight,resume_preflight}/`。
这些 checkpoint 明确标为 implementation_smoke，不能当作准确率结果。
分来源验证实测 SDPI 39/39、BiT 39/39、BiAnchor 16/39 可用；BiAnchor
另外 23 例按原化学切分门槛拒绝，不伪造双锚材料。三类 loss 分别为
4.911246、3.558633、0.601546，量纲/目标不同，不相互比较、不平均。
完整记录为 `results/three_source_seed7315_20260921/validation_preflight.json`。
33 项针对性检查及模型构建、训练契约检查通过；GPU 短检已结束。

**尚未完成的独立事项**：BiAnchor 当前训练参考仍沿用 native 派生 chart，
尚未改成部署同构的独立二元/化学参考；三来源覆盖不能替代这项对齐。
因此本轮只检查接线，不启动旧参考上的另一轮长训。

前一轮 BiAnchor-only 已在第 13909 步停止。训练后正式保存节点中按开发
loss 选出的 step 4096（0.646851）七例 32-walker oracle 为 0.3318595，
旧 baseline 为 0.3330815；3 升 4 降，无平均收益。结果见
`results/bianchor_seed7315_20260921/evaluation_step4096/summary.json`。
全历史开发 loss 最优仍是未更新的 step 0，不能把它的结果冒充标志位训练效果。

## BiAnchor 来源标志与单轨继续训练（2026-09-21）

当前执行路线覆盖下方较早的架构提案：只保留 SDPI、BiTSeed、BiAnchor
三种初始化身份，不实现六种来源/运动组合，不添加动作模块。chemical-cut
中段不能直接冒充真实 PROTAC linker；此前“修通 linker”的建议尚未实施，
也不是本轮任务。提交 `855dcda` 保存既有依赖，`288a40d` 单独引入来源标志。

- BiAnchor 使用原 `source_embedding` 的新第 2 行及已有门控；迁移复制旧双锚
  模型使用的第 0 行，保留旧 SDPI/BiT 行。旧 checkpoint 保持旧来源语义。
- 从 prior_joint 最佳 step 65536 EMA warm-start；原 checkpoint SHA256
  `a2477b9b4c5de97f328103bf379f6b1c34200ee4d1d5dac577b6447b15274c76`。
  保留动作模块、化学先验及二元边界；重建 optimizer，单 seed 7315。
- 前台短检完成 2 次真实更新；BiAnchor 梯度范数为 0.00431018、0.00389200，
  SDPI/BiT 行梯度为零，专家无梯度，96 次验证调用均使用 BiAnchor。
  初始开发损失精确复现 0.6014466606041727；两步后 0.601465340082844，
  仅作接线检查，不作质量结论。
- 5T35_D_A_759 的 2 walkers × 2 steps 采样通过，报告来源为 BiAnchor，
  未计算 native score。正式轨迹从这 2 步恢复 optimizer/EMA/RNG/数据游标，
  不另开 seed；GPU 5 前台运行，无后台 supervisor、无更新数或时间上限。
  每 2048 步内部验证并保存，后续仍按开发损失选权重。

本轮配置、审计、预测及训练记录统一位于
`results/bianchor_seed7315_20260921/`：`preflight_audit.json`、
`sampling_preflight.json`、`train/config.json`、`train/metrics.jsonl`。
生成脚本 `run_foreground.py` 记录完整命令；运行中状态以进程和落盘文件为准。
最终结构比较沿用七例、32 候选槽位、20 步协议；不是 ranking、模型 top1，
也不是完整 PROTAC22。尚无新模型 oracle 结果，不承诺标志位本身提高准确率。

## 架构整理与下一步决策（2026-09-21，尚未实施改造）

### 结果与适用范围

五条前台评估全部正常结束（各 7/7，exit 0），训练已停止。七例平均 oracle
DockQ：prior_joint **0.333081**、soft_assembly 0.239825、dual_core 0.229576、
dual_binary 0.227811、dual_contact_core 0.184376。每例 32 候选槽位、20 步、
seed 7315；相同初态可能复用预测，不是模型 top1，不是完整 PROTAC22。
prior_joint 相比 dual_binary 七例六胜，绝对均值增加 0.105271；不同路线
不都共享同一运动空间/先验分布，不能据此归因某一个模块或声称超过 DeepTernary。
权重由各任务开发损失选出；prior_joint 为 step 65536，保留作后续基线。
结果见 `results/prior_modes_20260920/prior_joint/night_unlimited/evaluation_foreground/summary.json`，
其余同各模式目录。以下运行中描述均为历史。

### 先说明已有能力，避免重复设计

核对胜出任务 training/config.json 中的源码哈希，以下前向关键实现与训练时一致：

| 路径 | 已有能力 | 对应实现 |
| --- | --- | --- |
| 二元条件 | 完整化学映射备选、蛋白局部距离、角色与可用性 | condition_features.py |
| 动态二元几何 | 逐配体原子—蛋白节点误差、方向/力矩、配体键消息传播 | binary_geometry.py |
| 三元界面 | 有化学特征的两侧接触、蛋白间接触、方向与手性信息 | interface_geometry.py、geometry_encoders.py |
| 条件融合 | 两侧独立上下文及联合上下文；条件进入节点、门控、刚体和 torsion 输出 | conditional_consensus.py |
| 联合决策 | 五个聚合 token 协商、逐键上下文；新头中 K 个键与四个位姿向量 token 共同 attention | consensus.py、prior_joint_action.py |
| 执行 | 联合运动求解后加全状态修正，单一解码器执行更新 | anchored_consensus.py、bounded_joint_motion.py |

实际链路是“冻结专家建议 + 已条件化的三元几何编码 → 融合动作 → 联合运动
求解 → 读取已有上下文及先验导数的联合动作修正 → 一次坐标更新”。先验
在求解前后都参与，不能描述为“只有末端几个标量补救”；也不能把另一条
TriFuse 实现中的 expert bridge 当成当前 Consensus 分支实际启用的能力。

### 事实、假设与改动边界

| 问题 | 当前证据及边界 | 对应动作 |
| --- | --- | --- |
| 专门 linker 条件分支训练/部署不一致 | **已确认调用路径**：build_training_condition_features 不传 linker_indices，角色第 2 列为零；SideAwareConditionFusion.linker 因 availability=0 被清零。5T35_D 部署 manifest 则有十个 linker 原子。CPU fixture 确认 linker context 零输出。不是全部 linker 建模失效 | 优先修通已有条件合同；使用 chemical-cut 明确的 middle_atoms，不把未知外围全部当 linker。原有检查及 checkpoint 版本同步更新 |
| 接触身份在动作决策前聚合 | **代码事实、瓶颈待证实**：局部消息进入节点，界面再汇总；动作头接收 global/bond 上下文，未直接保留完整接触对。既有信息并非全部丢失 | 若需要更细决策，复用现有 pair message，延迟聚合并直接连到已有动作 token；不另建一套重复接触编码器 |
| 动作决策分两段 | **代码事实、性能影响待证实**：normalized_velocity → joint solve → velocity + correction；不是两次坐标更新 | 候选改造是合并输出职责。保留专家建议及可执行运动几何，撤除被替代的输出头/相加路径，不在后面再接第三个修正器 |
| 训练参考内部几何与部署不同 | **已确认**：训练 chart 从 native 构建，独立化学先验主要迁移 torsion；soft_assembly 仍继承此参考，只放开位姿并加错位噪声 | 先明确从独立二元/化学构象构建训练输入后的目标可达性；当前编码器不能表达的内部形变必须显式处理，不能把投影后的目标称为原 native |

### 执行顺序与保留/替换清单

1. **先修已有路径的实际断点**：梳理 chemical-cut 中段、完整 warhead 和
   protected core 三类标注，修通 linker 训练条件；训练/部署均保留 unknown
   的语义。验收为真实样本角色非空、已有 linker 分支可获梯度、未知不冒充
   linker、旧 checkpoint 不被静默重解释。无需增加模型模块。
2. **再确定一份融合接口改造**：明确接触级消息如何进入现有 joint action，
   哪些旧输出头停止生成动作、几何求解在哪一层执行。先写张量/坐标/单位
   合同与参数增减清单，尚不能把两段决策认定为主因。只保留一条完整可执行
   更新路径，保留旧 checkpoint 加载能力但不同时运行旧新动作头。
3. **训练输入同步对齐部署**：独立局部二元参考不带 native 跨蛋白变换；
   native 仅作监督，先验来自化学图/局部二元证据。检查非扭转几何错配和
   长链有效样本覆盖，不能仅把 native-derived chart 的 noise 调大就声称解决。
4. 新路径通过实际梯度、几何一致性、checkpoint 恢复及部署检查后，才做
   一个 seed 的主线训练。优先一张卡训练、一张卡评估；白天前台，未获新的
   夜间授权不自行后台。保留固定预算七例回归，并分先验可用性补齐其余案例，
   不改变测试权重选择规则，不启动五路线/多 seed 扫描。

**保留**：冻结专家、已实现的局部二元/化学/三元几何编码、联合生成框架、
单一坐标更新、明确条件来源与映射、独立 seed 和数据隔离。
**拟替换而非堆叠**：重复承担动作输出职责的头、过早聚合到全局的部分接口。
**暂不改**：ranking 分支，不预设放开全部几何约束会更好，不宣称因架构调整
就能超过 DeepTernary。本节是整理和拟定顺序，尚未修改模型或启动实验。

## 白天切换前台评估（2026-09-21）

用户要求白天前台执行。已停止五个旧后台队列及其评估进程；当时均在首例，
没有已写出的 predictions/scores。原训练与日志保留，首例未落盘采样重算。
`scripts/evaluate_five_foreground.py` 已在前台 PTY 会话启动并监督五张卡，
仍使用各自冻结快照、best.json、原七例及相同候选预算。控制进程退出或收到
中断时终止其拥有的评估进程组，不再交由夜间队列独立运行。
新结果目录为各模式 `night_unlimited/evaluation_foreground/`，状态记录为
`night_unlimited/foreground_status.json`；旧 `evaluation/` 是中断记录。
评估 PID：dual_binary 2538711、dual_core 2538712、dual_contact_core 2538713、
prior_joint 2538714、soft_assembly 2538715。以下后台评估描述已被本节替代。

## 当前状态：五条训练已停止，最佳权重评估中（2026-09-21）

按用户指令向五个训练子进程发送 SIGUSR1，均正常完成最终验证与保存，
随后原队列自动转入后台 oracle 评估；不再继续训练。以下旧的“训练中”
描述为历史。五个 best.json 对应的 checkpoint SHA256 均已核对。

| 模式 | GPU | 最终更新 | 开发损失最佳 step | 评估进程 PID |
| --- | --- | --- | --- | --- |
| dual_binary | 3 | 109979 | 69632 | 2527462 |
| dual_core | 4 | 98824 | 24576 | 2527312 |
| dual_contact_core | 0 | 98285 | 0 | 2527310 |
| prior_joint | 5 | 96735 | 65536 | 2527478 |
| soft_assembly | 7 | 106003 | 102400 | 2527460 |

统一评估七个原已验证双侧二元案例，每例 32 槽位、20 步、seed 7315，
沿用显式分层 chemical/uniform 混合与重复化学组合缓存协议。使用开发损失
选中的 EMA，native 仅供结构评分，不做 ranking、不按 oracle 再选权重。
不同模型运动空间与装配先验不完全相同，不能宣称初态完全配对。
这不是完整 PROTAC22 benchmark。结果在各模式
`night_unlimited/evaluation/summary.json`，未完整完成前不报完整均值。
多核心最佳为 step 0，明确代表夜间训练没有刷新其开发损失最佳值。

## 新增两条非“调整保护范围”路线（2026-09-20）

用户授权另用两张空闲 GPU；检查 GPU 5/7 为零显存、零利用率且无计算进程。
原 GPU 0/3/4 三个冻结快照任务保持运行，不修改其运行代码。

- `prior_joint`（GPU 5）：替换原动作模块为 `PriorJointActionAdapter`，不是叠加。
  显式读取两侧完整映射的局部距离误差及粗粒度蛋白重叠信号，计算当前解码器
  下的状态导数；所有 linker token 与四组蛋白位姿向量 token 共同 attention，
  同时输出 K 个 torsion 和 12 个位姿修正。向量仅作标量加权组合，保持几何
  表示的一致性；坐标仍仅由原联合解码路径更新。checkpoint training v9，
  `binary_evidence_joint_action_v1`，保留 dual_binary 的原运动空间。
- `soft_assembly`（GPU 7）：新增 `AssemblyFlow` / `dual_assembly`，保留原统一
  动作头，二元相对平移改为线性参数、旋转改为无球映射截断的指数映射，
  不再有 .5 Å/.2 rad 的二元位姿硬上限。内部结合片段几何仍刚性，不宣称
  全分子无约束。先验一半保留二元位姿，一半注入每分量 5 Å 平移和 .5 rad
  旋转向量高斯错位；训练和推理遵守同一混合先验，linker 仍用化学/均匀混合。
  流匹配之外增加预测终点的二元局部距离重建与粗粒度重叠损失。

两者均从原 step 6144 迁移共享骨干、重新初始化动作模块，仅 seed 7315；
专家冻结，不使用 benchmark native 构造条件或筛选候选，不做 ranking。
新增几何/梯度/先验测试、既有合同回归及 model-build 通过；两模式实际
训练两步、固定开发验证、保存与 5T35_D 两 walker 两步部署检查通过。
这些是实现验收，不是结构质量结果。soft_assembly 短跑可出现大幅位姿偏离，
这是取消硬上限后的风险，不代表更高准确率。其高斯尺度和损失权重为工程
设定，尚未校准；独立二元的非扭转内部形状错配仍未解决。

配置及验收产物位于 `results/prior_modes_20260920/`，后台正式运行记录以
各模式 `night_unlimited/status.json` 为准。无时间和更新次数上限，每 2048
更新开发验证并保存；SIGUSR1 发送给训练子进程可正常保存收尾并进入评估。
两条均已后台启动：GPU 5 队列 PID 158612，GPU 7 队列 PID 158613；
冻结代码位于 `results/runtime_snapshots/prior_modes_20260920/`。两者实际
恢复 checkpoint 后完成第 3 步并保存的检查也通过；正式长训仍从原定父权重
重新迁移，不从 smoke 续训。装配部署坐标重建最大误差 8.23e-5 Å。

## 当前运行：三卡无时限重启（2026-09-20）

用户取消时间上限，要求直接重来，并授权使用检查为空闲的其他 GPU。
已停止旧 GPU 3/4 队列（82827/82828），保留日志，不从这些短跑续训。
三条轨迹从原定 step 6144 共享骨干重新迁移并重置动作头，仍仅 seed 7315。

| 物理 GPU | 模式 | 新队列 PID |
| --- | --- | --- |
| 3 | dual_binary | 99502 |
| 4 | dual_core | 99503 |
| 0 | dual_contact_core | 99504 |

GPU 0 启动前为零显存、零利用率且无计算进程。新独立进程会话使用快照
`results/runtime_snapshots/three_mode_unlimited_20260920/`；运行记录在
`results/three_mode_unified_20260920/<模式>/night_unlimited/`。
实际命令无 `--max-seconds`，`--max-updates 0` 表示无更新次数上限；
队列 `deadline_unix=null`。每 2048 更新仍做开发验证并保存，异常即停。
不设置十小时自动切换评估。需要正常结束训练并进入固定 oracle 评估时，
向 status.json 中 training 子进程发送 SIGUSR1：完成当前更新后保存退出。
以下十小时方案和第三卡待分配描述均为历史，已被本节替代。

## 三模式统一动作头夜间方案（2026-09-20，替代下方两卡草案）

用户现已授权三卡夜间后台运行；此前明确分配的卡为物理 GPU 3/4，第三卡
仍待指定，不能因其他卡暂时空闲而占用 ranking 资源。

- `dual_binary`：保护完整二元结合端，主要通过 linker 与联合刚体动作装配。
- `dual_core`：每侧保护主要接触刚性块，释放外围合法转键。
- `dual_contact_core`：保护所有合格接触块及其化学连接子树，释放非接触分支。

三者均保留完整二元模板软条件，共用冻结专家和一个联合动作头。删除旧的
body-aware MLP 与 coupled 输出相加路径，不再串联堆叠；动作模块参数由
342,438 降为 300,965（减少 12.1%）。新合同为 training v8 / unified action v3。
从旧 step 6144 显式迁移共享骨干并重置动作头、优化器及 EMA，不是精确续训。

三模式均通过实际训练更新、固定开发验证、限时保存和 5T35_D 两步部署检查；
dual_core 另通过真实 checkpoint 续训。这些是实现验收，不是准确率结果。
配置及检查产物位于 `results/three_mode_unified_20260920/`。

`scripts/run_anchored_overnight.py` 每卡串行运行：9 小时 10 分钟训练预算，
另留 5 分钟保存余量与 45 分钟固定七例评估，总墙钟上限十小时。仅 seed 7315，
评估每例 32 槽位、20 步；开发损失选权重，native 只供最终 oracle 评分。
不同模式运动空间不同，不声称初态完全配对；不做 ranking。失败即停，不重复
失败或复制候选凑时长。实际启动以各模式 `night_detached/status.json` 为准。
GPU 3/4 已分别启动 dual_binary/dual_core，队列 PID 为 82827/82828。
首次 shell 后台启动未保住队列父进程，已停止其两个训练子进程并改用独立
session 重启；旧 `night/` 目录仅为失败启动记录，不代表有效十小时队列。
第三模式通过验收但未启动，等待第三张物理 GPU 的分配。

以下核心实现记录及两卡草案保留作历史。长链任务重平衡、外围校准容差、
独立二元几何错配仍未解决；本夜方案不宣称已实现这些能力。

## 核心保护／外围可动改造：已开始实现

新增 `binding_core.py`：仅由每侧已映射二元片段的化学图和本侧蛋白接触，
沿合法化学旋转键划分刚性块。保护核心要求至少三个接触原子、可定向的
rank-two 几何；按接触数及块大小选择，最优块并列时明确拒绝，不按原子编号
或 ternary native 选优。这是初版工程规则，不是已校准的生物学核心判定。

新增 `core_training.py` 与 `core_conditions.py`，训练入口支持
`--conditioning-task dual_core`，采样按 checkpoint 任务自动使用相同核心规则。
**只缩小硬运动约束，不删除其余二元先验**：完整结合端的模板距离/角色信息
保留为网络条件；只有核心随蛋白强约束，其余合法扭转进入已有联合运动求解。
训练中完整局部模板随核心刚体变换，而不随随机化的外围 torsion 偷偷改变。
恢复禁止 whole-fragment/core 任务混用；部署目前要求两侧已验证 manifest，
尚不接受实验 analogue-core augmentation 的混合输入，缺失侧不虚构。

合成检查证明：外围自由度增加、两个核心局部关系保持、native 目标可达；
旋转/平移及原子重编号不改变核心选择；部署不读取完整配体坐标；checkpoint
任务隔离。新增与既有训练合同共 16 项测试通过。
真实 1AFQ 反传通过，刚体、torsion、body-aware、coupled 头均更新且专家冻结：
`results/core_protected_motion_20260920/full_template_cpu_gradient.json`。
同目录 `real_cpu_gradient.json` 是较早的 core-only 条件 smoke，已被保留完整
模板的上述检查替代，不能作为当前条件构造合同的完整证据。

5T35_D 部署初始化的 CPU 几何检查：核心大小 8/16 原子，保留 23 个合法
参考，每个参考 21 个可动 torsion（旧全片段路径为 10 个）。这是运动空间
确实改变的证据，不是结构准确率结果；没有用 native 筛选核心或候选。

尚未完成：外围内部扭转的校准容差、独立二元／native 的非扭转几何错配、
长链任务平衡，以及更完整的联合动作决策层改造。当前外围只受化学转键与
既有解码器约束，不能宣称已有经过校准的内部形变上界；不启动十小时长训。

## 下一夜间窗口草案（仅规划，未启动）

用户要求先规划每卡十小时任务。范围仍为获分配的物理 GPU 3/4，每卡一个
串行队列；未收到晚安/后台启动指令前，不启动新的后台队列。正在进行的
step 6144 前台评估不变。以下为待实现、待验收的任务设计，不是运行记录。

| 卡 | 主任务 | 需要解决的问题 |
| --- | --- | --- |
| 3 | 现有 coupled-linker 模型的长链双侧条件训练 | 实际更新过度集中于 2–4 个自由度；用训练集自身的有效运动学任务平衡长短链，保留已有短链能力与双侧约束 |
| 4 | 同一模型体系的部分核心先验补全训练 | 推理新增部分核心先验，而训练主要使用完整化学切割端；训练条件、保护核心及允许自由度需一致，未知部分不伪装成已知 |

两条轨迹均只用 seed 7315、冻结专家、原有 benchmark 排除与开发划分。
新任务属于显式 warm-start，不能伪称改变训练合同后仍在精确 resume。
不做 ranking、不扩候选预算、不复现 DeepTernary、不用测试 native 生成条件。

预算暂按每卡约 8–8.5 小时分段训练、余下 1–1.5 小时固定结构评估与安全
收尾安排；实际步数由新任务的短跑吞吐确定，不照搬旧任务速度。旧任务实测
2,048 更新约 11.6 分钟，七例/九例评估约 19.3/30.2 分钟，不能据此声称
几个短任务就能覆盖十小时。十小时为墙钟窗口，不保证 GPU 始终满利用率；
若非有限、合同错误或持续无进展，不以重复失败或重复候选凑时长。

开跑前准备门槛：实现并验证长链任务调度与部分核心任务构造；固定新合同及
候选协议；真实梯度 smoke；恢复游标/EMA/优化器检查；限时队列的安全保存、
截止及失败处理。旧 `run_conditioned_overnight.py` 调用的是旧训练入口，不能
直接用于当前 anchored 模型。正式命令、各阶段截止时间、停止判据及备选任务
在门槛通过后再冻结；当前没有已验收的十小时可执行队列。

## 最新进度：v7 续训后的同协议结构评估

v7 续训已完成至 step 6144，固定开发 loss 从 step 4096 的 0.584893
降至 0.562585；按既有开发规则选择 step 6144，权重 SHA256：
`16ed023f744e723037a2a1d205c3ac214e3fe79c9bc74561c5310a188c5aae21`。
尚未证明结构精度随之改善，不继续自动追加训练预算。

step 4096 的新增核心先验九例面板已全部完成，平均 oracle DockQ
**0.168237**；最后一例 6W8I_E 为 0.151712。完整结果在
`results/coupled_linker_core_oracle_panel9_cycles_seed7315_20260920/summary.json`。
它与原七例面板的案例、条件和候选协议不同，不合并均值，也不能当作全 22 例
或模型 top-1 成绩。目标尚未达到。

已将 step 6144 同时送入两个前台评估，仍仅 seed 7315、32 槽位、20 步：

- GPU 3：`results/coupled_linker_dual_oracle_panel7_seed7315_20260920_cont6144/`。
- GPU 4：`results/coupled_linker_core_oracle_panel9_cycles_seed7315_20260920_cont6144/`。

分别复用各自 step 4096 评估的冻结代码。两对 config 的源代码 hash、案例顺序、
条件路径、缓存路径、seed、预算和混合协议逐项一致，只更换所选权重及输出目录；
完成后仍需逐例核对实际初始参数，避免把候选池变化误报为训练收益。

## 当前结论与正在交付的修改（2026-09-20）

双侧有界联合运动模型完成 3,072 次更新，EMA 开发 loss 为 2.078114；
权重按固定开发 loss 选择，未按 benchmark 选优。固定 7 例、32 walkers、20 步、
单一 seed 7315 的完整结构评估平均 oracle DockQ **0.2164329**，没有达到目标。
逐例为 5T35 0.1140、6HAX B/A 0.2568、6HAX F/E 0.1049、6HAY B/A 0.2591、
6HAY F/E 0.4299、6HR2 B/A 0.2027、6HR2 F/E 0.1475。不是 model top-1，
也不是全部 22 例。结果：`results/bounded_joint_oracle_panel7_seed7315_20260920/summary.json`；
权重：`results/bounded_joint_training_seed7315_20260920_cont3072/step_3072.pt`。
模块完整性、低训练 loss 或约束保持均不能替代结构准确率。

发现当前初始化会丢掉化学构象的扭转关联：二元组装完成后又独立均匀随机化
所有可动键。新增 `chemical_torsion_prior.py` 从去除全部输入坐标的化学图生成
ETKDGv3/MMFF94s 松弛构象，仅向 chart 转移可动二面角，不转移键长/键角或
已知结合片段内部形状。已知二元偏移从零开始，仍允许模型在既定 bounds 内
修正；不再为噪声而主动破坏输入姿态。训练与部署共同使用此策略，不给旧权重
偷偷换 prior。它不保证转移后全分子无碰撞/低能，也未解决 native 内部几何与
独立二元组装的分布差异。

早期 v1 误将 MMFF 在 500 步内未收敛也列为拒绝，丢弃了大量长柔性分子；
该 CPU 构象库准备已停止，旧产物仅留作诊断，不用于新训练。v2 将优化收敛状态
与几何有效性分开记录：允许有限、键几何有效的未收敛松弛构象作为 proposal，
不称其为能量极小值；无参数、嵌入失败、非有限或键几何失败仍拒绝，无 native fallback。
新库：`results/chemical_priors_relaxed_seed7315_20260920/{train,val}/`。

单侧模型已接入统一训练器和采样入口：`--conditioning-task single_anchor`。
已知 E3–片段保持有界约束，未知 POI 保留独立六自由度，不虚构另一端 warhead。
只支持第一侧已知，反侧输入显式拒绝，尚未实现自动侧交换或完全无先验路径。
CPU 可用性检查为 318/504 可用、185 几何/化学拒绝、1 无片段；真实专家反传通过。
首次两更新保存/7JTO 两步采样均为 implementation smoke，不代表训练成绩。

新增任务/化学先验使用 checkpoint v4；旧双侧 uniform prior 保持 v3，不能静默
互换。checkpoint 记录 prior 配置、训练/验证构象库 hash、任务类型及原有来源；
resume 不允许改这些值，跨任务仅显式转移 EMA 权重并重置优化器和选优历史。
当前仅一个训练 seed；不做 ranking、不增加 seed 扫描，GPU 仅用获分配的 3/4，
前台运行。下方按时间保留的旧提案（多 seed、优先消融等）不再代表当前执行要求。

### 化学先验训练、参考构象库与后续尺度修正

v4 两个任务均完成 2,048 次更新。双侧 EMA 开发 loss 0.744677 → 0.640794，
单侧 2.539829 → 2.031201；均按开发 loss 选 step 2048，不能跨任务比较数值。
目录为 `results/chemical_{dual,single}_training_v4_seed7315_20260920/`。
最初不含 `_v4_` 的两个目录因 RDKit binary molecule 未解码而在初始验证退出，
未训练；已修复并补充 bytes 输入测试，不能将那些目录当作有效训练。

构象库代表集为 train 482/504、val 38/39 可生成。二元初始化不再必须先压缩为
一个 clash-selected 参考：`binary_chart_bank.py` 保留全部通过相同化学/拟合检查
的离散参考，化学 prior 的双侧采样按参考循环、构象无放回安排，不重复同一
参考/构象对。5T35 的 24 个原始候选保留 12 个，可组成 32 个不同组合。
这是 reference 内部几何/映射的不确定性，不是 native 选优或 learned ranking；
不保证不同组合的最终结构不同。旧 uniform checkpoint 仍用旧单参考策略。

新双侧 7 例/单侧 13 例面板已完成，均为每例 32 walkers、20 步、seed 7315。
运行代码封存于 `results/runtime_snapshots/chemical_prior_v4_20260920/`，每例启动
前校验源文件 hash；原 worktree 后续改动不进入此轮评估。面板结果分别位于
`results/chemical_dual_oracle_panel7_seed7315_20260920/` 与
`results/chemical_single_oracle_panel13_seed7315_20260920/`。双侧平均 oracle
DockQ **0.159237**，低于上一版 0.216433；单侧为 **0.167879**，不达标。
双侧逐例重建同一批初态并独立评分，初始 oracle 与最终值几乎相同（5T35
0.39835 → 0.37681，其他六例变化绝对值不超过 0.0012）。这说明该轮生成
尚未有效改善先验装配；不能将单例先验收益解释成网络生成能力。重建先验证
保存终态参数与原输出一致（最大允许误差 0.001 Å），没有额外模型采样。

复用已有缓存和 CCD，将数据恢复扩展到完整原索引：6,834 中 3,273 成功，
保留 527 簇；其余明确记录缓存/CCD 缺失或化学不兼容，不重新运行 ESM。
`results/anchored_training_chemistry_20260920/all_cached/` 包含完整报告与簇成员。
其中 2,868 个已有化学 prior，图完全相同的构象库按图 hash 复用，非坐标复用。
下一阶段采用每簇一次访问、80% 任意成员/20% representative 的确定性调度；
大簇不因成员多而自动获得更多梯度更新，完整索引和调度随 checkpoint 固定。

还发现单位尺度问题：旧 binary latent MSE 使亚埃级模板偏移与全蛋白尺度
位移使用不恰当的同等权重。`anchored_training_metric.py` 已实现用 decoder
精确 Jacobian 转成实际平移/旋转速度，再按 receptor size/π 归一化的 binary
metric，保留 1e-4 latent ridge；torsion、free POI 和整体 Cartesian 项不变。
这是尚待训练/结构验证的尺度修正，不改正在评估的 v4。新 metric/簇调度使用
v5 合同；切换时显式 warm-start EMA，重置 optimizer/validation，不混比 loss。

### v6：整蛋白随动动作头与先验覆盖

`anchored_action_adapter.py` 在原有骨干和联合运动转换之后加入可训练的
可行 torsion 残差，读取原 bond/global context，以及真实解码器下 ligand/POI
的扭转切向量与学习到的界面方向的收缩。明确包含整 POI 随连接键运动，
不是另做 ranking，也未替换或解冻 DiffDock experts。末层零初始化保证迁移
初始函数不变；真实冻结专家 CPU 反传已验证新旧动作头均有参数更新。
这只证明训练链路通畅，不证明准确率提高。含新头的 checkpoint 使用 v6。

双侧与单侧分别从对应 v4 EMA warm-start，在 GPU 3/4 前台训练各 4,096 次
更新，仅 seed 7315。双侧已完成，开发 loss 0.632240 → 0.614824（step 3072，
选中）→ 0.617033（4096）；固定 7 例面板已完成，平均 oracle DockQ
**0.1779511**，仍低于早期有界联合模型的 0.216433，不按 benchmark 选权重。
单侧也完成 4,096 步，开发 loss 1.315258 → 1.316679，最优仍为 step 0；
这轮未交付优于 warm-start 的权重，不再追加整套单侧评估。
使用上述物理 binary metric、簇调度，以及 50% 化学
构象/50% 均匀 torsion 混合 prior；已知二元偏移仍从零开始，给自由小分子
构象库未覆盖的结合构象保留支持。训练与推理共用同一配置并写入 checkpoint。
目录：`results/body_aware_{dual,single}_training_v6_seed7315_20260920/`。

扩大后的数据使用 `all_cached/index_frozen_source.json` 和对应先验
`all_cached/manifest_frozen_source.json`：`rebind_training_index_source.py`
已证明原索引与冻结 v7 索引仅路径相对/绝对表示不同，完整条目、元数据、簇
在路径规范化后相等。新文件保留来源 hash 和证明；没有放宽 train/val 隔离。

可信部署条件仍只有 7 双侧、13 单侧、2 无侧。新增独立的
`conserved_binary_cores.py` 审查类似物二元模板可保留的精确化学核心：仅沿
非环单键切分，检查原父图的键级、杂化及立体对应，不使用坐标选映射。
当前仅 audit-only，未改变部署条件；局部核心一致不能自动证明类似物姿态
可以严格转移，更不能把未匹配原子全部标成 linker。

新增互变异构体归一化审查 `tautomer_binary_cores.py`，仅接受枚举完整且重原子
身份、连接与立体中心不变的结果。7JTO 第二侧可对应 36/36 个原子（原直接
化学核心 25/36），7Q2J 为 33/36；不是允许不同氧化态/不同骨架任意匹配。
当前输出仍是明确的不确定模板假设，不能宣称实验姿态完全等价。

`prepare_conserved_core_charts.py` 保留既有可信侧，仅给缺失侧增加假设。
10 个有新增第二侧核心的案例中 9 个通过既定化学键/二元拟合/运动学检查，
7JTP 几何拒绝，未 fallback。报告位于
`results/conserved_binary_cores_20260920/chart_feasibility/`。

`conserved_core_conditions.py` 及采样/评估入口的显式参数
`--conserved-core-chemistry-report` 提供独立实验通路：从已校验 hash 的原二元
化学输入重新计算映射，不将 audit JSON 直接当部署 manifest；同时接入局部
距离条件和硬运动学。保留原模型化学图/所有可信侧，不把其余原子标为 linker。
新侧 confidence 特征为工程常数 0.5，不称校准概率；仍使用 checkpoint 既定
二元有限偏移边界，并在预测 audit 中标记 experimental/未校准姿态不确定性。
此通路尚无结构准确率结论，不替换默认条件。普通入口新增 audit-only manifest
拒绝门，旧 MCS 审查集不能通过普通采样入口被静默用于硬约束。

首次新增核心面板全部在推理前的身份检查退出，没有有效预测。查明缓存存在
非手性原子上的伪手性标签；先加 atom-map 身份标签会使相同取代基人为不等价。
现在先清理非手性中心标签，再进行带原子身份的图比较，真实立体差异仍拒绝。
清理仅作用于条件证明的临时图，不修改原缓存/模型输入；清理位置记录于 audit。
九例真实缓存核对、完整配体坐标扰动不影响条件的测试均通过。
重启目录：`results/conserved_core_dual_oracle_panel9_seed7315_20260920_cachefix/`，
固定 v6 双侧 step 3072、32 walkers、20 步、seed 7315。前两例 5T35_H 为
0.272461、7JTO 为 0.124426，一升一降；面板未完，不提前报告整体提升。

### 当前 v7：周期二面角与跨键动作交互

`linker_action_geometry.py` 从当前坐标计算前三阶二面角 Fourier 特征及键对
几何，汇总全部邻接四原子组合，不依赖任意参考 chart 原点或最低原子编号。
`coupled_linker_action.py` 在 v6 动作头上增加两层跨键注意力，键对偏置包括
轴方向、距离、转动子树重叠和 POI 随动关系；没有按键序号的位置编码。
新输出零初始化，保留原骨干、v6 头和冻结专家；checkpoint v7 显式迁移，
不允许恢复时丢掉新增模块。不是独立的新生成模型或 ranking 模块。

真实 1AFQ CPU 反传验证新旧头均更新，专家无梯度；见
`results/coupled_linker_action_20260920/real_cpu_gradient.json`。相关运动学、
训练合同、化学先验和新增模块的 71 个断言测试通过，仅代表实现检查。
GPU 3 前台训练已完成，目录 `results/coupled_linker_dual_training_v7_seed7315_20260920/`，
从 v6 双侧 step 3072 EMA warm-start，单 seed 7315、完成 4,096 更新，沿用
相同数据、混合 prior、物理 binary metric 和簇调度。step 0 开发 loss 精确
复现 0.614824，step 1024 为 0.617743、2048 为 0.619715、3072 为
0.602655、4096 为 **0.584893**，选中 step 4096；不能将其视为结构精度已达标。
权重 SHA256：`8f388b9886f5b66b0c73e0adb9fc3490130f52b4e95653d5ebf03324be4bf4a4`。
固定原有 7 例/32 walkers/20 步/seed 7315 的结构评估已在 GPU 3 前台启动：
`results/coupled_linker_dual_oracle_panel7_seed7315_20260920/`。本轮不启用新增
核心条件或分层混合采样，保持与 v6 七例面板相同的候选协议；源代码冻结在
`results/runtime_snapshots/coupled_linker_v7_stratified_20260920/`（目录名表示
代码包含可选调度功能，不表示本轮启用了该选项）。模型构建测试全部通过。
首例 5T35_D oracle 为 **0.487885**（v6 同例 0.223534）；两轮 32 个初始
参数、参考 chart audit、prior audit 与采样协议逐项完全一致。其余案例未完，
不能由首例提升宣称达到 DeepTernary 或七例整体改善。

七例随后全部完成，v7 平均 oracle **0.201503**，高于 v6 的 0.177951，仍
低于 v3 的 0.216433（v3 的 prior 不同，不能作为同池训练收益比较）。v7 逐例：
5T35_D 0.487885、6HAX_B 0.115022、6HAX_F 0.329663、6HAY_B 0.116376、
6HAY_F 0.088717、6HR2_B 0.156347、6HR2_F 0.116507；相对 v6 三升四降。
因此不以首例成绩代替整体结论，尚未达到目标。

已在 GPU 3 前台启动同一 v7 轨迹的 2,048 次更新续训，总步数目标 6,144：
`results/coupled_linker_dual_training_v7_seed7315_20260920_cont6144/`。
从 step 4096 精确 resume，保留优化器、EMA、数据游标、随机状态和 best 历史；
不增加 seed、不改变先验/数据调度/度量/开发 cohort，也未增加模型模块。
此前末两次开发 loss 持续改善，且七例均值相对 v6 改善，作为有限续训依据；
续训不等于结构精度必然提升，长链任务分布偏短的问题仍未解决。

本轮实际 4,096 次更新中，2–4 个可动扭转的任务占 2,722 次（66.5%），
至少 10 个的仅 271 次（6.6%）；这些是训练日志中实际 chart 自由度，不是
分子总旋转键数。当前 PROTAC chart 存在 10/12/17 个自由度的任务，说明
数据并非完全没有长链，而是训练任务分布偏短。后续优先考虑在现有训练集内
平衡有效装配自由度、保留簇与 benchmark 排除边界；该调整尚未实施或开训，
不能把数据分布观察直接解释成性能瓶颈的因果证明。

### 混合候选调度修复与初态诊断

新增先验面板中 6ZHC 在生成候选前失败：旧调度把连续均匀扭转候选也计入
有限的参考/化学构象组合容量。`stratified_anchored_prior.py` 将 32 个候选明确
分成 16 个化学构象与 16 个均匀扭转，仅前者消耗离散组合；真实化学容量不足
仍拒绝，不扩大预算。采样与面板入口通过 `--stratified-mixture` 显式启用，
单候选的混合先验边缘分布不变，但候选池联合分布与旧 iid 分量选择不同，
因此单独记录协议，不能与旧面板混算完整平均值。三项调度/分量合同测试通过。
实际 6ZHC 分层重试仍在模型采样前失败：化学分量自身也不能提供 16 个唯一
组合，说明原先仅归因为连续候选占位并不完整。CPU 复核为 12 个有效参考、
ETKDG 请求 16 个但仅嵌入并接纳 1 个构象，总容量为 12。失败目录为
`results/conserved_core_6zhc_stratified_seed7315_20260920/`；不降低几何门槛或
偷偷更改混合比例。旧九例面板最终完成八例、失败一例，不报告完整平均值；
最后一例 6W8I_E 为 0.057587，不能省略该退化案例。

已完成八例均完成同池初态重建和评分，未额外生成候选。逐例 Initial → Final：
5T35_H 0.192679 → 0.272461；7JTO 0.144697 → 0.124426；
7Q2J 0.320370 → 0.268121；6W7O_C 0.072523 → 0.074398；
6W7O_D 0.144047 → 0.166226；6W8I_F 0.463650 → 0.384602；
6W8I_D 0.264082 → 0.419173；6W8I_E 0.053440 → 0.057587。
这八例为五升三降，不代表包含失败案例的完整面板结论。

后续显式增加 `--stratified-mixture --allow-chemical-repeats`：协议
`stratified_chemical_cycles_uniform_reference_pool_v2`。化学组合用完后才重新
随机循环，仍保留 16 个化学/16 个均匀分量，不把短缺转换成额外均匀样本。
这改变候选池联合分布，不改变单候选的混合先验边缘分布。相同参考与完全相同
初始参数的确定性轨迹只推理一次；每个复用项记录原 walker，输出另列
`model_trajectories_computed`，不将重复项称为独立新结构。五项调度测试通过；
默认 v1 仍拒绝短缺，历史协议保持不变。

v7 step 4096 接新增核心条件的九例评估已在 GPU 4 前台启动，目录：
`results/coupled_linker_core_oracle_panel9_cycles_seed7315_20260920/`。
统一 32 个候选槽位、20 步、seed 7315，代码冻结于
`results/runtime_snapshots/coupled_linker_v7_cycles_20260920/`。6ZHC 已越过旧的
容量错误进入采样。此轮同时换了权重与候选池联合协议，不能拿其与 v6 的差值
作为单独的训练收益；GPU 3 原七例评估继续使用旧协议，承担同池比较。
6ZHC 已完成 32 个槽位、28 次实际轨迹计算；4 个复用项的初始参数、参考编号、
终态参数和两侧坐标均与原 walker 完全一致，化学/均匀计数仍为 16/16。
调度、面板完整性和实际 Consensus 反传/采样的九项测试通过。

额外 CPU 检查仅诊断了 6ZHC 化学嵌入的缓存伪手性影响：清理非手性原子 26
的标签后，保持原 embedding seed 180403623 和请求数 16，可接纳构象由 1
变为 2；仍不足以说明主要覆盖问题已解决。该检查未改变当前训练库、模型输入
或已启动评估的先验合同，不能将清理后的产物混入本轮候选。

`reconstruct_anchored_initials.py` 已支持实验保守核心条件：重验源报告、条件
audit 和保存终态重建误差，再恢复同批初态，不追加网络推理。7JTO 的 32 个
候选 oracle 从初态 0.144697 降至终态 0.124426，终态重建误差小于 0.000073 Å；
因此缺少条件不是唯一问题，模型更新尚未改善该例。6W8I_F 新条件终态
0.384602，相比旧单侧模型 0.089776 提升；进一步还原其同批初态得到
0.463650，说明这一例同样由先验提供较好候选，当前 v6 更新反而降低 oracle。
这不是训练增益证明，也不能仅凭这两个案例断言全部模型更新均无效。
CPU chart 准备脚本现复用实际采样的条件构造函数；旧 `chart_feasibility/`
报告早于缓存伪手性清理，不作为当前采样完全等价的证据。

## 条件运动学 flow（2026-09-20，训练/验证/采样链路已接通）

### 训练数据接入与化学修复

训练索引共有 6,834 entries / 587 clusters，未提供真实两端片段字段。
`anchored_fragments.py` 提供独立标识的 **chemical-cut training augmentation**：
只沿合法旋转键切出三个连通分量，两端完整片段至少各 3 原子，几何接触仅决定
整个片段的侧别和适用性，不逐原子推断 warhead。中间未标注原子保留 unknown。
这不是 warhead 发现算法，不应用于部署的真实片段身份识别。

`audit_anchored_training.py` 仅读取签名有效的已有特征缓存，按训练簇代表检查，
不下载模型、不重新计算 ESM，也不改变 benchmark 排除规则。初次检查发现
1AFQ 缓存将约 1.21–1.23 Å 的两处 C–O 键标为单键；CCD 恢复后为羰基双键，
并恢复芳香环。原始 587 个代表中有 314 个触发键几何/图不一致检查，不能
据此断言每一个都是相同的键级错误。

`prepare_anchored_training_chemistry.py` 复用 `restore_ligand_chemistry.py`，从明确的
[CCD 化学定义](https://www.wwpdb.org/data/ccd) 恢复键级/电荷；缺失时通过显式
`--fetch-missing-ccd` 从 PDBe 获取定义，只使用化学字段，保留原始沉积坐标。
`ligand_chemistry.py` 校验元素顺序及仅平移的坐标对应，重建全部 ligand node/
edge/torsion tensors，校验后复用原蛋白特征。旧数据、缓存、checkpoint 不修改。
独立产物 `results/anchored_training_chemistry_20260920/representatives/`：
504 恢复成功（370 个 canonical SMILES 改变），83 拒绝，其中 81 个不完整/修改
组分、2 个缺 CCD。这里只处理每簇一个代表，不能称全量训练数据已修复。

同时确认 vendored torsion mask 仅检查图桥键，不筛选键级。新 chart 保持原输出
索引，但通过 RDKit rotatable-bond SMARTS 及酰基—杂原子排除规则限制自由度，
不允许双键、三键或此类共振键自由转动。该筛选不是完整势能模型。
修复后 504 个代表使用此规则检查：265 可用、61 无合适切分、178 被几何检查拒绝。
其中 175 个为键长 bounds 不一致，3 个为目标重建误差超阈值。部分磷酸基的 CCD
中性质子化模型与观测键长不一致，不能简单归因于坐标损坏；当前仍保守拒绝，
没有为提高通过率放宽阈值。前后统计的化学筛选及样本集合不同，不作配对提升解读。

真实训练目标烟测 `scripts/smoke_anchored_training.py` 已在物理 GPU 3 前台执行，
仅用修复后的训练例 1AFQ、seed 7315、一次 AdamW 更新。真实冻结专家无梯度，
torsion head 更新范数约 1.13e-4，loss 约 0.131（仅接口检查值）。结果：
`results/anchored_training_audit_20260920/real_training_step_1AFQ.json`。
此次单样本烟测未保存 checkpoint、未训练 benchmark，也没有新增性能成绩。

### 训练器、断点与数值精度（后续实现）

`anchored_trainer.py` 已提供独立 AdamW/EMA、目标可达性检查、梯度有限性检查、
冻结专家保护与 checkpoint。当前格式为 `anchored_consensus_training_v2`，显式记录
生成过程、条件/片段策略、旧 source embedding 0 的复用策略、父权重/训练索引
SHA256、expert identity、activation clamp、torus 版本、数据游标和 CPU/CUDA RNG。
旧 conditioned loader 拒绝新格式；新 loader 默认拒绝 `implementation_smoke`。
保存仅发布新文件，不覆盖已有 checkpoint；恢复拒绝不同数据、父权重或运行设置。

`scripts/train_anchored_entry.py` 是有显式更新预算的前台入口，要求 benchmark
排除、SDF 及缓存中的 CCD 修复来源。逐例记录增强/几何拒绝，模型或 backward
异常不伪装成数据跳过。使用 case/epoch 派生随机流；当前选优规则见下文，
最初无验证的 v1 checkpoint 只属于历史烟测。
`results/anchored_trainer_contract_20260920/{continuous,resumed}/` 的 v1 内核烟测
验证：连续两次更新，与第一次保存后恢复的第二次更新，模型/EMA 的所有张量
差值为 0，游标及 CPU/CUDA RNG 相同。产物都是烟测，不是正式训练权重。

此前 3 个目标重建超阈值例，在固定同一角度后用 float64 重算，误差降至约
1e-12 Å，确认为长杠杆累积的 float32 数值误差，而非目标不可达。运动学旋转
与二面角恢复现使用 float64 内部计算，网络与输出仍为 float32，验收阈值仍为
0.001 Å。生成过程升级为 `anchored_torsion_chart_v2_fp64_geometry`，明确拒绝
旧 v1 烟测 checkpoint。三个例现在 POI 最大误差约 1.82e-4、1.43e-5、5.91e-5 Å。
当前代表样本检查为 **268 可用 / 61 无切分 / 175 键几何不一致**，见
`results/anchored_training_audit_20260920/restored_fp64_kernel.json`。
新内核的真实训练器更新见 `results/anchored_trainer_contract_20260920/fp64_kernel/`，
仍仅一次烟测更新。后续验证协议和首段训练见下文；尚无结构性能结论。

### EMA 验证、选优和独立采样入口

`anchored_validation.py` 对 EMA 使用固定 case seed、化学切分、角度先验及
时间点 `[0.25, 0.5, 0.75]`，先按例平均再跨例平均 loss。验证不改变在线权重、
优化器或训练 RNG，异常退出也恢复状态。每次记录有效样本集合与先验的 hash；
集合变化时拒绝选优，模型异常不可当作数据跳过。此指标不是 DockQ 或 oracle。

验证化学恢复显式要求 `--disjoint-from` 指向**修复前完整训练索引**；筛掉的
训练样本也不能流入验证。`validation/index.json` 39 个恢复成功的簇代表来自
原 46 个代表，其余 7 个拒绝。隔离覆盖原训练 1,453 个 PDB，恢复后训练 454 个
PDB 与验证 39 个 PDB 无重叠。父模型用过历史验证集，因此仍称内部开发验证，
不声称全新独立 holdout。产物位于
`results/anchored_training_chemistry_20260920/validation/`。

当前验证 16 个代表有效，23 拒绝（18 键几何不一致、5 无合法切分）。checkpoint
v2 保存验证索引 hash、协议、固定 seed、cohort hash 与 best loss/step。正式
`scope=training` 必须有完成的验证；resume 必须携带匹配的 `--resume-best`，
不能偷偷换验证数据或丢失之前的最佳权重。best 文件由 EMA 开发 loss 选择，
而不是 benchmark 或 native oracle。`best.json` 明确记录选择路径与判据。

真实验证/恢复烟测位于 `results/anchored_validation_contract_20260920/`：
step 0/1/2 的 EMA loss 约 0.542906 / 0.542904 / 0.542903，cohort 相同。
这些微小变化不作为学习收益。`sample_anchored_entry.py` 已从新格式加载 EMA，
直接走真实 binary 初始化和约束采样，不借用旧 SDPI evaluator。默认拒绝烟测
权重；显式 `--allow-smoke` 的 5T35 三步检查锚端误差 0，另一端结合距离最大
漂移约 2.48e-5 Å，输出为 `unscored_5T35.json`，没有计算 native 分数。

首段实际训练已完成，位于 `results/anchored_training_seed7315_20260920/`：
固定 seed 7315，477 个样本访问、256 次有效更新、221 次拒绝；step 0/128/256
固定 EMA 开发 loss 为 0.542906232 / 0.542259659 / 0.541160532，cohort 一致，
按既定规则选择 step 256。这只是阶段权重，不代表收敛或超过 DeepTernary。

部署完整采样发现旧初始化先按碰撞选优，可能选中连接键被拉长的候选。
新增显式候选验证回调，约束路径在碰撞排序前检查键几何、二元拟合、E3 帧
和化学切分；旧初始化调用不传回调时维持旧策略。全部拒绝则报错，不放宽
阈值。审计策略为 `anchored_validity_before_clash_selection_v1`。
同 seed 的 5T35 默认 24 候选中 12 个通过、12 个拒绝，完整 20 步采样锚端
误差 0、另一端二元距离最大漂移 3.39e-5 Å。相关 31 个函数检查通过，包含
筛选顺序和全拒绝回归。此变更不改变训练路径或已保存权重。

`scripts/score_anchored_prediction.py` 独立读取已落盘预测，复用现有全原子
DockQ 桥接；bound 评分模板仅在预测完成后使用。该 5T35 单构象 DockQ 为
0.017528、Fnat 为 0，结果在 `scored_5T35_step256.json`；不是候选池 oracle，
不能作为先验利用已带来准确率提升的证据。单侧/不确定先验、全量数据与
完整 benchmark 候选池指标仍待完成。

### 运动学接口进度

联合动作层（2026-09-20）：`anchored_joint_motion.py` 将两侧接口的平移、旋转
和允许运动键的 torsion 建议转换成同一角速度，再由原非线性解码器更新。
POI 侧输出是配体相对 POI 的运动，不能当作全局 POI 速度。接口平移取配体
质心，旋转使用配体最小二乘刚体分量；相对速度扣除 POI 在配体质心处的
速度。所有向量先按原 SDPI 尺度归一化，每个三维刚体分量取均方权重，
torsion 整体取均方权重，并加 1e-6 的 torsion-centered ridge。求解在
float64 中进行，梯度传向刚体与 torsion 输出，几何状态不参与参数学习。

动作协议为 `anchored_joint_interface_least_squares_v1`，checkpoint 升为
`anchored_consensus_training_v3` 并强制保存/检查 `action_policy`。旧 v2
torsion-only 权重不能伪装成新协议恢复；历史结果仍属于旧协议。

已通过解析雅可比与自动微分对照（含乱序嵌套键）、相容动作恢复、刚体
等变性、实际 SDPI 刚体更新的有限差分对照、所有动作组梯度，以及真实
consensus 反传测试。真实冻结专家 1AFQ 更新检查同时改变 torsion 头和刚体
头，专家不接收梯度；产物为
`results/anchored_joint_motion_20260920/real_gradient_smoke.json`。这些是模块
正确性证据，不是结构准确率证据。

历史 torsion-only 续训已正常完成至 step 1024，固定开发 loss 为
0.534915716，位于 `results/anchored_training_seed7315_20260920_cont1024/`。
它不含上述联合动作层。

严格约束联合版已在 `results/anchored_joint_training_seed7315_20260920/`
完成 1,024 次更新，固定开发 loss 从 0.576743669 降至 0.541709983。它的
动作转换不同于 torsion-only，不能据此宣称结构收益；尚未完成候选池评估。

### 有界二元偏离模块

`bounded_binary_flow.py` 在 K 个 torsion 外加入两侧各 6 个偏离变量。
三维变量通过径向映射进入声明的平移/旋转范数球；每步都从原参考重建，
不是在上一帧叠加允许误差。平移上限指结合片段质心，不是远处的蛋白质心。
E3 侧偏离移动配体/POI 整体，POI 侧偏离在随结合端输运的参考系中移动 POI。
不会改变配体键长、受保护片段内部形状或蛋白内部结构。

`bounded_joint_motion.py` 同时求解 torsion 和偏离速度；动作协议为
`bounded_joint_interface_least_squares_v2`。偏离速度采用 1e-4 数值阻尼，
不能像归一化刚体输出一样加单位大小惩罚，否则会严重压制亚埃级动作。
早期 v1 单更新烟测仅为开发记录，不是当前策略。训练 loss 显式区分 torsion
角速度、偏离 latent 速度和实际笛卡尔速度；不同任务总 loss 不可直接比较。

`bounded_binary_training.py` 先用声明半径一半的范围扰动训练模板，重新构造
side-local 条件，再编码原训练 native 为可达目标。目标偏离非零，避免网络
只学会输出零修正。checkpoint 保存两侧 `binary_bounds`、独立 flow/action/
training 协议；resume 禁止改变范围。范围外目标和内部形状不匹配仍拒绝。
该增强只覆盖刚体模板差异，不声称消除了独立模板的全部分布差异。

当前工程检查声明两侧平移 0.5 Å、旋转 0.2 rad；并非经实验校准的普适
可信区间，也不根据测试 DockQ 调整。已通过各方向自动微分对照、全局坐标
等变性、目标往返编码、连续 100 步不累积放宽范围、零范围退化到严格版、
新目标非零及权重/优化器恢复隔离。真实专家反传及带固定开发验证的两步
保存/恢复位于 `results/bounded_joint_contract_20260920*`，均为 smoke。

训练入口新增显式 `--warm-start`：只迁移已训练的 EMA 网络权重，检查网络和
专家身份，重建新任务的优化器、验证和步数。与 `--resume` 的完整恢复严格
区分，checkpoint parent hash 指向实际迁移来源。

当前有界版单 seed 训练位于
`results/bounded_joint_training_seed7315_20260920/`，从严格联合版 step 1024
EMA 迁移，物理 GPU 3 前台运行。两侧固定 0.5 Å / 0.2 rad，不做容差扫描。
完成状态与选中权重以该目录日志和 best 文件为准。

采样入口支持 `--n-walkers`：同一个根 seed 的 RNG 连续抽取独立参数初态，
共用一个已验证的二元参考，不新增训练 seed。这个协议尚不覆盖多个参考
内部几何族，不能冒充旧全池 256 候选预算。全部候选生成并落盘后，独立
评分入口才读取 native，分别输出逐候选 DockQ、池内 oracle 和均值；不计算
模型排序 top-1，不把第一个 walker 当作 top-1。两 walker GPU 接口检查用
smoke 权重完成，不能作为准确率结果。

局部实现进度：`tricoopdiff/anchored_torsion.py` 已提供桥键约束编译与可微
torsion 增量更新，复用现有旋转键列表；锚端不动，另一端所在分量移动时同步
旋转 POI 并累积其 body frame。只保证保持输入时已有的二元几何，不修复初态。
`tests/test_anchored_torsion.py` 已验证多键更新下结合距离/键长保持、刚体等变性、
POI body frame、自微分与有限差分一致及非法环/不连通输入。

后续局部进度：`anchored_flow.py` 实现参考构象上的周期角度路径、反向角度积分、
统一解码以及 ligand/POI 的 `J_D v` 笛卡尔速度目标。`consensus_velocity` 按原
torsion 索引读取现有逐键输出，保留其 pi 单位归一化，只使用允许运动的键。
`tests/test_anchored_flow.py` 验证跨 ±pi 的路径/积分一致性、每步二元几何保持、
时间有限差分与笛卡尔速度一致，以及损失梯度仅进入允许的逐键输出。
`encode_target` 已提供目标角度恢复及 ligand/POI 端点重建检查；不可达目标明确
报错，不通过全局对齐或最近点投影偷换监督。`anchored_target_loss` 在模型前向
之前执行此检查。`anchored_training.py` 已将显式片段/atom maps、本侧 native
二元监督、随机参考构象、独立均匀角度先验和目标恢复封装成训练样本；未映射
原子仍为 unknown，不从接触分配化学身份。条件距离编码与部署复用同一实现，
但训练专用入口允许 `training_native_supervision`，部署入口仍明确拒绝。
固定内部几何来自 native 训练构象的分布，与独立 binary 装配仍有差异，审计中
明确标注。已有化学切分增强、代表样本审计和有界训练入口；全量数据修复、
真实片段身份、单侧条件及不确定先验策略仍未完成。

`anchored_consensus.py` 复用现有 consensus/torsion head，联合逐键速度与 ligand/POI
笛卡尔速度损失，`sample_anchored` 使用同一解码器积分。完整双侧路径不积分旧
刚体输出。source=0 复用旧 source embedding；新 checkpoint 已另立生成过程/
条件/source 合同，不能伪装成旧 SDPI 训练。烟测与阶段训练权重的 scope 分开记录。

部署参考通过既有 `BinaryPoseInitializer` 构建，不读取 ternary native；新增选中
atom maps、两侧拟合误差与 E3 坐标系拟合审计。新 chart 拒绝非刚性蛋白坐标
不匹配，并以 RDKit 键长 bounds 加显式 0.15 Å 容差检查共价键；这不是完整能量、
键角或立体化学验收。修正了 initializer 的 body rotation：从当前 POI 坐标系
映射到装配后的坐标系，而不是默认当前输入与二元资产的坐标系相同。

真实专家烟测入口为 `scripts/smoke_anchored_flow.py`，物理 GPU 3 前台运行，固定
seed=7315、2 个初始化候选、1 次 backward、3 步采样。使用 v8 epoch10 仅检验集成，
零角度合成目标不作为准确率证据。结果
`results/anchored_flow_smoke_20260920/5T35_target_checked.json`：5T35 的 25 个原始
torsion 中允许 11 个；冻结专家无梯度，torsion head 梯度非零；锚端位移 0，
另一端与 POI 距离最大漂移约 6.5e-5 Å。初态有蛋白/配体碰撞，尚不证明生成质量。
单侧条件、不确定二元几何残差与正式端到端训练仍未完成。

目标是让已知 binary binding geometry 决定生成状态的可用自由度。
v8 的 side pooling 仍未改变自由度、初始化或 flow coupling，不能视为完成此目标。
v8 训练与 epoch10 correct 评测已人工停止；中途指标不构成最终验收。

### 已核实的参考实现边界

原仓库 `DeepTernary/deepternary/configs/protac.py` 选择
`IEGMN_QueryDecoder_KnownPocket_GtVal`，配置 `lig1_coords_mode=pocket`、
`lig2_coords_mode=aligned_pocket`、训练 `lig_kpt_mode_train=mask_pocket`、
验证 `lig_kpt_mode_val=mask_predict`，两阶段 receptor keypoints 均为 `mask_pocket`。
`models/iegmn_gtval.py` 此类 forward 中逐层 pocket 坐标回填被注释。
其他 decoder 确有重复回填，不能据此宣称默认 PROTAC 路径每层硬投影。
可以借鉴的实质是直接使用局部二元坐标和已知对应关系组织初始化与装配，
不是假定所有类、训练与验证分支具有相同行为。

### 状态、解码与化学合同

记固定 E3 坐标系为全局系，完整配体坐标为 L，POI 为 P。
条件包含各自蛋白局部系中的完整映射片段、有效 atom map 假设和可信度；
绝不包含两个蛋白的 native 相对变换。一个轨迹固定一个合法映射/pose 假设，
避免逐步切换离散映射造成速度不连续。

- 双侧完整：状态主要为可动 torsion 向量 q；解码 D(q;c) 同时输出 L、P。
  第一端与 E3 保持给定结合关系；第二端与 POI 作为一起运动的几何单元。
- 单侧完整：以已知侧作锚，未知蛋白保留独立 SE(3) 状态；若已知侧是 POI，
  内部可交换锚定侧，输出再转换回统一 E3 坐标系和蛋白身份。
- 无完整二元姿态：保留通用刚体加 torsion 路径。仅 pocket mask 不足以固定姿态。
- 不确定条件：以后扩展局部位姿残差及其训练先验；不能将置信度未经校准直接
  当作角度/距离方差。首个实现明确区分可固定和不可固定的条件。

先构建化学图的刚性片段与可旋转连接关系。切断候选旋转键后，检查两端
映射原子是否各自整体位于某个连通分量；会拆开受保护片段的键不能成为
该轨迹自由度。连接键可以活动，前提是完整 warhead 及其蛋白随同一变换运动。
采用锚定片段为根的旋转方向与拓扑顺序，不能沿用 vendored 的最小连通分量
方向并默认为根保持不动。环闭合和跨片段多连接需要专门的约束处理，未支持
时明确拒绝该受约束路径，不能随意断环。

未映射原子不等于 linker；接触优势不明确也不等于 linker。v8 的此类标签
是启发式 pseudo 标注，不作为新化学分解依据。缺少片段信息保留 unknown。
初始完整配体必须与局部 warhead 内部几何兼容；刚体拟合只能消除位姿差异，
不能消除键长/键角差异。独立片段坐标直接覆盖后可能破坏连接键，必须通过
约束构象构建并检查连接几何，不能声称 Kabsch 自动保证两端完全满足。

### 训练路径与速度

对固定片段几何、映射和拓扑，采用周期 torsion 路径：
`delta = wrap(q_prior - q_data)`，`q(t) = q_data + t * delta`，
时间沿用 clean=0、prior=1；目标为 `dq/dt = delta`。
生成状态与 Cartesian 目标分别为 `X(t)=D(q(t);c)`、`dX/dt=J_D(q(t);c)*delta`。
反向一步为 `q <- wrap(q - dt*v_q)`，随后统一解码；不能调用旧 SDPI 的独立
双界面刚体 step 或整条配体 Kabsch 去刚体规范。对于未知侧 SE(3)，其状态与
目标另外保留，仍通过同一解码定义 Cartesian 运动。

训练与部署使用相同的允许自由度、坐标解码和先验采样器。现有 SDPI/BiTSeed
缓存的 coupling 不满足新合同，不能原样复用。条件 dropout 必须先于运动路径
构建：去掉一侧后，自由度和先验一起改变，不能只清零特征而保持旧 target。
训练数据的 native 只能用于本侧监督条件和目标；跨蛋白 native 关系不能进入
条件。每个样本需验证 `D(q_data;c)` 能否达到目标；若固定片段、环构象或键几何
导致不可达，记录误差并采用明确的数据适用策略，不能让网络拟合不存在的解。

### 与现有模型的连接

保留冻结 DiffDock 和可学习 consensus 的表征能力。新增按可动键输出的速度头，
输入包括该键两侧化学/条件信息、当前三类界面消息、时间，以及对应运动对
ligand 和整个移动蛋白的影响。输出直接在允许自由度中积分。
旧 expert score 经过现有预条件化也不是机械力；只能作为建议特征或明确学习的
速度参数化。若使用 `J^T W v` 作为输入，它是投影相关特征；若要解释为广义
速度，应求解带正则的 `(J^T W J + lambda I) dq = J^T W v`，不能混淆两者。
权重需按分子/界面归一化，避免大蛋白仅凭节点数支配更新。

### 实现顺序及完成门槛

1. 化学约束编译器：映射、受保护片段、合法 torsion、锚定根、离散假设。
2. 可微解码器：ligand 和 POI 同步运动；先验证结合关系、连接键、刚体等变性，
   再用有限差分验证 linker 和 POI 的 Jacobian。
3. 条件 flow：统一训练路径/采样 step，验证端点可达、无噪声目标积分及
   单侧/无侧情况；独立版本化 checkpoint，禁止静默加载旧优化器或 coupling。
4. 接入键级速度头与冻结专家，检查真实反向传播后才进行单 seed 正式训练。

这是一份待实现设计，不代表已证明提高 DockQ，也不承诺硬约束适合所有模板。

## v8 side-aware linker 模型（2026-09-20）

`conditioned_residual_consensus_v8_side_aware_linker` 从同一 C40 EMA 主干迁移，
继续冻结两个 DiffDock expert。它不再把两侧条件直接平均：E3 end、POI end 和
linker 分别汇聚，保留侧别、缺失掩码及两端乘积/差异，再生成联合条件上下文。
侧别上下文同时注入对应蛋白节点和 warhead 原子；linker 上下文只注入 linker
原子。联合上下文驱动 consensus gate、刚体速度和 torsion 残差。

训练监督改为互斥 pseudo-warhead 合同：按原子到两侧蛋白的局部接触优势分配
两端，模糊或 dilation 冲突区域作为 linker；本侧 pocket 和 binary distance map
只由本侧 end 定义。这样避免旧合同中一个原子同时充当两端，以及整条配体定义
两侧 pocket 的训练—部署错配。新 checkpoint 与 v7 不兼容，必须从 C40 重新训练。

## 分支与职责

### 当前实现更新：二元姿态参与自主生成（2026-09-17）

`conditioned_residual_consensus_v7_interface_geometry` 已接入逐步训练和采样。
在原 role/pocket/radial 输入之外，`binary1/binary2` 分别保存
`[N_maps, N_ligand, N_protein, 3]` 的距离、距离副本和置信度。蛋白节点顺序保留；
只计算各自二元复合物内部距离，没有跨蛋白的真实三元距离。
化学对称映射保留完整原子对应，以当前候选的整体距离残差选择每侧一个合法映射；
不再使用逐对独立区间，避免接受任何合法映射都无法解释的混合距离。
映射按化学原子序规范排序，重复映射去重，不读取真实三元结构进行选择。

`binary_geometry.py` 在当前候选上重算约束偏差，编码配体—残基消息，并沿
配体化学键传播。两侧的恢复方向和力矩通过可学习系数进入刚体 velocity；
两端约束还会投影到 sampler 的实际可旋转键及 moving-atom mask，直接进入
共享 torsion 更新。linker 构象沿原有 torsion 自由度更新，保留既有化学拓扑。
`structure_velocity.py` 统一 torsion 轴方向与 Kabsch 去刚体分量，并给出
与实际 sampler 一致的笛卡尔运动切向量。结构损失匹配同一 flow 状态/目标下
的原子运动及本侧二元距离变化率；不把带噪中间态直接强制拉到终态距离。
`--structure-loss-weight` 默认0.1，并记录/校验于checkpoint，恢复时不得静默改变。

`interface_geometry.py` 补充当前E3–POI、E3–ligand、POI–ligand界面消息，
复用现有蛋白/配体化学特征编码。边包含12 Å内的全部节点对和每个节点的
最近邻，以覆盖碰撞位置及分离状态；距离RBF、粗粒度重叠特征和化学信息
生成局部消息与方向，进入刚体和共享torsion输出。该通道不读取真实界面。
相应结构损失监督三个界面的距离变化率，目标仍为同一flow coupling的运动。
这一通道在二元条件缺失时仍可读取当前几何；只有二元条件通道随dropout清除。

训练数据构建、条件 dropout、manifest 加载、checkpoint 和训练/采样入口均已
连接。缺失二元条件会清除全部对应距离约束。冻结 DiffDock 保持原输入通道。
新架构从 C 模型迁移，门控零初始化；旧径向特征 checkpoint 无法直接加载为
新架构。新模块需要训练，旧实验指标不代表本架构的精度。

实现验收使用 `tests/test_binary_geometry.py` 和
`scripts/smoke_binary_geometry.py`；后者直接运行 SDPI/BiTSeed，记录实际来源
与 fallback，不读取真实三元结构评分，不需要 DeepTernary 预测结果。
当前训练链路烟测位于 `results/interface_geometry_build_smoke_20260917/`。
`tests/test_structure_velocity.py` 核对完整映射选择、采样器有限差分及刚体/torsion梯度。
距离约束在蛋白Cα节点分辨率工作，不等价于全重原子碰撞模型；独立top-1质量正在
PROTAC22上评估。两个训练seed已完成160轮，并分别按内部验证选定
epoch77与epoch41。v5/v6烟测目录保留为历史记录，权重不能直接载入v7。
full/pocket-only/absent结构损失包含不同数量的条件约束项，因此这些loss不能
直接相减解释为条件带来的预测收益。

- 分支：`protac`
- worktree：`/mnt/disk/wl/MoRE-tri/CoopDiff-protac`
- 基线：`model-v3` 的合并提交 `9002dda`
- 唯一主问题：**哪些 native-free、部署时真实可得的 PROTAC 信息，能够提高 TriCoopDiff 精修后的 DockQ？**

本分支负责条件资产、条件编码、训练—部署一致性和结构级验证。候选池排序、256→64 selector、post-ranker 与 walker 间排序通信属于 `rank` 分支，不在这里并行修改。需要共享接口时先写清张量或 manifest 合同，再协调合并。

## 已知结论，不要重复

当前候选与精修基线为 Hybrid-56/8、64 walkers、20步、seed 31000，父模型为 C40 EMA。PROTAC22 上三组严格配对结果为：

| 条件训练 / 推理输入 | Final | Union | Final ≥0.23/0.49 |
| --- | ---: | ---: | --- |
| default / pocket | 0.3066 | 0.3289 | 15/1 |
| default / absent | 0.3244 | 0.3374 | 17/2 |
| control / absent | 0.2936 | 0.3175 | 16/0 |

同一权重下 `pocket - absent` 的 Final 为 `-0.0178`（9胜/13负）。完整256池的 case-best 也没有改善高质量尾部。因此：

1. 6 Å Cα pocket-only mask **不能作为默认条件**。
2. default-absent 相对 control 的提升只能说明训练方式可能有益，不能声称模型利用了 pocket。
3. PROTAC22 中严格双侧 warhead 化学映射为 0/22；禁止用坐标最近邻或宽松子图结果伪造 atom map。
4. v2 pocket manifest 有修饰残基节点错位，已废弃；只能使用与 DiffDock/ProDy 节点顺序一致的 v3 资产。

完整证据见 `todo.md`、`Q.md` 和 `RANKING_PROTAC_RESEARCH.md`。不要从失败或 partial 目录拼接结果。

## 当前后台作业

原仓库 `/mnt/disk/wl/MoRE-tri/CoopDiff` 正在运行 P90/P100 多 seed 的训练与配对评估，使用封存 runtime。它们检验 pocket-only 训练—部署匹配，不代表新方案。不要停止、覆盖或从本 worktree 重启这些任务；等统一审计报告落盘后只补充结论。

生成数据、checkpoint 与缓存仍位于原仓库的 `data/`、`results/`、`checkpoints/`。本 worktree 不复制大文件；需要时使用绝对路径，并把新输出写入带协议和日期的独立目录。

## 下一阶段优先级

### P0：恢复可信的两端角色

先解决 warhead/靶蛋白侧的身份问题，再训练模型。优先使用有明确化学来源的片段 SMILES、键级/电荷修复和人工不可知的确定性消歧规则。任何 atom map 必须验证 canonical ligand identity、原子数、立体化学、唯一子图、连接键和两侧来源。无法唯一确认时标记 missing，不猜测。

### P1：同侧局部二元几何

在每个 protein–warhead 自己的局部坐标系中编码信息，例如：

- warhead 原子角色、linker attachment bond 与 linker atom mask；
- protein–fragment 距离/RBF、接触和方向性局部边；
- DiffDock 置信度或多姿态不确定性；
- 条件存在性、来源和可信度 mask。

不得输入 native ternary 坐标、两个蛋白的真实相对变换或由其派生的跨蛋白距离。条件应对每侧共同刚体变换保持不变或等变。

### P2：先做“信息是否被使用”的小实验

在正式长训前至少比较正确条件、全缺失条件、侧别交换、错误配对条件和几何扰动条件。要求同 checkpoint、同候选、同初态和同 diffusion seed。若正确条件与错误或缺失条件没有可重复差异，不进入大规模训练。

## 实验门槛

所有结构实验必须报告 Initial、Final、Union，以及 ≥0.23/0.49；同时给出逐例胜/平/负。主要因果量是**同一权重下正确条件减 absent**，而不是新权重减旧权重。进入默认协议至少要求：

- 全部预注册 case 完成，不能只统计成功子集；
- proposal pool hash、selected candidate ID、初态坐标/DockQ 和随机种子完全一致；
- Final 均值提高，Final ≥0.23 不减少，胜数不少于负数；
- 至少两个训练 seed 方向一致；
- 条件打乱或侧别交换后收益明显下降。

PROTAC22 是开发集，只能用于机制筛选；最终结论需要独立的 PROTAC holdout。case-best/oracle 只能判断上限，不能写成部署 top-1。

## 代码入口与检查

优先复用：

- `tricoopdiff/protac_conditions.py`：化学映射与本侧口袋审计；
- `tricoopdiff/condition_manifest.py`：严格部署 manifest；
- `tricoopdiff/condition_features.py`：条件张量；
- `tricoopdiff/conditional_consensus.py`：条件融合；
- `scripts/train_conditioned_entry.py`：配对训练；
- `scripts/evaluate_selected_refinement.py`：固定候选结构评估。

最小回归：

```bash
python tests/test_protac_conditions.py
python tests/test_condition_manifest.py
python tests/test_condition_features.py
python tests/test_conditioned_training.py
python tests/test_selected_refinement_contract.py
```

运行入口必须先导入 `tricoopdiff.diffdock_compat`。冻结两个 DiffDock 专家，不恢复 LoRA，不引入 SLURM，不让测试自动下载模型。提交前记录命令、checkpoint SHA256、manifest 指纹、数据 split 和输出路径。

## 首个交付物

先提交一份“可用信息资产审计”：逐 case 列出两侧 fragment 来源、严格映射状态、attachment/linker 定义、局部二元几何是否可用及失败原因。只有审计证明存在足够覆盖率后，才实现新 encoder；否则应明确报告数据缺口，而不是用 pocket-only 重跑替代。

## v3 二元先验实现（2026-09-17）

已实现 DeepTernary 同类的高可信输入链，但不复制其坐标近邻原子映射：

- `prepare_protac_binary_chemistry.py` 可从 CCD 恢复实际观测的裁剪片段，并按严格化学、仅电荷标准化、保守 MCS 分级记录角色映射。MCS 仅用于审计，不进入首轮部署 manifest。
- PROTAC22 实际审计恢复 44/44 侧；34 侧角色原子集合明确，11 例双侧不重叠。高可信门槛下 27/44 侧可用、7/22 例双侧完整。
- `make_protac_binary_manifests.py` 只接纳严格或仅质子化差异的映射，输出两端原子角色、linker、两侧口袋和各自 protein–fragment 二元坐标。生成和加载均不读取 ternary native。
- 条件 v3 将单一最近距离升级为 ligand/protein 双向径向距离轮廓；所有对称等价 atom map 共同平均。该表示对每侧独立刚体变换及化学对称置换不变。
- 新架构标识为 `conditioned_residual_consensus_v4_binary_radial_profile`。旧条件 checkpoint 维度不兼容，必须由同一 C40 parent 重新训练；旧 pocket-only 结果不能作为新模型成绩。

资产生成示例：

```bash
python scripts/prepare_protac_binary_chemistry.py --index <protac22.json> \
  --ccd-dir <ccd_dir> --root /mnt/disk/wl/MoRE-tri/CoopDiff --out <new_audit_dir>
python scripts/make_protac_binary_manifests.py --index <protac22.json> \
  --chemistry-report <new_audit_dir>/report.json \
  --root /mnt/disk/wl/MoRE-tri/CoopDiff --out <new_manifest_dir>
```

下一实验应先在这 7 个双侧 case 做同 checkpoint 的 correct / absent / side-swap / wrong-pair 配对机制检查。超过 DeepTernary 仍需独立结构评估证明，当前实现只补齐输入能力和可信资产合同。
