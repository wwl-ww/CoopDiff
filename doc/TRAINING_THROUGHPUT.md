# 训练吞吐与多卡准备

## 当前状态（2026-10-11）

四项性能改动已集中接入，并完成真实训练评估。当前已保存 update358；v5通过原失败病例所在真实GPU批次，后续358复用已有编译图，无新增重编译日志，耗时261.60秒，完整保存审计通过。后台主训继续20 epochs、global batch4、seed73，GPU0/1。最新日志为 `/root/.local/share/coopdiff/main_training_seed73/resume_compiled_normalized.log`。这里只使用当前容器：32核共享CPU配额无法调整，没有取得独立CPU资源，已落实容器内缓解。

| 优化项 | 已有实际证据 | 适用范围与限制 |
| --- | --- | --- |
| 容器内CPU缓解 | 缓存读取约0.1–0.3秒；正式主训5秒通信等待CPU由0.907秒降至0.0058秒，native栈确认nanosleep | 32核共享硬配额保持；不能在当前容器取得独立CPU预算 |
| 消息算子编译/融合 | 正式128/32宽度跨长度复用；v5主训357/358提交，358无新增重编译、261.60秒，数值有限且两rank参数一致，原metadata失败批次通过 | 不同批次不能直接计算加速比；每区域32上限尚非全数据证明，保留默认metadata检查与显式失败 |
| 静态拓扑与索引复用 | 缓存版本/梯度不变量检查及实际354–358更新通过；坐标和梯度仍实时计算 | 无独立整轮加速比，不将合并收益全部归于缓存 |
| 计算量分片 | 实际1+3及2+2分片、全局样本/RNG/加权梯度/保存游标正确；358一例609节点对三例322/145/392节点，等待52.61秒、梯度归约约0.010秒 | 基于计算量估计而非精确耗时；不可拆大样本、目标投影和共享CPU仍可能造成等待 |

显存保护、冻结专家、数据身份校验、每update完整保存继续保留。357包含v5首次编译耗时758.09秒，不能视为稳定update速度。历史上356曾触发默认编译上限8，修复后356已提交；357的metadata失败修复后也已提交，未跳过更新。下文记录历史检查和现场变化，以当前日志、manifest和检查点为准。

2026-10-10历史范围：当时只授权短测，不自动启动400-update训练；后续授权和运行状态见上。

## 单卡

当前 `--batch_size` 是顺序 microbatch 的平均梯度累积；增加它不会让多个复合物同时进入模型。装配模块明确要求每次输入一个节点对齐的复合物，不能直接把坐标 padding 成 batch。真正的图批处理仍需另行实现并检查跨图边、扭转索引和逐图损失。

`--no-checkpoint-ternary` 保留更多三元主干激活，避免其反向重计算；冻结二元专家的输入梯度和专家重计算仍保持原实现。专家权重继续冻结。不要通过关闭输入梯度、减少积分步数或改变损失来声称同等训练提速。

首轮完整试跑：开启重计算更新 148.64 s、峰值 allocated 1.07 GiB；关闭为 101.08 s、7.74 GiB。这是不同共享 GPU 的首个更新，含编译影响，不能直接作为严格加速比。配对基准入口：

```bash
source results/deployment/activate.sh
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_training_throughput.py \
  --checkpoint-mode on --output-dir results/throughput_on
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_training_throughput.py \
  --checkpoint-mode off --output-dir results/throughput_off
```

两模式分别从 seed=73 新模型开始，固定 8 例、有效 batch=4、20 步展开，默认预热一次再计时两次；不运行 DockQ。以预热后的样本/秒、同步墙时和 allocated/reserved 峰值比较，整体 GPU 占用包含其他任务，不能视为本进程显存。

实际配对短测记录：开启重计算完成 1 次预热和 2 次计时，计时平均 586.77 s/update、0.006817 sample/s，峰值 allocated 1.875 GiB。关闭重计算的新模型初始可训练参数 SHA 与开启模式完全一致，但其预热在共享 GPU/系统负载下超过 40 分钟仍未完成，已仅终止本实验进程；状态为 `aborted_environment_contention`。当时本进程占 7.70 GB 左右，另外两个进程占约 7.34/15.55 GB，CPU 时间主要消耗在内核。缺少关闭模式的完成更新，因此不能得出配对加速比或更新后数值一致性结论；结果和负载记录保存在 `results/small_sample_seed73/paired_summary.json`。

## 多卡

`scripts/train_distributed.py` 采用每 GPU 一个进程和显式梯度 all-reduce；不是包装标准 DDP 后直接并行整个装配模型。先对各卡局部平均梯度按样本数加权求全局平均，再做一次全局范数裁剪、优化器更新和 EMA 更新。

全局有效 batch 保持一致时，各卡分担样本；每卡复制冻结专家，不能指望模型显存自动分片。near-target 的样本序号需要全局偏移，`samples_seen` 按全局样本数推进。只有 rank 0 写公共 checkpoint，所有 rank 的 RNG 和运行拓扑须记录。入口是否支持分布式续跑必须明确声明，单卡 checkpoint 不能自动当作多卡可精确恢复的证据。

验收顺序：CPU/Gloo 梯度与计数契约、失败传播测试 → 双 GPU 单次真实更新 → 同全局 batch 单卡/双卡预热后吞吐比较。未完成实际验收前，不宣称线性加速或生产多卡训练已就绪。

示例（需已经预处理的完整索引和可用 GPU；全局 batch=4，两卡各2例；仅一次更新）：

```bash
source results/deployment/activate.sh
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --standalone \
  --nproc-per-node=2 scripts/train_distributed.py \
  --train_data results/small_sample_seed73/train.json \
  --val_data results/small_sample_seed73/same_complex_val.json \
  --output_dir results/distributed_short_check \
  --cache_dir results/featurize_cache --expert-backend matcha \
  --training-arm A1 --seed 73 --batch_size 4 --lr .0001 \
  --no-random-flip-proteins --no-checkpoint-ternary --max-updates 1
```

当前入口只训练，不运行验证；必须另行验证检查点，不能从 loss 或吞吐推断 DockQ。只支持 A1，拒绝缓存过滤和不完整全局尾批，不静默丢弃或复制样本。性能测量的更新墙时不包含特征装载和保存，需同时观察端到端耗时。

### 断点续训

使用相同训练参数，添加 `--resume <checkpoint_updateN.pt>`、新的输出目录和累计目标 `--max-updates M`。例如上例的 checkpoint_update1.pt 加 `--max-updates 2` 只再运行一次更新。

恢复模型、Adam 状态、scheduler、EMA、global_step、samples_seen、每个 rank 的随机状态以及下一批数据游标。当前要求 world size、全局 batch、seed、数据索引和采样配置匹配；换物理 GPU 可以，改变卡数需另做迁移协议。新 checkpoint 记录索引 SHA，标记 `distributed_training`，标准单卡入口拒绝误恢复。

最早的多卡 smoke artifact 虽标记 inference-only，仍保留完整优化器及每卡 RNG，可以由多卡入口结合旁边的 manifest 恢复；它没有保存数据索引历史 SHA，只能校验原索引路径，因此记录 `legacy_dataset_identity=path_only`。新保存的断点使用内容哈希校验。

真实双卡续跑已完成：从 update1 恢复到 update2，只新增一次更新，`samples_seen=8`，下一批游标为 epoch1/offset0；Adam step=2，模型及 EMA 更新，两卡参数 SHA 完全一致，新断点 `resume_supported=true`。该次更新 100.71 s；结果记录在 `results/small_sample_seed73/distributed_resume_summary.json`。针对性 CPU 检查也验证恢复后的下一次更新与不中断运行一致，以及错误拓扑/单卡入口拒绝恢复；未额外启动长训练。

已完成 CPU/Gloo 加权梯度、缺失梯度、失败传播和全局 near-target 调度检查；最终训练契约模块 73 项通过。真实 GPU 1+2 的最终一次更新耗时 50.29 s，全局样本数 4，显存峰值 6.47/7.75 GiB；两 rank 更新后参数 SHA 完全一致，检查点保存两份 RNG。结果：`results/small_sample_seed73/distributed_smoke_summary.json`。此轮使用生产默认 lr=.001，仅用于集成验收，不能与 lr=.0001 的研究实验比较指标，也不是预热后的吞吐基准。

## E 研究入口审查修复

2026-10-10 的 E20 研究已中止并标无效：32 个预测中心只有 8 个通过原有非 native 接触几何条件，24 个不合格。生成入口误把有限候选等同有效中心，现生成和多卡加载两端均失败即停止，加载端从实际缓存坐标重算，不依赖标签。第5次更新另因 .2 人为显存上限退出（物理卡仍有45.67 GiB空闲），该批次包含 main20+near20；共完成4次更新，只有 checkpoint0。未来有效库运行配置改为 .4 上限及每次更新保存；尚未重启。

评分审查未见公共坐标系或原子顺序错误；1XK4_F_J_FLC 原生两蛋白没有5 Å接触，自评分2/3。评估入口保留所有案例并分层报告零接触样本，不改变 DockQ 公式。证据见本地研究目录的 failure_report.json、center_bank_audit.json、debug_dockq/。


## 2026-10-10 全量主训练与显存预算调整

全量主训练使用完整6834例/587clusters训练索引、253例/65clusters验证索引，fresh seed73可训练权重与冻结预训练Matcha；结构起点在线重采SDPI，global batch4，GPU1/3，20轮，每轮147更新。正式入口`--run-epochs`具备轮末EMA验证与完整续训；上述旧“仅训练”描述仅适用于`--max-updates`研究模式。

正式首更新153.79秒，两rank峰值allocated2.215/1.331GiB，检查点参数哈希一致。已从update1严格恢复，将`--memory-fraction`由0.4提高为0.75（约59.7GiB/进程），保留显存余量供外部缓存writer及同机进程；此参数是分配上限，不会主动填满显存。

新增`--ternary-checkpoint-min-nodes`运行时策略，默认0保持所有样本重计算；例如600让节点数小于600的样本直接计算，较大样本继续重计算，以显存换取吞吐。数学模型与`checkpoint_ternary=True`续训契约不变，阈值单独记录在manifest的memory_runtime_policy。真实255节点20步＋near更新配对通过：阈值0耗时47.98秒、峰值0.807GiB；阈值600耗时38.87秒、峰值3.332GiB，更新后参数逐张量完全相同。共享GPU负载及首次编译会影响计时，不能外推整个数据集加速比例。

`--ternary-checkpoint-reserve-gib 8`另启动态保护：小样本也必须有足够余量覆盖8GiB保留量＋114688×节点数平方字节的保守激活估计；余量取物理空闲＋可复用allocator缓存与进程预算剩余的较小值。不足即强制重计算。节点数≥600始终重计算。估计并非OOM保证，发生失败时沿既有两rank错误协调中止，清除未提交梯度并恢复RNG，不静默跳过样本或重放已同步梯度。已从update5严格恢复并实际启用该保护（PID391297），manifest确认initial_global_step=5及上述runtime策略；每次更新保存checkpoint，减少异常后的重跑量。训练运行在独立session、无控制终端，stdin为/dev/null，输出落盘；后台监控将健康状态写入运行根目录background_status.json，关闭VS Code不依赖终端存活。

实际命令、日志和检查点位于`/root/.local/share/coopdiff/main_training_seed73/`，正式恢复使用static rendezvous，不使用`--standalone`动态store。缓存由独立writer准备完整7087例，训练rank等待身份匹配的feature/Matcha metadata，不进行cached-only人口过滤。


### 2026-10-11 后台运行复核

主训练PID649331在GPU0/1独立session继续，已验证update350检查点（1398样本，epoch2/offset224），两rank参数哈希一致，优化器/EMA/两rank RNG齐全。epoch1/2均已真实执行EMA验证、scheduler及best保存，PP DockQ分别0.045351/0.050801，selection loss分别125.9485/112.3154。完整7087例feature与Matcha metadata均准备完成、0失败。启动与续训目标已达成，20轮训练仍在后台进行。复核证据：运行根目录launch_completion_audit.json。

性能尚未解决：updates348/349/350分别723.49/641.31/397.70秒，rank等待分别可达442.5/239.4/126.2秒；不能沿用早期约99秒/update或81小时总时长估算。换卡已执行，不代表长期吞吐问题解决。


### 2026-10-11 共享CPU限流与query分块优化

容器cpu.max=3200000/100000，只有32核总预算。5秒窗口CPU用量160.22核秒，50/50周期被throttle，累计被限流673.34核秒；另有两进程约1369%/1282%CPU。独立2秒复核21/21周期限流。此时我方GPU SM利用仅数个百分点，native栈有大量小算子/上下文管理/分配及反向等待。不能将GPU显存余量或宿主192核数视为足够计算资源。没有改他人任务或共享配额；仅提升自有进程nice被系统拒绝。

新增runtime `--ternary-query-chunk-size 256`（默认64兼容旧行为）：节点数<600时使用大块，≥600固定64，内层重计算保留。CPU forward/gradient与大图回退测试通过；GPU255节点、width64/channels16的单块测试forward+重算_messages调用8→2，output最大差4.77e-7、gradient最大差3.73e-9，峰值95.55→183.47MiB。该测试不是完整Matcha rollout，不能外推整训练吞吐或完整峰值。正式切换还将OMP/MKL/OpenBLAS设1并使用passive等待。

并行选择：目前模型/单样本可放入单卡，保留双卡数据并行。模型或流水线并行仍需CPU发射并增加传输/同步，无法消除32核共享硬配额；独立双训练改变单模型主训目标。cost-balanced分片可能减少等待，但必须保留样本全局near-target索引及rank RNG契约，尚未直接重排。证据在运行根目录cpu_quota_bottleneck.json及results/deployment/runtime_bottleneck/。


### 2026-10-11 消除只读缓存重复ESM哈希（已上线）

DDP rank0启动时完整内容哈希ESM权重一次，广播并在manifest记录run固定编码器身份。只读train/val/probe读取路径传递expected_esm_encoder_identity，每例仍验证缓存身份和源结构内容，不再重新读取2.5GB encoder权重。普通可写featurization仍保留原身份计算逻辑；有固定身份时不接受未验证legacy缓存。

实际后台PID3373552从update352恢复，GPU0/1、global4及显存/续训保护不变。首个实际训练批次两rank加载分别0.3037秒/0.0975秒，此前约30–33秒，已进入update353计算。唯一针对性回归测试通过（连续读取不再hash、错误身份/变更源结构被拒绝）；未追加long smoke。代码与证据在pinned_esm_source_snapshot和pinned_esm_completion_audit.json。CPU共享限流仍存在，不据首次加载结果声称整轮提速。


### 2026-10-11 当前容器内的三项计算优化

用户限定只使用当前容器。cgroup只读，缺少CAP_SYS_NICE，未找到可用资源控制入口，不能取得独立CPU预算；维持32核共享配额，通过减少小算子发射、静态处理和rank等待缓解。没有修改其他任务。

局部编译仅包裹EquivariantInteraction._messages，使用真实Inductor、fullgraph=True、dynamic=True；无静默eager回退，不改变TF32精度策略。255节点、width64/channels16的GPU验收：输出最大差4.77e-7，全部输入/参数梯度最大差3.73e-9。冷编译361.60秒，进程内复用无新增图；新进程磁盘缓存首次调用及再次调用CPU/CUDA RNG均完全不变。该块级测试不能外推正式width128或整update速度。证据：results/deployment/runtime_bottleneck/compile_messages_gpu_smoke.json和compile_messages_rng_smoke.json。

静态复用缓存带storage/view/version键和有界保留：复用身份特征、component索引和配体拓扑/化学路径签名；坐标、坐标梯度及动态属性仍逐步计算。可训练输入特征绕过缓存，原输入或缓存张量in-place变更使其失效，inference无version张量不进入训练复用缓存。

计算量均衡保持原逻辑rank的数据加载、flip、初始化seed和near扰动随机流，再按估计rollout计算量分配物理rank。global batch仍为4；允许1+3等不等样本数分配，梯度按样本数加权，全局near调度索引不变。checkpoint记录分片协议；浮点累加分组改变允许舍入差异。实际正式续训结果另以后台日志和检查点为准，不能由专项检查宣称全量加速。

首正式均衡批次update354实际分片1+3，全局索引1410..1413各一次，重分配0.793秒。该批含2019节点单例，估计最大负载仅下降1.55%，不能期待消除不可拆单例瓶颈。现场另发现继承LD_LIBRARY_PATH尾空项让动态加载器搜索共享项目cwd：同nvcc --version原环境5.197秒超时，去空项0.00414秒完成。activate.sh已过滤继承空项；为使后续compiler子进程生效，授权进行一次环境修复重启，保留Inductor磁盘缓存，从最后已提交353重放未完成354。证据：runtime_bottleneck/compiler_loader_probe.json及运行根目录balanced_compiled_first_batch_audit.json。


正式接入发现编译方案尚不适合生产：环境修复后37分钟仍在main rollout的首次AOT/Triton编译，没有提交update354。已临时关闭compile，从已提交353恢复后台主训，保留成本均衡、静态复用、加载器环境修复和显存/保存保护；日志resume_balanced_eager.log，恢复审计balanced_eager_restart_audit.json。局部编译目标仍未完成，不以专项数值一致性代替正式性能证明。

审查发现两处实际编译问题：dynamic=True同时符号化固定channels，已用compile-only wrapper mark_static固定feature/channel/xyz/bond维；正式width128/ch32/250nodes单块验收341.40秒、121kernels、输出最大差4.77e-7、全部梯度最大差1.86e-9、CPU/CUDA RNG不变，但生成代码节点长度仍具体化，不能证明跨样本复用。三输入einsum调用opt_einsum.contract_path，其contract.py将shape转int；正在以两输入运算替代，并用AOT符号形状及不同节点长度实际复用验证，未完成前不重新开启正式compile。

两输入radial收缩修复已通过CPU AOT直接验收：8个fixture覆盖full/chunk、E3==POI/不等及同模式不同节点/组分长度；总计2个forward和2个backward图，同模式换长度无新增图，输出和全部输入/参数梯度差均为0。AOT输入明确保留coords=[s53,3]、vectors=[s53,4,3]、component ids=[s90]/[s8]等符号维。真实width128/ch32跨shape Inductor验收仍需通过后才接回正式训练；证据message_aot_dynamic_summary.json。

真实Inductor跨shape验收已通过：正式width128/ch32，Q64/N250→Q96/N320，首次231.60秒、1图/110kernels；第二shape零新增图/零新增kernel，0.0067秒。输出最大差4.77e-7、全部输入/参数梯度最大差9.31e-10，首次编译和跨shape复用CPU/CUDA RNG都不变。生成Triton signature含动态ks0参数而非固定节点数。证据inductor_dynamic_cross_shape.json；仅_messages块，不外推整训练提速。

主训练关闭旧编译后已完成真实update354（675.30秒、samples1414），1+3分片/全局索引正确；cp354所有浮点tensor及4例loss有限，两rank参数SHA一致，optimizer/EMA/两logicalrank RNG/cursor齐全，游标epoch2/offset240。恢复审计checkpoint354_contract_audit.json。已从354后台恢复修复后compile，runtime版本two_operand_einsum_fixed_channels_dynamic_nodes_v2、COMPILE_THREADS/MAX_JOBS=4，日志resume_dynamic_compiled.log；实际首新update与复用耗时仍需观察。

通信等待现场另证实CPU空转：等待rank主线程check_error→failed.item→cudaStreamSynchronize，5秒内CPU用时0.917秒、runqueue4.061秒。下一恢复启用TORCH_NCCL_BLOCKING_WAIT=1。与安装torch2.13.0对应的官方[ProcessGroupNCCL源码](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)明确该wait循环每次sleep1ms，并由主线程检查timeout；不启独立watchdog。保留10800秒collective timeout及现有错误协调。该调整减少等候时CPU抢占，不改变容器32核配额，实际CPU改善仍待新进程观测。

正式compile首批现场只读审计：启动约25分钟时update355仍未提交，两rank存活，rank0在符号编译、rank1在真实backward内生成AOT反向。现有生成图同时有Q=N和独立Q/N结构，节点维仍为SymInt；生产vectors stride=(96,1,32)，此前块级GPU验收使用(96,3,1)，验收并未覆盖全部生产布局/grad/alias契约。cache中的12份forward wrapper跨两rank及不同AOT输入签名，不能据文件数认定超过同code的recompile_limit；当前日志没有limit警告。此时GPU0/1利用率约7%/6%，不可声称主训练已提速。继续保留当前进程，等待首次正式提交及复用更新；只读证据live_compile_shape_audit.json。

实际阻塞等待设置单独无效：主训等待5秒CPU0.907秒，旧设置0.917秒，不能声称缓解。精确torch提交的sync all_reduce(async_op=False)返回nullptr，Python不调用Work.wait，随后item()仍CUDAstream自旋；已改梯度/错误协调/报告归约为async_op=True再显式work.wait。独立微型双卡实际3秒延迟比较：原同步CPU0.52627秒，显式Work.wait CPU0.00457秒，结果一致（约115倍减少CPU等待用量）；证据nccl_explicit_work_wait.json。仍需正式恢复验证，保留process-group超时。

修复后动态compile已实际提交update355，samples1418，含首次编译1811.89秒，checkpoint35,748,733bytes。随后356首次调用触发_messages默认recompile_limit=8（lastreason NoneType object has no attribute size），未提交356；已在355保存点停止自有launcher，保留cache与原始日志，正在修复局部compile上下文容量/隔离，未静默回退eager。该故障说明块级跨shape验证不足以证明完整rollout编译契约。

编译上限修复仅作用于compile区域：原生recompile_limit=32/isolate_recompiles=True，保持fullgraph失败显式传播、全局默认不变。结构/grad/alias契约确实需要多图，但现有证据不能将NoneType守卫失败归因于实际输入为None，也不能证明全587cluster永远不超过32；后续生产日志记录recompiles继续检查。针对compile配置/固定维hint/数学等价3项通过；通信修改既有Gloo加权梯度与单rank失败传播检查1项通过。355完整审计通过，cursor(epoch2,offset244)，参数SHA ab7f43359c762b1262e159a5b24bbc80fe210f69aae98bb65bd94e0379811270；保持cache从355后台接续，等待356/357实际保存与通信CPU现场证据。

正式356现场已证实通信等待CPU缓解：rank1完成3例计算/反向和RNGguard后进入check_error，native栈为all_reduce_and_wait → WorkNCCL::wait → nanosleep；5.0003秒内CPU仅0.0057818秒，旧sync路径同窗口0.9074秒，约减少99.36%。runqueue仍3.7837秒，说明共享CPU限流未消失。证据运行根目录explicit_workwait_cpu_audit.json与stack_workwait_rank1_wait_native.txt。另一rank仍生成大样本所需的AOT图，当前重编译日志明确coords requires_grad从0变1的合法契约变化；356尚未保存，不能把CPU等待下降直接当作整训练加速。

NoneType.size守卫机制已最小CPU精确复现：同shape/grad的_messages输入由view变detach后，旧guard中的vectors._base.size()/query_vectors._base.size()访问None；不是模型输入为None。AOT图新增序列view/detach/重复为[1,1,0,0]；同机制offset0/6×view/detach各重复，Dynamo图新增[1,0,1,0,0,0,0,0]，输出/全部输入参数梯度差为0。说明此机制为有限契约切换，不能把每次日志列出的历史None guard算成再次同类失效，也不能推广为全生产上下文上限证明。证据none_size_guard_audit.json及none_size_view_probe.json/none_size_offset_probe.json；未增加clone补丁或为此重启主训。

正式update356已atomic保存，samples1422/cursor(epoch2,offset248)，optimizer/EMA/全部checkpoint浮点及4例loss有限，两rank参数SHA ae2dcfa0ba2540a283b9e734a359d2fb7e526eb2e2089106a1f996bc3afe0d76 一致，Python/NumPy/CPU/CUDA logical-rank RNG完整。冷编译总1842.59秒；rank1等待1556.21秒（采用已证实的CPU休眠路径），梯度归约约0.094秒。峰值allocated rank0大例1.481GiB、rank1三例4.692GiB，未OOM。实际分片1+3，估计最大负载比contiguous下降16.81%，此冷批次不能证明相应实际提速。证据运行根目录checkpoint356_contract_audit.json；不重启，继续357观察生产复用更新。

357实际失败关闭：rank1 5OL2 backward在checkpoint metadata检查发现forward/recompute savedtensor不一致（2576×69 vs2576×72并伴随后续保存顺序变化），两rank协调abort，launcher自然退出，357未提交，356保全。原始日志resume_compiled_workwait.log与update357_failure_audit.json。torch异常建议稳定Dynamo缓存选择；官方[issue166926](https://github.com/pytorch/pytorch/issues/166926)记录多个有效compiled graph的LRU重排序可能让重计算选不同graph。正在保持所有checkpoint检查的前提下修复，不以关闭determinism或eager回退规避。当前编译项仍未完成。

已接入官方checkpoint图选择workaround：仅启用message compile时进程级torch._C._dynamo.eval_frame._set_lru_cache(False)，使用编译插入顺序，关闭单层compile不恢复LRU以免影响尚未反向的checkpoint；manifest v4明确process cache order。API/固定维hint/数学3项通过；CPU交错shape嵌套checkpoint验收与正式恢复并行。后台launcher695937（PPID1/SID独立）从356恢复，新日志resume_compiled_checkpointfix.log；metadata检查、严格RNG、原cache、显存和完整保存全部保留。真实357尚待通过，不能仅凭官方workaround宣称该故障已生产验收。

FIFO workaround尚不充分：CPU嵌套checkpoint交错形状probe在修正纯CPU默认device后，仍真实metadata mismatch（saved[31,31,8] vs recomputed[961,8]），不能称关闭LRU已验证修复。立即终止仅自有695937/700017/700018/695938恢复尝试，全部PID已不存在，357未提交，最新356保全；background_status=stopped_pending_checkpoint_compile_fix。保留cache/log，继续修复view/alias/grad输入与AOT保存契约，而非关闭metadata检查。证据compile_fifo_checkpoint_probe.log及fifo_restart_aborted.json。

更强图契约修复取得同fixture红→绿：FIFO之外，仅compile入口将query_vectors/query_coords/vectors/coords clone(contiguous_format)，规范no_view/offset0/layout并保留梯度，无detach。相同31/41/31/41交错forward再backward的CPU真实block、外内双层checkpoint默认metadata检查通过；2graphs，输出差0、全部输入/参数梯度最大差8.67e-19、RNG不变。编译入口契约/API/数学3项通过。证据compile_fifo_checkpoint_probe.json与同目录_red.log/_green.log；不外推GPU病例已验收。已从356 detached恢复launcher957119（PPID1/SID独立），日志resume_compiled_normalized.log，manifest v5与source snapshot记录规范化契约；保留原cache、显存/保存/严格RNG/metadata全部检查，等待真实357及后续更新。

v5正式update357已完成并保存checkpoint_update357.pt（35,748,733bytes），global_samples1426，日志计时758.09秒。原失败病例5OL2所在批次已通过默认checkpoint metadata检查与训练错误协调；不能将这个包含首次编译的时间当作稳定吞吐。当前继续同一后台进程，完整checkpoint审计及后续实际复用更新计时待补。

最终实际验收：357/358完整审计均通过，358 samples1430、cursor(epoch2,offset256)，两rank参数SHA均为2f7a77c4552426fe40a631da267ff128833e1d2af176b2c3ebb77ffb233d1011；全部checkpoint浮点张量（含optimizer/EMA）及四例loss有限，两logical rank的Python/NumPy/CPU/CUDA RNG齐全，scheduler/数据身份/续训配置仍由完整保存契约保留。证据运行根目录checkpoint357_contract_audit.json、checkpoint358_contract_audit.json与run/distributed_metrics.jsonl。358复用耗时261.596秒，load约0.100秒、redistribution约0.215秒，峰值allocated1.464/4.603GiB，未OOM。launcher957119为PPID1/独立SID，双rank962128/962129和独立monitor957166存活，继续359；显存0.75上限、8GiB余量、≥600节点重算与Q64、小例Q256、数据内容校验和每update atomic完整保存保持。专家源码SHA仍为原基线。此验收证明改动已实际运行，未声称整轮达到5090的80–85秒或显存已占满；共享CPU硬配额仍限制吞吐。
