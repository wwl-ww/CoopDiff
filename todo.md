# 当前工作与交接

更新：2026-09-15。这里仅记录执行状态，不重复协议或历史流水账。

## model-v3 起点与清理范围

本分支从已推送的 `master` 合并提交 `92cf1f0` 创建；尚未实现或训练 model-v3 模型。
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
