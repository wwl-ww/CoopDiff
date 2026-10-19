# 脚本索引

从仓库根目录执行命令。训练、模型诊断及评估需要项目环境、已有缓存和对应 checkpoint；GPU 参数使用本地物理卡号。先用 `--help` 查看参数。

## 常用入口

候选驱动框架使用一个入口 `assembly_pipeline.py`，按 `generate → train → predict`
运行。HDOCK 只生成蛋白位姿，冻结 Matcha 在固定双链受体上生成配体；两个
互为条件的 FiLM–Matcha 供三元主干读取，最终联合修正 POI 刚体与配体刚体/torsion。
运动监督使用可达终点；评分头监督实际终态相对晶体结构的五种几何代理误差，
不会把投影目标上的低 loss 当成晶体恢复。native 只用于训练标签和事后指标，不进入推理。

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/assembly_pipeline.py generate \
  --index data/ternary_train.json --bank results/assembly_train_bank \
  --output results/assembly_generation --candidates 5
CUDA_VISIBLE_DEVICES=4 python scripts/assembly_pipeline.py train \
  --index data/ternary_train.json --bank results/assembly_train_bank \
  --output results/assembly_training --steps 5 --epochs 20
CUDA_VISIBLE_DEVICES=4 python scripts/assembly_pipeline.py train \
  --index data/ternary_train.json --bank results/assembly_train_bank \
  --output results/assembly_training --steps 5 --epochs 20 \
  --resume results/assembly_training/progress.pt
CUDA_VISIBLE_DEVICES=4 python scripts/assembly_pipeline.py predict \
  --index data/ternary_val.json --bank results/assembly_val_bank \
  --output results/assembly_predictions --steps 5 \
  --checkpoint results/assembly_training/progress.pt
```

验证集需先以对应 index 单独 `generate`。必须使用已部署的 HDOCKlite、FFTW、
Matcha 三阶段权重、ESM 和特征环境；`--hdock-dir`/`--fftw-lib-dir` 可覆盖本机路径。
HDOCK 学术许可不允许向实验室之外重新分发二进制，仓库不包含它。
入口禁止模型隐式联网下载。训练每次更新原子保存参数、优化器、RNG 和 epoch/案例游标；
resume 校验 index/bank、生成器身份和步数，不允许静默改变来源协议。
数据按需载入，最多保留两个案例，不把整个训练集堆到 GPU。
运动与评分参数分组优化、独立裁剪；日志保存两组原始梯度范数、各loss分量与目标投影残差。
FiLM读取共享时间／步长／来源和两侧全原子接触；评分监督不更新运动主干。
候选修正器从严格零运动初始化，并在训练与推理中保留该轨迹自身的原始候选参考。
终点几何按项log平衡，逐项退化采用2倍软惩罚；新目标不能与旧原始MSE总loss直接比较。
旧自由运动模型的checkpoint不得作为这套修正器的恢复快照。
已有SDPI模型经`assembly_pipeline.py migrate --baseline ... --steps 20`做明确EMA转换，
新通道零贡献，且同起点20步行为保持验收通过后才发布新格式权重；该操作不是resume。
精调以`--weights-from`继承转换权重，保留专家参考系和已训练运动头，
`--sdpi-replay-weight .2`默认分配20%目标权重给每次新SDPI起点。migrate目前保留
旧固定锚点有效域，不声称兼容原专家无法处理的短链；不能把这点当作MGD94已全部打通。
推理必须使用已训练的新架构checkpoint，不提供未训练模型的smoke旁路。

新框架不是旧 checkpoint 的续训模式；先训练新主干与评分头。`predict` 导出每个候选的
完整 PDB、配体 SDF 与选择记录。可用 `--native-root` 添加**选择后的** DockQ 和原子顺序
配体 RMSD；它不改变候选或评分。该 RMSD 尚不是对称性修正值。
候选主干额外读取全原子跨组分距离、密度、重叠和等变方向，让配体节点看到侧链障碍。
全原子元素半径排斥是物理代理，不是完整力场；评分头未经独立校准，不能把评分当成功概率。
一次更新汇总完整案例的全部候选，依次反传，不同时保留多个rollout图；
日志逐候选记录，明确区分update数与rollout数。每四次更新添加一项目标保持/近目标
修正任务；比例不是已验证最优。评分有同池排序监督，独立学习率用`--quality-lr`设置。
默认输出保留 HDOCK 排名，不让少量更新后的未校准评分覆盖这个先验；
`--selection learned` 是显式实验开关，选择记录同时保存两种选择的索引。
候选专家采用输入 N-CA-C 骨架参考系及刚体运输，POI 全原子位姿和监督保留绕短链轴的旋转；
不能用只有两个 C-alpha 的拟合替代完整旋转。旧 checkpoint 不迁移到这套候选协议。

```bash
python scripts/run_cpu_tests.py
python scripts/make_deepternary_index.py --help
python scripts/train_entry.py --help
python scripts/evaluate_entry.py --help
python scripts/evaluate_index.py --help
```

`train_entry.py` 是标准训练入口；`train_matcha_single.py` 是 Matcha 专用训练；`train_distributed.py` 的多卡实现与验收边界见[吞吐说明](../doc/TRAINING_THROUGHPUT.md)。评估入口的 backend、权重和初始化支持范围不同，使用前核对参数及[项目交接](../doc/PROJECT_HANDOFF.md)。

## 顶层工具

| 用途 | 文件 |
| --- | --- |
| 索引与面板 | `make_deepternary_index.py`、`make_index_subset.py`、`prepare_validation_index.py`、`prepare_local_mgd_panel.py` |
| 化学与条件准备 | `restore_ligand_chemistry.py`、`prepare_protac_binary_chemistry.py`、`make_protac_pocket_manifests.py`、`protac_dual_anchor_init.py` |
| 特征与输入检查 | `generate_esm_embeddings.py`、`audit_ternary_inputs.py`、`audit_ternary_featurization.py`、`audit_protac_anchor_chemistry.py`、`audit_dual_anchor_reachability.py`、`audit_matcha_pretraining_overlap.py` |
| 评估准备与权重选择 | `prepare_checkpoint_evaluation.py`、`select_best_checkpoint.py`、`migrate_candidate_task_embedding.py` |
| 结构评估 | `evaluate_entry.py`、`evaluate_index.py`、`evaluate_selected_refinement.py`、`evaluate_binary_interfaces.py`、`score_binary_dockq.py` |
| 报告汇总 | `summarize_evaluation_reports.py`、`summarize_official_benchmark.py`、`summarize_candidate_refinement_pool.py` |
| 运行辅助 | `gpu_status.sh`、`launch_dual_init_factorial_local.sh`、`run_cpu_tests.py`、`benchmark_training_throughput.py` |
| 小样本实验 | `run_small_sample.py`、`run_small_sample_fixed.py`、`run_coupled_small_sample.py` |

三个小样本脚本是不同实验协议，暂保留独立入口，不视为重复实现。`featurize_common.py`、`evaluation_common.py` 和 `evaluation_runner.py` 是共享模块，保留在顶层供现有入口导入。

## 诊断工具：`diagnostics/`

以下文件从顶层迁入，文件名与参数保持不变；旧路径不再保留包装脚本。

| 文件 | 用途 |
| --- | --- |
| `validate_initialization.py` | 真实案例初始化检查 |
| `validate_sdpi_flow.py` | SDPI 几何检查及可选 GPU smoke |
| `validate_assembly_model.py` | 装配模型与采样检查 |
| `audit_initialization_consistency.py` | 初始化一致性审计 |
| `audit_endpoint_support.py` | 推理与训练配体终点差异 |
| `audit_protac_conditions.py` | PROTAC 条件输入审计 |
| `train_loss_diagnostic.py` | 固定子集 local/trajectory loss 实验 |

例如：`python scripts/diagnostics/validate_sdpi_flow.py --help`。诊断通过不能替代正式数据集评估。

## 结果分析：`analysis/`

以下文件同样从顶层迁入：

- `analyze_ensemble.py`：分析已保存的构象集合。
- `analyze_cooperativity.py`：协同性分析。
- `compare_exact_runs.py`：按相同案例和初态配对比较完整 DockQ 报告。

例如：`python scripts/analysis/compare_exact_runs.py --help`。

## 无训练双锚点 PROTAC 搜索（研究原型）

`protac_dual_anchor_init.py` 从两个实测二元配体位姿、对应蛋白和完整 PROTAC
化学图出发，生成 linker 构象并联合调整 linker torsion 与第二个蛋白的位姿；
不使用 HDOCK 候选作事后筛选。完整 PROTAC SDF 中的沉积坐标会被丢弃，
native 复合物仅在候选排序结束后用于 DockQ 评估。运行需已有 CCD CIF、
二元配体 PDB 和二元蛋白 PDB；此入口没有训练权重，也不会下载模型。

```bash
/home/wl/micromamba/envs/tricoopdiff/bin/python -m scripts.protac_dual_anchor_init \
  --index results/proposal_oracle_protac_20260913/protac22_unbound_v1.json \
  --ccd-dir results/protac_anchor_chemistry/ccd \
  --out results/protac_joint_search \
  --candidates 5 --refine-steps 300 --seed 73
```

`--search-conformers` 可使内部搜索起点多于最终输出候选数，但不能据此称为
“五次搜索”。当前物理有效性要求跨组分重原子距离不低于元素范德华
半径和的 85%（不重判已观测二元锚点与本侧口袋的接触），仍不是完整力场或实验可行性证明。二元参考化学与完整 PROTAC 不匹配时会
明确失败，不应靠降低共同核心阈值强行映射。现阶段属于原型，不能替代
正式全数据验证或声称达到 DockQ 0.4 门槛。

**输入坐标系警告：** 发布的 `protac22/*/unbound_protein1.pdb` 与
`unbound_protein2.pdb` 保留近似真实三元相对位姿；直接合并 22 例的事后
DockQ 均值为 0.8425（最低 0.7099）。任何以二者原始相对坐标作为初态或
先验的结果都不属于独立二元结构盲预测。双锚点生成器必须只使用每一侧的
蛋白–warhead **内部**几何，并从完整 PROTAC 化学图重新构造两侧相对位姿；
`audit_dual_anchor_reachability.py --posthoc-native` 会单独报告这一泄漏诊断。

## 维护约定

新增脚本先确认是否能扩展现有入口；专用诊断与离线分析分别放入对应子目录，并更新此索引。不要仅凭名称或缺少文档引用删除实验协议；删除前检查导入、调用方与复现需求。数据、结果及 checkpoint 不属于脚本清理范围。
