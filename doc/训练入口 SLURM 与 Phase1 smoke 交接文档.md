# 训练与 SLURM 运行指南

## 调度原则

诊断写成可复现 Python 脚本并提交 Git。集群 GPU 任务通过 SLURM 申请资源，不在登录节点运行；本地有独立 GPU 时可运行小型 smoke。仓库 wrapper 使用 `gpu_a800` 等站点配置，分区和资源是否可用须在提交时向实际调度器查询，不能由本文保证。

```bash
sbatch scripts/train_deepternary_phase1.slurm
squeue -j <job_id>
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,NodeList
```

## 环境与缓存

集群 wrapper 当前示例使用 `/data/home/scwb792/run/miniconda3` 下的 `tricoopdiff` 环境；迁移站点时同时修改项目与环境路径。环境预检应满足：

- 导入 `PIL`、PyTorch 和 `transformers.models.esm.modeling_esm` 做预检；
- 从 conda site-packages 收集 NVIDIA library 目录加入 `LD_LIBRARY_PATH`；
- 将 `TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR` 和 `TMPDIR` 指向项目盘；
- 检查 CUDA、DeepTernary 数据索引、expert 参数/权重及缓存可写性；
- Hugging Face 使用本地缓存并设置 offline 模式，作业不得静默下载模型。

特征缓存当前为 version 5，包含初始化所需的索引化 RDKit 化学图。旧缓存不能直接用于 `sdpi` 推理，需要通过正式特征化入口重建；模型权重格式仍为 version 17。

## Phase 1 门禁

训练前必须具备：

1. PROTAC/Molecular Glue 联合训练池固定为 7,198 cases、653 merged clusters；
2. 按 cluster 划分且互不重叠的 `data/ternary_train.json` 和
   `data/ternary_val.json`；
3. 两个正式 benchmark 的同 PDB case 已从训练池排除；
4. `checkpoints/score_model/model_parameters.yml` 与 expert 权重。

使用 `scripts/check_phase1_readiness.py` 汇总，只有 `ready=true` 才能进入正式训练。清理 `results/` 后必须重新生成门禁报告。

## Smoke 与正式训练

为节省调度次数，正式 wrapper 在同一个 allocation 中依次生成/验证索引、运行
104-train/8-validation cluster smoke，然后从头运行 100-epoch Phase 1。任一门禁
失败都会阻止正式训练启动：

```bash
sbatch scripts/train_deepternary_phase1.slurm
```

Smoke 验收至少检查：退出码 0；两个界面的 loss、总 loss和全部 TriFuse/LoRA/bridge 梯度有限；训练和验证均有有效 batch；FiLM/LoRA/torsion coordinator 参数发生更新；每 epoch checkpoint 可读；验证阶段保持 eval mode。日志应分别打印 E3–Ligand 与 POI–Ligand 的 translation、rotation、branch torsion 和 joint torsion loss。产出的 checkpoint 必须是 version 17 且标记 `architecture=trifuse_v13`、`generative_process=sdpi_flow_v1`；旧版本应明确失败。

## 固定路径优化诊断

训练 loss 异常时先运行 `scripts/diagnose_flow_training.py`。它固定训练结构、
时间和一次 SDPI coupling，关闭 dropout，保留生产损失、AdamW 参数组、
梯度裁剪和 LoRA warmup。默认 200 步覆盖前 100 步冻结和后 100 步适配。

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/diagnose_flow_training.py \
  --train-index data/ternary_train.json --case <training_case> \
  --out results/flow_fixed_pair --steps 200 --time 0.5
```

只有本地压缩包时，可用 `scripts/prepare_flow_diagnostic.py --out <new_directory>`
提取一个官方训练划分成员并生成小索引；该索引不能替代正式全量门禁。
禁止将 PROTAC22/MGD94 benchmark 用作拟合样本。

输出包含 `fixed_pair.pt`、`metadata.json`、`steps.jsonl` 和 `summary.json`。
逐步检查分项 loss、预测/目标 RMS、FiLM/bridge/coordinator/两侧 LoRA 的
梯度及参数增量、裁剪前范数和逐层 activation clamp 比例。分项梯度按原始
未加权 loss 计算，auxiliary torsion 在总目标中的权重仍为 0.125/侧。
初始为零的参数使相对更新比例不稳定，须同时查看绝对增量。

单样本降损失不代表泛化；通过后再做少量训练结构随机路径拟合和独立验证条件消融，
最后才扩全数据。`time=0.1/0.5/0.9` 分开运行，避免用单个时间点排除尺度问题。
`--steps 0 --lora-warmup-steps 0` 只记录初始尺度与梯度，不做优化。
`scripts/audit_flow_pair.py --pair <fixed_pair.pt> --out <audit.json>` 在实际
样本上检查三个时间点的联合 Cartesian tangent 与反向积分步长收敛。

第二阶段用 `prepare_flow_diagnostic.py --count 3` 构建小训练子集，再给
诊断入口加 `--random-paths`。此模式按 case 轮换、随机时间和路径、启用训练
dropout，并对每个结构保留一条独立路径，在三个时间点做训练前后对照。
对照属于未见路径，不是未见结构，不能称为全数据验证或条件有效性证明。
加 `--condition-ablation` 可在训练后保持 expert 输入和原目标不变，只将
TriFuse 看到的 POI 状态换成独立采样的状态。记录 E3-L 的 FiLM、速度变化，
以及正确/打乱条件的 E3-L 损失；该诊断只覆盖 TriFuse 条件通路，不覆盖
bridge 的全部条件通路，也不构成另一方向或全数据条件收益的证据。
随机路径模式保存 `diagnostic_model.pt` 供后续诊断；它不是正式评估权重。
`python scripts/summarize_flow_diagnostic.py <run_directory>` 汇总 case、
0.2 宽度时间段、torsion 数量和 warmup 前后的损失及裁剪统计。
环境若设置了错误的 `GCC_ROOT=/usr/bin/gcc-13`，可在运行命令前使用
`env -u GCC_ROOT`；不要为绕过本地编译问题升级训练依赖。

## 产物管理

`logs/` 和 `results/` 是可再生产物，可按保留策略清理；`checkpoints/` 不随日志清理。实验报告记录 commit、job ID、命令、seed、初始化模式、结果路径和关键指标。不要将 checkpoint 或大结果文件提交 Git。
