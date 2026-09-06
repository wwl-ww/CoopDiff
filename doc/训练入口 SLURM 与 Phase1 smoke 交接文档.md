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

## 产物管理

`logs/` 和 `results/` 是可再生产物，可按保留策略清理；`checkpoints/` 不随日志清理。实验报告记录 commit、job ID、命令、seed、初始化模式、结果路径和关键指标。不要将 checkpoint 或大结果文件提交 Git。
