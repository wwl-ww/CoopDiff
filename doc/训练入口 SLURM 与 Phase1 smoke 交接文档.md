# 训练与 SLURM 运行指南

## 调度原则

小型诊断应写成可复现 Python 脚本，提交 Git 后交由同学运行。训练、完整 GPU parity 和大规模评估才使用 SLURM。有效 GPU 分区包括 `gpu_a800`、`gpu_h100`、`gpu_h200`；提交前用真实调度器检查，不依赖静态 `sinfo` 包装输出。

```bash
sbatch scripts/train_deepternary_smoke.slurm
squeue -j <job_id>
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,NodeList
```

## 环境与缓存

统一激活 `/data/home/scwb792/run/miniconda3` 下的 `tricoopdiff` 环境。旧 smoke 曾同时暴露两个环境问题：缺少 Pillow 导致 ESM case 被跳过；Triton 编译缓存写入 home 时触发 `Errno 122: Disk quota exceeded`。当前 smoke 脚本应保持以下约束：

- 导入 `PIL`、PyTorch 和 `transformers.models.esm.modeling_esm` 做预检；
- 从 conda site-packages 收集 NVIDIA library 目录加入 `LD_LIBRARY_PATH`；
- 将 `TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR` 和 `TMPDIR` 指向项目盘；
- 检查 CUDA、DeepTernary 数据索引、expert 参数/权重及缓存可写性；
- Hugging Face 使用本地缓存并设置 offline 模式，作业不得静默下载模型。

## Phase 1 门禁

训练前必须具备新鲜的：

1. `results/sampling_parity_regression.json`：sub1/sub2 全部 finite/pass；
2. `results/initialization_frame_check.json`：全部 frame checks 通过；
3. 有效且互不重叠的 `data/ternary_train.json`、`data/ternary_val.json` 和
   `data/ternary_test.json`，所有路径在当前 clone 可解析；
4. `checkpoints/score_model/model_parameters.yml` 与 expert 权重。

使用 `scripts/check_phase1_readiness.py` 汇总，只有 `ready=true` 才能进入正式训练。清理 `results/` 后必须重新生成门禁报告。

## Smoke 与正式训练

```bash
# DeepTernary 独立 train/val 小样本 smoke
sbatch scripts/train_deepternary_smoke.slurm

# smoke 验收后才运行正式 Phase 1
sbatch scripts/train_deepternary_phase1.slurm
```

Smoke 验收至少检查：退出码 0；两个界面的 loss、总 loss 和梯度有限；训练和验证均有有效 batch；FiLM/LoRA 参数发生更新；每 epoch checkpoint 可读；验证阶段保持 eval mode。日志应分别打印 E3–Ligand 与 POI–Ligand 的 translation、rotation、torsion loss，避免某个 expert 长期无梯度而不自知。产出的 checkpoint 必须是 version 4 且标记 `architecture=interface_coordinate_v1`；加载 version 3 应明确失败。

## 产物管理

`logs/` 和 `results/` 是可再生产物，可按保留策略清理；`checkpoints/` 不随日志清理。实验报告记录 commit、job ID、命令、seed、初始化模式、结果路径和关键指标。不要将 checkpoint 或大结果文件提交 Git。
