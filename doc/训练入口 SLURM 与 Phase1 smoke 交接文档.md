# 训练入口 SLURM 与 Phase1 smoke 交接文档

> 来源：并行实例（另一个 Claude）在打通训练入口时踩到的坑 + 对 Phase1 smoke 跑批的观察。
> 补充 `def64a4`（训练入口接线）已提交但未覆盖的内容。本文只列非显然、可操作的点。

## 1. SLURM 提交要点（重要，容易被 `sinfo` 骗到）

- **真实分区只有 3 个**：`gpu_a800`、`gpu_h100`、`gpu_h200`。
  用 `/usr/local/bin/slurm/sinfo -h -O Partition` 或 `scontrol show partition` 查。
  **`hp_a800` 不存在**——`sinfo` 这个命令是 `/usr/bin/sinfo`（一个 bash 包装脚本），
  它的"公共partition"一栏读的是静态文件 `/data/apps/sinfo/sinfo.log`，里面还列着
  `hp_a800`，但实际调度器上没有。别拿它当提交依据。

- **CPU 配额**：`/etc/slurm/job_submit.lua` 的 `check_cpus()` 对非 h100/h200 分区
  强制 **每 GPU 恰好 6 核**。`--cpus-per-task` 必须 ≤6（建议用 4，会被默认 6 核/GPU
  的兜底抬到 6）。`--cpus-per-task=8` 会直接报
  `sbatch: error: slurm_job_submit_lua, Batch job submission failed: Unspecified error`。

- **gpu_a800 很挤**：30 节点里 12 个 drain、11 个全分配，1 卡作业通常被 backfill 排到
  ~30 分钟后才启动（`scontrol show job <id>` 看 `StartTime`）。这是正常排队，不是报错。

## 2. Phase1 smoke 关键观察：phys 和 se3 两项损失基本是死的

3-epoch B=1 smoke 跑下来，`loss_total ≈ loss_score`（典型 `loss=4.19 score=4.17`）。
而 `loss_total = 1.0·score + 0.5·phys + 0.3·se3`，所以：

**`loss_phys ≈ 0` 且 `loss_se3 ≈ 0`（量级 0.01，占总损失 <1%）**。

- `loss_phys ≈ 0`：`_physical_loss` 用的是固定 3.4 Å 硬截断（CLAUDE.md 已标记为占位），
  良分离复合物在噪声态几乎无 clash，relu(3.4 − dist) 全为 0。**预期内，但意味着
  Phase1 期间物理约束项根本不产生梯度**，只有等换成真实 VDW 半径才有意义。

- `loss_se3 ≈ 0`：**需要查**。SE3 head 是零初始化（输出≈0），如果这时 loss 仍≈0，
  说明要么 `se3_t_to_sigma` 的 σ 加权把它压没了，要么 target 量级不对。目前 Phase1
  唯一有效的梯度来自 score-matching 项（经 FiLM 路径），SE3 head 和 SE3 loss 等于没在训。

- **建议**：`train_entry.py` 的 per-batch 日志目前只打印 `loss_total` / `loss_score`，
  请把 `loss_phys`、`loss_se3`、`loss_se3_tr`、`loss_se3_rot` 一起打出来（
  `compute_loss` 的返回 dict 里都有），否则上面两个死项在训练时完全不可见。

## 3. checkpoint 体积与 resume 问题

- `TriCoopDiffTrainer.save_checkpoint` 同时存 `model_state_dict`（全模型，含冻结 expert，
  **~131MB**）和 `tribrain_state_dict`（~9MB）。但 `load_checkpoint` **只读
  `tribrain_state_dict`**。Phase1 冻结 expert 不变，`model_state_dict` 是纯冗余。
  建议去掉它，checkpoint 缩小 ~15×。否则 100 epoch @ save_interval=10 ≈ 1.3GB。

- `load_checkpoint` 目前只恢复 `tribrain_state_dict` + `global_step`，**不恢复
  optimizer / scheduler**。Phase1→Phase2 换新 optimizer 没问题，但若要在 Phase1 中途
  resume 续训，LR 调度会从 0 重来——需要的话补上。

## 4. （可选）已删除脚本里的三个增强，供合入正式脚本参考

并行实例那份 `train_phase1.slurm` 已删（与 `train_entry_smoke.slurm` 重复）。其中三个
对正式 Phase1 训练有用的点，需要的话可以合进 `train_entry_smoke.slurm`：

1. 训练前内联跑 `make_protac_index.py`，脚本自包含（不依赖外部先建好索引）。
2. `EPOCHS` / `LR` 用 `${EPOCHS:-3}` 暴露，`sbatch --export=EPOCHS=100,LR=1e-4` 可覆盖。
3. 输出目录用 `checkpoints/phase1/`（生产位置），而非 `results/train_smoke/`（冒烟位置）。

## 结论

接线本身已通（`def64a4` + 3-epoch smoke 无崩溃、checkpoint 可写）。**在扩到真实
50–100 epoch 之前，先解决第 2 节的 `loss_se3 ≈ 0`**——否则 Phase1 只是 FiLM 在 score
matching 上微调，SE3/协同目标完全没有被训练。
