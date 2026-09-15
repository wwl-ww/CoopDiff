# 运行、复现与后台作业

更新：2026-09-15。以下是操作参考，不自动授权启动训练或占用 GPU。文件名中的 SLURM 是历史命名，本机不使用 SLURM。

## 环境与资产预检

从仓库根目录运行。当前本机常用解释器为 `/home/wl/micromamba/envs/tricoopdiff/bin/python`；其他机器先核实环境，不照抄旧 `/tmp` 路径。仓库 [environment.yml](../environment.yml) 是导出快照，CUDA、PyTorch 和扩展 ABI 应以实际导入检查为准，不原地升级正在使用的环境。

预检包含 PyTorch/CUDA、RDKit、PIL、ESM 导入；确认专家 `model_parameters.yml` 与权重、Hugging Face 离线缓存、数据索引/模板、可写特征和数值缓存。设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，不让作业静默下载模型。新入口先导入 `tricoopdiff.diffdock_compat`。

特征缓存当前为 v7（见 `scripts/featurize_common.py`），不是旧文档的 v5；改变输入化学、torsion 或映射必须匹配缓存签名。数值协议采用 `wrapped_normal_quadrature_v1`；它不能重建历史 checkpoint 未记录的随机归一化表。

## 轻量验证

```bash
python tests/test_model_build.py
python tests/test_initialization.py
python tests/test_sdpi_flow.py
python tests/test_consensus.py
python tests/test_prepare_data.py
bash scripts/gpu_status.sh
```

按修改范围选择检查；以上不是本次文档整理重新跑过的测试声明。某些 `test_*.py` 仅定义函数，直接执行不一定运行全部断言，使用其实际 runner 或显式测试调用。GPU 模型检查先单例、单 walker，再扩规模；零初始化 smoke 不代表训练后准确率。

## MGD94 当前精修实验的最小复跑

原 checkpoint：

`results/paired_pilots/followup_seed28102_teacher_acc8_64cases_e16/epoch_002.pt`

身份为 `residual_consensus_factorial_v1` / version 1，frozen adaptation、CCD 化学、single_bond torsion，评估使用 EMA。精确哈希和其他参数查原始 `manifest.json`，不要替换成标准 version 18 或旧 v13。

以下命令需要已获授权的 GPU 和新输出目录；`--max-refinements 1` 仍会生成候选池，只限制精修条数：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/evaluate_selected_refinement.py \
  --index results/reproduction_20260911/index_current/mgd94_full_v1.json \
  --checkpoint results/paired_pilots/followup_seed28102_teacher_acc8_64cases_e16/epoch_002.pt \
  --allow-experimental-checkpoint --cases 3JRQ_B_A_A8S \
  --seed 31000 --conformers 4 --poses-per-side 8 \
  --bitseed-candidates 192 --sdpi-candidates 64 \
  --binary-steps 20 --steps 20 \
  --binary-retry-limit 2 --binary-consecutive-failure-limit 3 \
  --refine-full-pool --max-refinements 1 \
  --out-dir results/selected_refinement_smoke_new
```

完整审计删除 `--cases` 和 `--max-refinements`，换新输出目录；这会跑 94×256 条，不是普通 x64 成本。若不传 `--refine-full-pool`，入口精修各 selector 选中集合的并集，也不一定恰为 64 条。入口见 [evaluate_selected_refinement.py](../scripts/evaluate_selected_refinement.py)。

普通单例标准模型用 `scripts/evaluate_entry.py --trifuse-ckpt <version18.pt>`；实验权重使用支持实验门禁的入口，不通过重命名绕过加载契约。`--allow-zero-init` 是未训练 Consensus，不是官方独立 DiffDock baseline。

## 全池结果与历史脚本

最新审计根目录为 `results/mgd_full_pool_refinement_20260913/`，包含两条旧 worker 各完成的 1 例和四条新 worker 各 23 例。以 `summary_v1.json → integrity.case_files` 的 94 个唯一 case 为准，不将 smoke 或旧 manifest 中未完成的 case 重复计入。

已删除的一次性 `summarize_full_pool_refinement.py` 及其依赖 `tricoopdiff/refinement_audit.py` 可从 `a097801` 恢复到独立 Git worktree 后使用；主分支不保留重复汇总入口。原始逐例 JSON 与 manifest 才是复核依据，汇总可重建。旧 proposal oracle 脚本也在同一历史提交中，不再列为当前可执行命令。

## 后台、并行与资源

`bash scripts/gpu_status.sh` 一次显示所有卡；结合用户、完整命令、PPID、会话和日志确定所属任务。PPID=1 仅表示重新托管，不能单独证明断线安全。

本机直接使用用户授权的物理 GPU，持久作业使用已配置的 tmux/独立会话，日志重定向到运行目录；断开终端后复查任务和输出仍更新。旧 SLURM wrapper 仅为历史参考，不作为当前启动入口。tmux 若遇到环境中的 libtinfo 冲突，可只对 tmux 命令使用 `env -u LD_LIBRARY_PATH tmux ...`，不修改全局运行环境。

多 worker 使用互斥 case 划分或真正带原子认领的队列，单独日志；不要让两个进程同时写同一 case/manifest。显存够不保证 1.5–2 倍加速，记录吞吐、峰值显存和失败率。更改并发不改变 candidate ID、seed 或科学配置。

## 训练与交付门禁

联合训练实现已从 `exp/dual-init` 合入当前分支，原工作树及训练产物仍独立保留；合并代码不迁移checkpoint/cache。历史 v13/LoRA smoke 不能替代当前 Consensus/factorial 或双来源训练验收。旧夜间 `overnight_dual.json` 的双状态设想也不等于本次单模型双初始化目标。

训练前审计实际 PDB-disjoint split；随后检查 finite forward/backward、预期可训练参数更新、专家冻结、保存恢复、独立验证和双来源能力。固定一例降 loss 不是泛化证据。checkpoint 记录架构/配置/数据/数值身份，结果记录命令、commit、seed、进程或 job ID。不要覆盖旧权重，也不随文档清理删除数据或日志。
