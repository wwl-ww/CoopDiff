# BiTSeed Refinement 退化最小复现交接

## 任务边界

本分支只回答一个问题：**当前按 SDPI 路径训练的 TriCoopDiff，是否会稳定破坏一部分高质量 BiTSeed 初始化；如果会，现有训练路径与时间语义中有哪些可直接证明的不匹配？**

这不是训练任务，也不是新架构开发任务。不得实现 source embedding、双模型、LoRA 微调、selector 训练或大规模时间步消融。不得根据 MGD94 的 native DockQ 调参。`hh.md` 仅作为待审计提议，位于主工作区 `/mnt/disk/wl/MoRE-tri/CoopDiff/hh.md`，不得直接复制其结论。

工作位置与基线：

- worktree：`/mnt/disk/wl/MoRE-tri/CoopDiff-bitseed-mixed-path-repro`
- branch：`exp/bitseed-mixed-path-repro`
- base commit：`30af27a`
- Python：`/home/wl/micromamba/envs/tricoopdiff/bin/python`
- 所有命令从本 worktree 根目录运行，并设置 `PYTHONPATH=.`

## 已知事实（复现目标，不得重新定义）

昨晚 MGD94 full-pool 实验共有 94 个唯一 case，每例 256 个候选（192 BiTSeed + 64 SDPI），总计 24,064 条有效 refinement，失败为 0。聚合结果为：

- SDPI 每例最佳 DockQ：`0.2077376 -> 0.2515289`，70/94 case 提高；
- BiTSeed 全池每例最佳：`0.2725974 -> 0.2630622`，48/94 提高、46/94 降低；
- 候选级 BiTSeed：平均 `0.0397926 -> 0.0447878`，13,578/18,048 提高；
- 初始化 oracle 与最终 oracle 只有 25/94 是同一 candidate。

这组现象的准确表述是：**TriCoopDiff 平均改善多数 BiTSeed 候选，但不能稳定保护每例最好的 BiTSeed 候选。** 不得简化为“TriCoopDiff 对所有 BiTSeed 都有害”。

优先使用以下两个通过初始 `acceptable=True` 检查的确定性退化候选：

| Case | Candidate ID | diffusion seed | Initial | Refined | Delta |
|---|---|---:|---:|---:|---:|
| `5ZCG_C_A_A8S` | `['bitseed', 2, 3, 2]` | 2031115 | 0.4449419 | 0.2338222 | -0.2111197 |
| `2FAP_A_B_RAD` | `['bitseed', 3, 2, 1]` | 2031157 | 0.4597531 | 0.3532094 | -0.1065437 |

候选 `3UJL_A_B_A8S / ['bitseed',0,4,2]` 虽下降 `-0.26034`，但 `acceptable=False`，只能作为补充反例，不能作为核心证据。

## 唯一数据源

只读读取主工作区下列目录，不复制或修改结果 JSON：

```text
/mnt/disk/wl/MoRE-tri/CoopDiff/results/mgd_full_pool_refinement_20260913/full94_v1_gpu5
/mnt/disk/wl/MoRE-tri/CoopDiff/results/mgd_full_pool_refinement_20260913/full94_v1_gpu7
/mnt/disk/wl/MoRE-tri/CoopDiff/results/mgd_full_pool_refinement_20260913/full94_v2_gpu5a
/mnt/disk/wl/MoRE-tri/CoopDiff/results/mgd_full_pool_refinement_20260913/full94_v2_gpu5b
/mnt/disk/wl/MoRE-tri/CoopDiff/results/mgd_full_pool_refinement_20260913/full94_v2_gpu7a
/mnt/disk/wl/MoRE-tri/CoopDiff/results/mgd_full_pool_refinement_20260913/full94_v2_gpu7b
```

排除 `manifest.json`、`failures.json` 和 `smoke_3jrq_v1`，并在统计前断言：94 个唯一 case、每例 `union_evaluated_count=256`、每例 192/64 来源构成、所有 refinement `valid=True`。

实验 checkpoint 仅用于必要的小规模重放：

```text
/mnt/disk/wl/MoRE-tri/CoopDiff/results/paired_pilots/followup_seed28102_teacher_acc8_64cases_e16/epoch_002.pt
```

## 必须按顺序完成的工作

### P0：离线统计复现

新增一个轻量脚本，接受显式 `--results-root` 或多个 `--input-dir`，复算上述总体事实，并输出：

1. 完整性检查；
2. SDPI/BiTSeed 的 candidate-level 初始、最终、delta 和增减数量；
3. source-level 每例最佳 Initial/Final/Safe；
4. 每例最佳 BiTSeed 被改善/破坏的数量；
5. 高质量且 `acceptable=True` 的最大退化候选表。

统计逻辑必须有小型合成 JSON 单元测试。不得只把预期数字硬编码进脚本。

### P1：路径可表达性审计

检查 `SDPIFlow.path()` 的 endpoint 参数化是否能对任意外部 BiTSeed 状态进行精确逆映射。审计必须分别讨论：

- ligand 刚体平移和旋转；
- POI 相对刚体；
- ligand torsion；
- 外部 conformer 与 native conformer 存在非扭转内部形变时的残差；
- body-frame/gauge 是否唯一。

优先写纯 CPU 的 round-trip/残差测试。此阶段只能新增独立诊断 helper；不得改变生产 `SDPIFlow.path()`、sampler 或训练行为。结论必须区分“数学上不可表达”“当前代码没有 inverse”“数值/规范选择不唯一”。

### P2：双时间语义审计

沿代码调用链记录同一个 `t` 当前控制的内容：flow 插值进度、DiffDock `t_to_sigma`/time embedding、TriFuse Fourier time、`diffusion_state` 和 dynamic crop。回答：把结构化 BiTSeed endpoint 直接标为 `t=1` 时，哪些模块仍把它解释为 SDPI 最大噪声。

这里只写证据和最小接口草案，**不得实现时间解耦**。草案应使用不同符号，例如 `tau_flow` 与 `t_expert_e3l/t_expert_pol`，并说明 checkpoint 兼容边界。

### P3：可选的单候选 GPU 重放

只有 P0-P2 完成且确实需要验证存档结果可重放时才允许。先运行 `nvidia-smi`，禁止使用 GPU 6；最多重放上表两个候选、每个仅基准 `t_start=1, steps=20`，不得做 grid sweep。必须使用指定 Python，若依赖缺失或无法从 JSON 恢复精确坐标，记录 blocker 后停止，不得另造近似候选。

## 禁止事项

- 不启动任何训练、SLURM 作业或 94-case 重跑；
- 不修改主 worktree；
- 不修改 frozen vendored DiffDock；
- 不用 MGD94 native 指标选择超参数；
- 不把相关性写成已证明根因；
- 不提交 checkpoint、results、logs、cache 或大文件；
- 不顺手修复无关问题；
- 不修改已有 `scripts/evaluate_refinement_time_sweep.py`，除非是完成单候选精确重放所必需；若无必要，撤销本 agent 自己对此文件的改动。

## 验收与提交

最低交付物：

- 一个可重复的离线审计脚本及测试；
- endpoint 可表达性 CPU 测试或明确的不可实现证据；
- `doc/` 下的简短结果报告，逐项回答 P0-P2；
- `git diff --check` 通过；相关测试使用指定 Python 通过；
- 一次范围清晰的 Conventional Commit。

最终汇报只包含：commit hash、运行命令、复现数字、两个核心案例、已证明事实、仍属假设的根因、下一步最小实验。若 P0 的完整性断言不满足，立即停止并报告，不继续 P1-P3。
