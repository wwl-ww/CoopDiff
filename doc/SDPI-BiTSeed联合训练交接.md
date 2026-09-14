# SDPI–BiTSeed 联合训练交接

## 最终目标

本分支的唯一目标是：**训练一个共享的 TriCoopDiff checkpoint，使同一个模型同时兼容 SDPI 与 BiTSeed 两种初始化。**

目标模型在推理时接收候选来源条件：

```text
source=0: SDPI 初始化
source=1: BiTSeed 初始化
```

模型必须真正见过两类初态及其各自到 native ternary target 的训练路径。不得拆成两个 CoopDiff 模型；不得只在推理时增加标志位；不得把 BiTSeed 直接当成未经训练的 SDPI `t=1` 状态。

工作位置：

- worktree：`/mnt/disk/wl/MoRE-tri/CoopDiff-bitseed-mixed-path-repro`
- branch：`exp/dual-init`
- base：`30af27a`
- Python：`/home/wl/micromamba/envs/tricoopdiff/bin/python`
- 提议原文：主工作区 `/mnt/disk/wl/MoRE-tri/CoopDiff/hh.md`

## 实验依据

MGD94 full-pool 结果已经说明训练新模型的必要性：

- SDPI 每例最佳：`0.2077376 -> 0.2515289`，说明当前模型能 refinement SDPI；
- BiTSeed 每例最佳：`0.2725974 -> 0.2630622`，说明当前模型不能稳定保护高质量 BiTSeed；
- BiTSeed 候选平均仍有改善，说明问题不是“BiTSeed 完全不可用”，而是训练路径和条件不匹配；
- 初始化 oracle 与最终 oracle 只有 25/94 为同一候选，训练和验证必须分别评价 initial、final 与 delta。

这些数字是旧 checkpoint 的基线，不是本分支的终点。

## 目标方案

共享一个 `TriCoopDiff` 主模型和运动输出，不增加第二套模型。第一版增加一个两类可学习 source embedding，并将它接入 **TriFuse 的条件/FiLM 路径**。冻结 DiffDock base，不直接改 vendored DiffDock 的原生 time embedding。

形式上：

\[
v_\theta(X_t,t,\mathcal C,s),\qquad s\in\{\mathrm{SDPI},\mathrm{BiTSeed}\}.
\]

source 条件必须沿整条 trajectory 保持不变。由 fallback 实际生成的 SDPI 候选标为 SDPI，不能按原计划配额伪装成 BiTSeed。

训练初态默认按 `SDPI:BiTSeed = 1:1` 采样。该比例只是第一轮受控配置，不等于推理阶段的 48/16 walker 配额。

## 必须实现的训练路径

### SDPI 路径

保留当前 `SDPIFlow` 的 native→SDPI coupling、速度目标、单位和归一化，作为旧能力控制组。

### BiTSeed 路径

训练时使用 frozen binary DiffDock experts 产生 native-free BiTSeed endpoint，但不复制
推理阶段的候选搜索预算。训练协议固定为 `C=1, M=1`：生成一个 ETKDGv3 conformer，
E3–ligand 与 POI–ligand expert 各运行一条 20-step trajectory，不重试、不排序。两侧都
有效时组合成唯一 endpoint；任一侧无效时整例回退 SDPI，实际来源记为 `source=0`。
不得使用 native DockQ 选择 endpoint。

endpoint 在训练时首次访问案例时生成，随后按 case 与固定 seed 原子缓存并跨 epoch
复用。训练缓存记录：

- case、conformer seed、两侧 binary pose id；
- ligand、POI、body-frame endpoint 坐标；
- 实际来源与 fallback 元数据；
- 生成协议版本及 frozen expert checkpoint hash。

必须为这个实际 endpoint 构造到 native target 的合法条件 flow/velocity target。先完成 endpoint round-trip：在刚体、POI body frame 和可旋转键自由度允许的误差内，路径 `t=1` 必须重建缓存的 BiTSeed endpoint。若外部 conformer 含当前 torsion+rigid 参数化无法表达的内部形变，应明确投影并记录残差，不能静默假定精确。

### 时间语义

必须明确区分：

- `tau_flow`：从 native 到相应来源 endpoint 的路径进度；
- `t_expert_e3l/t_expert_pol`：frozen DiffDock expert 的噪声/尺度条件。

当前一个 `t` 同时控制 flow、expert sigma、TriFuse time 和 dynamic crop。不得未经消融就把结构化 BiTSeed endpoint 等同于 SDPI 最大噪声。第一版可以先实现显式接口，再比较“共享时钟”和“校准 expert 时钟”；所有 checkpoint 必须记录时间契约版本。

## 公平训练消融

使用相同训练 split、batch/accumulation、优化步数、随机种子和验证候选：

| Arm | 训练初态 | source embedding | 用途 |
|---|---|---|---|
| A0 | 不继续训练 | 无 | 当前 checkpoint 基线 |
| A1 | SDPI-only | 无 | 控制额外训练预算 |
| B | SDPI + BiTSeed | 无 | 测混合路径本身 |
| C | SDPI + BiTSeed | 有 | 测 source 条件的额外价值 |

必须保留 A1；否则 B/C 相对 A0 的变化可能只是多训练了一段时间。B 与 C 必须使用完全相同的缓存 endpoint 和 batch 顺序。

## 执行顺序

1. **数据契约 smoke**：对少量 train/validation case 生成并缓存两类 endpoint，验证无 native 参与候选生成和筛选。
2. **路径 round-trip**：CPU 单元测试覆盖 SDPI 与 BiTSeed endpoint；检查坐标、torsion、body frame、速度 shape 和有限性。
3. **source conditioning**：将两类 embedding 接入 TriFuse 条件路径，新增 checkpoint 字段；旧 checkpoint 加载时 source 默认为 SDPI，保证回归兼容。
4. **单 batch 反向 smoke**：分别跑 SDPI、BiTSeed，确认 loss finite、source embedding 在 C 中有梯度、frozen expert base 无梯度。
5. **短训练 smoke**：A1/B/C 各用相同极小预算，先检查 loss 与 validation delta，不直接上长训练。
6. **正式训练**：smoke gate 全部通过后再启动同预算 A1/B/C；训练必须后台持久化并记录 PID、GPU、日志和配置 hash。
7. **评价**：只在内部 validation 选择 checkpoint/超参数；协议固定后，MGD94 仅做一次最终评价。

## 验收指标

验证时必须按来源分别报告：

- 初始最佳 DockQ、最终最佳 DockQ、`Final-Initial`；
- 改善/破坏 case 数量；
- 高质量初态（至少 `initial DockQ >= 0.23`）的保留率和平均下降；
- SDPI final 相对 A0/A1 的变化，检测 catastrophic forgetting；
- BiTSeed final 相对原始 BiTSeed 和旧模型的变化；
- Hybrid-48/16 的 Initial、Final、`max(Initial,Final)` 三张榜单；
- 物理 acceptability 与失败率。

成功条件不是“混合平均 loss 更低”，而是：

1. C 对 BiTSeed 的 final/delta 明显优于旧模型；
2. C 不显著损伤 SDPI final；
3. C 相对 B 的收益证明 source embedding 有价值；
4. 高质量 BiTSeed 不再被系统性破坏。

## 约束与交付

- 不训练两个独立模型；
- 不微调 frozen binary DiffDock base；
- 不用 MGD94 调参或选择 checkpoint；
- 不把 source bit 当作大步/小步硬开关；
- 不提交 checkpoint、cache、results 或 logs；
- 所有新增训练/推理字段进入 checkpoint/runtime contract；
- 代码、测试、训练配置和文档分批提交，使用 Conventional Commit；
- 每次交接必须给出 commit、命令、日志路径、GPU/PID、已通过 gate 和下一步。

本分支允许并期望进行训练，但必须在路径、条件、梯度和短训练 smoke 通过后再投入正式 GPU 预算。

## 当前实现状态

本分支已实现第一版共享时钟合同 `shared_flow_expert_clock_v1`：`tau_flow` 同时传给
frozen expert，后续校准 expert 时钟仍须单独消融。`ConsensusFuse` 可选两类 source
embedding；source 专属参数使用隔离 RNG 初始化，确保 B/C 的全部共享参数逐 tensor
同初值。采样器依据实际初始化结果固定 source，BiTSeed fallback 始终标为 SDPI。

`tricoopdiff/online_endpoints.py` 实现训练内 lazy endpoint：只生成唯一两专家组合，
将外部 endpoint 投影到现有 torsion+SE(3) chart，并保存原始 endpoint、训练 endpoint、
投影残差、seed、实际来源、fallback 和 frozen expert 身份。B/C 使用同一 case-seed 公式，
且各自 cache 可逐案例审计；A1 不构造该 store。推理仍可使用 `C=4, M=8` 提高覆盖率，
不得把推理预算重新带入训练。

GPU 7 的真实两步 C smoke 已覆盖 SDPI→BiTSeed 交替、lazy 首次生成/复用和 validation；
checkpoint v20 记录 source/time/mix 及 `online_bitseed_training_c1_m1_v1` 合同。已知 OOD
案例 `1ALX_A_B_MOH` 恰好产生两条 NaN 诊断（每侧一次），随后合法回退 SDPI。正式训练
和内部 validation DockQ 评价尚未运行。本机为直接 GPU 环境，不使用 SLURM；三臂命令为：

```bash
scripts/launch_dual_init_factorial_local.sh 5 6 7 /mnt/disk/wl/MoRE-tri/CoopDiff
```

## LoRA 接续阶段（已准备，尚未启动）

`scripts/run_dual_init_lora_arm_local.sh` 可在 frozen arm 完整结束后，从该 arm 的
EMA `best_model.pt` 初始化 TriFuse，并零初始化两套 branch-specific LoRA。DiffDock
base 权重始终冻结；训练前向分别启用 `sub1/sub2` adapter，而 lazy BiTSeed endpoint
生成会临时旁路 LoRA，继续使用固定 pretrained base，避免缓存随训练漂移。LoRA 与
TriFuse 均保存 raw/EMA 权重，parent checkpoint SHA256 写入隔离的实验合同。

两个前台终端分别运行：

```bash
scripts/run_dual_init_lora_arm_local.sh B 5 /mnt/disk/wl/MoRE-tri/CoopDiff
scripts/run_dual_init_lora_arm_local.sh C 7 /mnt/disk/wl/MoRE-tri/CoopDiff
```

默认配置为 late rank/alpha `4/8`、前四层 rank/alpha `2/4`、LoRA LR
`1e-5`（前层 `5e-6`）、EMA `0.999`、100 epochs。脚本要求 parent 的最终完成标志
`tricoopdiff_zero.pt` 存在，拒绝覆盖已有输出。正式启动前仍需在空闲 GPU 上完成真实
单 batch backward、base 无梯度和 LoRA checkpoint round-trip gate。
