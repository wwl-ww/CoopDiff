# 采样保真度调查任务 brief（派给 agent）

> 面向下一个接手「零-FiLM 基线采样保真度」调查的 agent。本任务**只调查、只诊断、不改代码**。
> 上游结论见 `doc/问题清单与修复状态 交接文档.md` §2。本文是它的可执行展开。

## 0. 背景（已确立的事实，不用再验证）

- 零-FiLM 基线（不传 `--tribrain-ckpt`）应该复现**纯 DiffDock E3-L 二体**的 PROTAC 分布（Go/No-Go 判据）。
- 实测（case `5T35_D_A_759`，32 walkers × 20 steps）：
  - E3-interface DockQ best = **0.215**（旧双重计数 0.153，纯 DiffDock E3-L 目标 **0.844**）。
  - `protac_pocket_dist_mean = 49.8 Å`（32 walker 平均离口袋 50 Å，best 才 5.5 Å）；`protac_rmsd_best = 4.99 Å`。
- **score 层面透明性已确认**：GATE1（`mvp_smoke_test`）证明 `g≡0 → s_q = s_q1`（纯 E3-L，数值差 3.8e-06）。所以 0.215↛0.844 的缺口**不在合并**，而在 **TriCoopDiff 的 E3-L 反向采样循环没有复现 DiffDock 原生 inference**。

## 1. 目标

找出「TriCoopDiff 反向循环」与「DiffDock 原生 inference」之间**具体的、可定位到行号的差异**，判断每个差异的方向与量级影响，**按可能性排序**给出根因假设 + 证据 + 建议修法。目标是回答：为什么零-FiLM（score 已是纯 E3-L）下 PROTAC 采样会漂到离口袋 ~50 Å、DockQ 只有 0.215 而非 0.844。

**交付物**：一份诊断报告（写入 `doc/` 或直接回报），含：
1. 逐项对照表：`TriCoopDiff 位置` vs `DiffDock 位置` vs `差异` vs `对采样的影响（方向/量级）`。
2. 按可能性排序的根因假设（每个附代码证据行号）。
3. 可选的 CPU-only 数值验证脚本（仿 `scripts/verify_rot_convention.py`，纯 numpy/torch/scipy，不跑模型）。
4. **不提交修复**——报告后由主线程决定是否修。

## 2. 对照清单（两边逐项 diff）

### 2.1 初始化（t=1）

- TriCoopDiff：`tricoopdiff/diffusion.py` `TernaryInitializer.initialize_protac`（平移球/高斯 + 旋转 + 扭转初始）。
- DiffDock：`../DiffDock/utils/sampling.py` `randomize_position`（L16）。
- 对照点：平移初始分布（是否以口袋中心为均值、σ 是否与 `tr_sigma_max` 一致）、旋转初始（DiffDock 用 IGSO3 `so3.py:sample`/`sample_vec`，TriCoopDiff 用 `sample_so3_noise`/`sample_so3_noise_wrapped`——**核对是否同分布**）、扭转初始（DiffDock `no_torsion`/均匀 vs TriCoopDiff）。

### 2.2 调度（t → σ 映射 + 步数）

- TriCoopDiff：`tricoopdiff/diffusion.py` `get_inference_schedule`（线性 t=1→0，所有 DOF 共用）、`t_to_sigma`。
- DiffDock：`../DiffDock/utils/diffusion_utils.py` `t_to_sigma`（L28）、`get_t_schedule`（L138）、`../DiffDock/inference.py` 里的 `tr_schedule/rot_schedule/tor_schedule` 构造（含 `inf_sched_alpha`/`inf_sched_beta`）。
- 对照点：**步数是否一致**（eval 用 `steps=20`；DiffDock 原生 `inference_steps` 默认多少？）、t 的离散方式（线性 vs `get_t_schedule` 的 alpha/beta 调度）、σ_min/σ_max、tr/rot/tor 是否用不同 σ（TriCoopDiff 是否把它们共用成一个 t）。

### 2.3 前向噪声施加（旋转约定）

- TriCoopDiff：`tricoopdiff/diffusion.py` `add_noise_protac`（**已加 `.transpose(-1,-2)`**，见问题清单 §1.1）。
- DiffDock：`../DiffDock/utils/diffusion_utils.py` `modify_conformer`（L35）/`modify_conformer_coordinates`（L81）。
- 对照点：**逐行核对** tr/rot/tor 的更新顺序与质心处理——DiffDock 是「先绕质心旋转、再加平移」，TriCoopDiff 现在的写法是否完全一致；旋转矩阵是 `@ R` 还是 `@ Rᵀ`（我修成了 `@ Rᵀ`，需与 `modify_conformer` 逐字核对是否还有第二处约定错）。

### 2.4 反向 tr/rot 步（离散化 + score 归一化）

- TriCoopDiff：`tricoopdiff/diffusion.py` `reverse_step_protac`（含 `so3_score_norm`、`torus_score_norm` 的乘入位置）。
- DiffDock：`../DiffDock/utils/sampling.py` `sampling`（L69）主循环里的 tr/rot 反向更新。
- 对照点：Euler-Maruyama vs 论文反向 SDE 的离散形式是否一致；`dt`、`sqrt(dt)` 的噪声项（非末步 `add_noise`）；**score 归一化因子 `so3_score_norm`/`torus_score_norm` 是否乘对、乘在什么量上**（这是最容易差一个数量级的地方）。

### 2.5 反向扭转步

- TriCoopDiff：`tricoopdiff/diffusion.py` `reverse_step_torsion`。
- DiffDock：`../DiffDock/utils/sampling.py` `sampling` 主循环里的 tor 反向（torus reverse）。
- 对照点：扭转角更新公式、`mask_rotate`/`edge_mask` 的作用方式、扭转 score 归一化。

### 2.6 反向 SE3 步（次要，零-FiLM 下 SE3 head 输出≈0）

- TriCoopDiff：`tricoopdiff/diffusion.py` `reverse_step_se3`。
- 零-FiLM 时 `s_se3≈0`，POI 基本是随机游走（实测 `poi_com_dist_mean≈290 Å`）。这与 E3-PROTAC DockQ 主缺口关系不大，但顺手确认 `reverse_step_se3` 是否也自洽（是否放大了某方向）。

## 3. 环境与约束（重要）

- **SLURM 集群，不在登录节点跑模型 / 不 import `tricoopdiff`**（需要 conda env `tricoopdiff` + e3nn/torch_geometric，且占用计算）。只允许 `python -m py_compile <file>` 做纯语法检查。
- 需要数值验证时写 CPU-only 脚本（纯 numpy/torch，加载 `.so3_*` cache `.npy`，仿 `scripts/verify_rot_convention.py`），经 `sbatch` 跑。
- **不要改代码**（本任务只诊断）。尤其不要碰：`diffusion.py` 里已修好的 `.transpose(-1,-2)`（旋转约定）、`merging.py`/`model.py`/`tribrain.py`（合并）、`train.py`（#2/#4 刚修）、以及工作区其它未提交改动。
- DiffDock 源码两处可用：卖方的 `tricoopdiff/diffdock/`（模型实际跑在这上面）与兄弟 checkout `../DiffDock/`（含 `utils/sampling.py`、`inference.py`）。反向循环的「原生参照」在 `../DiffDock/utils/sampling.py`。

## 4. 验收

调查成功 = 报告能解释「为什么 score 是纯 E3-L 时，采样仍漂到 ~50 Å、DockQ 0.215」。理想情况下指出 1–3 个**具体行号级**差异，每个附方向/量级估计。最终修复后的判据仍是：零-FiLM 基线 E3-interface DockQ ≈ 0.844（同 seed 下 <0.02）。
