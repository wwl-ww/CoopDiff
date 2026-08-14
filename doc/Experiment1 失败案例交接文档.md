# Experiment 1 失败案例交接文档

> 本文档用于交接 Experiment 1（DiffDock-L 基线推理）中 5 个 NaN 失败案例的调查结论、已应用的修复、以及留给下一个 AI 的后续任务。

## 1. 背景与当前状态

Experiment 1 的目标：用冻结的 DiffDock-L（20M 参数）对 PROTAC 基准（23 个三元复合物，每个拆成 E3+PROTAC `e3l` 与 POI+PROTAC `poil` 两个配对，共 46 个配对）跑基线推理，得到 DockQ / RMSD / 冲突率等指标，作为 TriCoopDiff 的对比基准。

经过多轮修复，成功率从最初的 27/46 提升到 **41/46（89.1%）**。剩下 5 个配对的失败模式从早期的 "No edges and no nodes"（受体被裁剪为空）变为纯数值问题：**预测配体坐标含 NaN**，报错 `'x' must be finite, check for nan or inf values`（在 `compute_fnat` → `query_ball_point` 处抛出，脚本第 433 行附近）。

最新完整基线结果文件：`results/diffdock_baseline_v5.json`（`--samples 10`，5 个配对报错）。

## 2. 五个失败案例清单

| 案例 | 配对 | 子图 | PROTAC 配体 | 备注 |
|------|------|------|-------------|------|
| `6W8I_D_A_TKY` | e3l | E3 + PROTAC | TKY | 6W8I 两个复合物均失败 |
| `6W8I_F_C_TKY` | e3l | E3 + PROTAC | TKY | 同一配体 TKY |
| `6ZHC_A_D_QL8` | poil | POI + PROTAC | QL8 | |
| `7KHH_C_D_WEP` | poil | POI + PROTAC | WEP | |
| `7Q2J_C_D_8KH` | poil | POI + PROTAC | 8KH | |

规律：2 个 e3l（同一 PROTAC 配体 TKY 的两个不同 E3 复合物）+ 3 个 poil。**失败集中在 PROTAC 配体本身较大的、配体在扩散早期漂移最远的案例**。

## 3. 已确认的根本原因（有硬证据，NAN_TRACE 逐层定位）

通过向 `cg_model.py` / `tensor_layers.py` 注入 `NAN_TRACE=1` 逐层诊断（`tp_scatter_multigroup` 记录 `fc_max`/`sh_max`/`tp_max`，conv 循环逐层记录 `node_attr.absmax`）确认：

- **NaN 首次出现在第 7 层**（`l=7`），直接原因是 **`FCBlock` 输出的每条边张量积权重 `fc_max` 爆到 4.5e22**，随后 `node_attr` 溢出为 NaN。
- **不是退化图 / 零长度边**：全程 `edge_sh` 有界（`sh_max≈2.235`）、`edge_sh_finite=True`、失败时 `div_min=8`（边存在）。之前第 4 节推断的"交叉边裁空→孤立配体→稀疏图上发散"**被实测证伪**。
- **不是 0/0 除零**：`tr_norm`/`rot_norm` 不接近 0，`tr_pred/tr_norm` 只是下游症状。
- **真实机制是 FCBlock→TensorProduct 正反馈指数爆炸**：`node_attr[:ns]` 标量被拼进 `edge_attr_` → FCBlock 线性层算出每条边权重 → 权重乘回 `node_attr` 生成下一层 `node_attr`。该回路谱半径 >1 时逐层放大：

| 层 | lig_absmax | fc_max（边权重） | tp_max（张量积） |
|----|-----------|-----------------|-----------------|
| l=2 | 5.0e1 | 1.0e4 | 7.5e4 |
| l=3 | 6.4e3 | 1.7e4 | 3.2e6 |
| l=4 | 3.9e6 | 8.9e5 | 1.3e9 |
| l=5 | 2.1e11 | 7.3e7 | 5.4e13 |
| l=6 | 2.0e21 | 8.1e12 | 2.9e23 |
| l=7 | NaN | 4.5e22 | 溢出 |

每层放大 ~10²–10⁴，到第 7 层 float32（max 3.4e38）溢出。健康前向 `node_attr.absmax` 全程 ≤ ~3e2，与溢出区间（l=3 起 6e3）之间有清晰鸿沟。

- 触发条件：**后段扩散步**（`tr_sigma` 0.37→0.17），此时 cross edge 收缩、蛋白/配体位置收敛，`node_attr` 激活进入正反馈。**依赖随机种子**（同一案例某些种子正常、某些种子溢出）。

## 4. 成因机制（实测确认，非假设）

1. `node_attr` 标量块（`:48`）是**无界**的，被拼入每条边特征后经 FCBlock（2 层 MLP，无归一化/有界激活）产生张量积权重。
2. 权重与 `node_attr` 相乘（`tp`）生成下一层 `node_attr`，形成**乘法正反馈**。
3. 绝大多数前向该回路稳定（`node_attr` 收敛在 ~10²），但大型 PROTAC 配体在晚期扩散步会进入发散分支，逐层指数放大直至 float32 溢出。

> 关键点：`crop_beyond` 关闭只是必要条件；cross edge 动态收缩（`tr_sigma*3+20` 硬编码）会压缩有效图，但**根因是数值反馈爆炸，不是图退化**。

## 5. 已应用的修复

### 5.1 脚本层（`scripts/experiment1_diffdock_baseline.py`）

- **默认关闭 crop_beyond**：`--crop-beyond` 缺省时把 `score_model_args.crop_beyond = None`（原默认 20）；也可 `--crop-beyond X` 显式覆盖。
- **NaN 样本过滤**：计算指标时，跳过坐标非有限的样本（标记 `failed=True`），只在有限样本内按 DockQ 排序取 best；若全部样本 NaN 则记 `error: all_samples_nan`。每配对额外记录 `n_nan_samples`。
- 更早修复：`_scatter_pytorch` OOB 索引保护；`build_complex_pdb` 双字母元素原子名修复；7PI4 配体 SDF→PDB 回退。

### 5.2 DiffDock 依赖层（`./DiffDock`，分支 `nan-stability-fixes`）

- **根因修复（Bug A）**：`models/cg_model.py` 的 conv 循环每层后加 `node_attr = torch.clamp(node_attr, ±NAN_CLAMP)`，钳住反馈回路的状态变量，从源头消除 NaN。由环境变量 `NAN_CLAMP` 门控（缺省关闭，不改 DiffDock 默认行为）。阈值 `1e4` 比健康前向上限（~3e2）高 30 倍、比溢出起点（l=3 的 6e3）高，干净分离两个区间，健康案例完全不受影响（bit-identical）。
- **回退门修复（Bug B）**：`utils/sampling.py` 的 NaN 回退原先只用 `torch.mean(tr_score)` 计数，`rot_score`/`tor_score` 里的 NaN 会漏过回退并传播到最终配体坐标。已改为对三者任一 NaN 都触发回退。
- 更早修复：`tor_bonds[0]` 的 `edge_sigma` 索引；空 `edge_index`/残差形状守卫；forward 前 deepcopy；`nanmean` 全 NaN 回退 0.01。

### 5.3 诊断代码（保留，`NAN_TRACE=1` 门控）

`cg_model.py` / `tensor_layers.py` 里留了 `NAN_TRACE` 逐层诊断（`[TRACE init/l/TP]`），缺省关闭，需要复现/再定位时 `export NAN_TRACE=1` 打开。

## 6. 验证结果

### 6.1 根因修复验证（`nan_fix_verify`）

`scripts/experiment1_nan_localize.slurm`（`--samples 20 --inference-steps 20 --seed 0`，`NAN_CLAMP=10000`）对 5 个失败案例全跑，**NaN samples 全部为 0**，结果在 `results/nan_fix_verify.json`：

| 配对 | DockQ | RMSD(Å) | NaN 样本数 |
|------|-------|---------|------------|
| 6W8I_D_A_TKY/e3l | 0.0854 | 8.02 | 0 |
| 6W8I_D_A_TKY/poil | 0.5737 | 16.96 | 0 |
| 6W8I_F_C_TKY/e3l | 0.0471 | 8.99 | 0 |
| 6W8I_F_C_TKY/poil | 0.4423 | 12.36 | 0 |
| 6ZHC_A_D_QL8/e3l | 0.4696 | 10.46 | 0 |
| 6ZHC_A_D_QL8/poil | 0.3473 | 11.68 | 0 |
| 7KHH_C_D_WEP/e3l | 0.5217 | 10.90 | 0 |
| 7KHH_C_D_WEP/poil | 0.4540 | 10.99 | 0 |
| 7Q2J_C_D_8KH/e3l | 0.6653 | 8.13 | 0 |
| 7Q2J_C_D_8KH/poil | 0.8098 | 12.37 | 0 |

汇总：E3-L DockQ=0.3578±0.2469；POI-L DockQ=0.5254±0.1593；Overall n=10 DockQ=0.4416±0.2240。

> 说明：`6W8I_*_TKY/e3l`（DockQ 0.05–0.09）等仍属低质量对接——这是**真实难度**（大 PROTAC 配体），不是 NaN 伪影，如实保留在基线里。

### 6.2 早期兜底运行（nan_diag，已被根因修复取代）

早期仅靠"NaN 过滤"兜底的 `nan_diag` 运行（`results/nan_diag_test.json`）显示 5 个配对虽有 0–10% 样本 NaN 但 best 恢复；现在根因修复后 NaN 样本为 0。

## 7. 留给下一个 AI 的待办

1. **[进行中] 重跑完整基线**：`scripts/experiment1_diffdock_baseline_v6.slurm`（`--samples 10 --inference-steps 20 --batch-size 10 --seed 0`，`NAN_CLAMP=10000`）对全部 23 案例 / 46 配对跑，产出 `diffdock_baseline_v6.json`，确认 46/46 无 NaN。**已提交（job 待 GPU 释放）**。
2. **[建议] A/B 测试 crop_beyond**：`--crop-beyond 20` vs `None` 各跑一遍，量化动态裁剪对命中率与 DockQ 的影响。
3. **[可选] 阈值调优**：`NAN_CLAMP=10000` 是保守值；若想更贴近模型原始分布，可测 `1000`（健康上限 ~3e2 仍有 3 倍裕量）是否同样消除 NaN 且对低质量案例 DockQ 影响更小。
4. 清理诊断打印：`[NaN DIAG]` / `[DEBUG torsion mismatch]` / `NAN_TRACE` 在正式跑分时缺省静默（均已门控），无需处理。

## 8. 关键文件与代码位置索引

| 位置 | 内容 |
|------|------|
| `scripts/experiment1_diffdock_baseline.py` | 主脚本：NaN 过滤（~756-830 行）、`--crop-beyond` 与 crop_beyond 关闭逻辑（~561-571 行） |
| `scripts/experiment1_nan_localize.slurm` | 根因修复验证运行（5 案例，`NAN_CLAMP=10000`，已完成） |
| `scripts/experiment1_diffdock_baseline_v6.slurm` | 完整基线 v6（`NAN_CLAMP=10000 --seed 0`，待 GPU） |
| `scripts/experiment1_nan_diag.slurm` | 早期兜底诊断运行（已被根因修复取代） |
| `scripts/experiment1_full.slurm` / `_rerun_failed.slurm` | 完整 / 重跑失败案例的 SLURM 模板 |
| `./DiffDock/models/cg_model.py` | conv 循环 `NAN_CLAMP` 钳位（根因修复）、`NAN_TRACE` 诊断、tor_bonds[0] 修复 |
| `./DiffDock/models/tensor_layers.py` | `NAN_TRACE` TP 诊断、空 edge_index 守卫 |
| `./DiffDock/utils/sampling.py` | deepcopy + NaN 回退门修复（Bug B） |
| `results/diffdock_baseline_v5.json` | 修复前完整基线（5 报错） |
| `results/nan_fix_verify.json` | 根因修复验证结果（10/10 有限） |

## 9. Git 状态

- **CoopDiff**（分支 `fix-nan-root-cause`）：已合并入 `master`。
- **DiffDock**（分支 `nan-stability-fixes`）：已合并；DiffDock 现位于仓库内 `./DiffDock`。
