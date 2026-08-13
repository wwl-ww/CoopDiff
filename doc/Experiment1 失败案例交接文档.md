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

## 3. 已确认的根本原因（有硬证据）

通过向 `DiffDock/models/cg_model.py` 注入 `[NaN DIAG]` 诊断（位于 tr/rot 归一化点）确认：

- **`lig_node_attr` 变为 NaN**（`lig_node_attr finite=False`），发生在 **e3nn 张量积卷积层**（`TensorProductConvLayer`）内。
- 此时 `global_pred` 仍然 finite，且 `tr_norm` / `rot_norm` **不接近 0**。
- 结论：NaN 源自 **e3nn 消息传递节点特征的数值溢出**，触发点在后段扩散步（`tr_sigma ≤ 2.33`），**依赖随机种子**（同一案例某些种子正常、某些种子溢出）。
- 之前怀疑的 `tr_pred / tr_norm` 的 0/0 除零，被证明只是**下游症状**，不是根因（诊断显示 tr_norm 并不为 0）。

## 4. 推断的成因机制（假设）

综合两层裁剪机制，失败机制推断如下：

1. **`crop_beyond=20`（DiffDock 默认）**：每步把受体裁到 `tr_sigma*3 + 20` Å。当配体漂移远时，受体被裁成空集 → 报 "No edges and no nodes"（这是早期失败，已被脚本默认禁用 crop_beyond 解决）。
2. **`dynamic_max_cross: true`（硬编码）**：即使 `crop_beyond` 关闭、受体保留全原子，交叉边仍然被 `cg_model.py` 里硬编码的 `tr_sigma*3 + 20` 动态收缩（**这个值不来自 crop_beyond 配置，改 crop_beyond 不影响它**）。后段 `tr_sigma` 很小时，配体与受体的交叉边被裁空 → 配体在消息传递图上近乎孤立 → e3nn 卷积在稀疏/退化图上数值发散 → `lig_node_attr` 溢出为 NaN。
3. 这 5 个配对恰好是配体漂移最远、几何最难的案例，因此最容易被上述机制命中。

> 关键点：**关闭 crop_beyond 只是必要条件，不足以根治**，因为 cross edge 的动态收缩是独立硬编码的。

## 5. 已应用的修复

### 5.1 脚本层（`scripts/experiment1_diffdock_baseline.py`）

- **默认关闭 crop_beyond**：`--crop-beyond` 缺省时把 `score_model_args.crop_beyond = None`（原默认 20）；也可 `--crop-beyond X` 显式覆盖。
- **NaN 样本过滤**：计算指标时，跳过坐标非有限的样本（标记 `failed=True`），只在有限样本内按 DockQ 排序取 best；若全部样本 NaN 则记 `error: all_samples_nan`。每配对额外记录 `n_nan_samples`。
- 更早修复：`_scatter_pytorch` OOB 索引保护；`build_complex_pdb` 双字母元素原子名修复；7PI4 配体 SDF→PDB 回退。

### 5.2 DiffDock 依赖层（`../DiffDock`，分支 `nan-stability-fixes`）

- `models/cg_model.py`：修 `tor_bonds[0]` 的 `edge_sigma` 索引；注入 `[NaN DIAG]` 定位 NaN 来源。
- `models/tensor_layers.py`：`TensorProductConvLayer` 对空 `edge_index` 的守卫 + 残差形状守卫。
- `utils/sampling.py`：每次 forward 前 deepcopy（避免原地修改污染后续扩散步）；NaN score 稳健回退（`nanmean` 全 NaN 时回退 0.01）。

> 注意：这些是**缓解/兜底**手段（让案例能产出有限指标），并未消除 NaN 的**根因**（e3nn 数值溢出本身）。

## 6. 验证结果（nan_diag 运行）

`scripts/experiment1_nan_diag.slurm`（`--samples 20 --inference-steps 20`，crop_beyond 关闭 + NaN 过滤）已完成，结果在 `results/nan_diag_test.json` / `results/diffdock_nan_diag/`。**5 个失败配对全部恢复**，NaN 样本占比仅 0–10%：

| 配对 | DockQ | RMSD(Å) | NaN 样本数/20 |
|------|-------|---------|---------------|
| 6W8I_D_A_TKY/e3l | 0.2645 | 15.01 | 2 |
| 6W8I_F_C_TKY/e3l | 0.6794 | 16.69 | 0 |
| 6ZHC_A_D_QL8/poil | 0.1550 | 10.20 | 0 |
| 7KHH_C_D_WEP/poil | 0.4254 | 8.31 | 1 |
| 7Q2J_C_D_8KH/poil | 0.5787 | 9.32 | 1 |

说明：恢复出的案例中，`6ZHC_A_D_QL8/poil`（DockQ=0.155）、`6W8I_D_A_TKY/e3l`（DockQ=0.265）仍属低质量对接——这是**真实难度**，不是 NaN 伪影，应保留在最终基线里如实报告。

## 7. 留给下一个 AI 的待办

1. **[必须] 重跑完整基线**：用 NaN 过滤 + crop_beyond 关闭的脚本，对全部 23 案例 / 46 配对跑正式配置（`--samples 10` 或对齐最终设定的样本数），产出 `diffdock_baseline_v6.json`，确认 46/46 无报错，得到最终基线指标。
2. **[建议] A/B 测试 crop_beyond**：`--crop-beyond 20` vs `None` 各跑一遍，量化动态裁剪对命中率与 DockQ 的影响，验证第 4 节的机制推断。
3. **[可选] 根因根治**：若想在源头消除 NaN（而非过滤），方向是给 e3nn `TensorProductConvLayer` 加数值稳定化——float32→float64、特征 clamp/重归一化、或对退化图做显式短路。诊断代码（`[NaN DIAG]`）已留在 `cg_model.py` 便于复现。
4. 清理或保留诊断打印（`[NaN DIAG]` / `[DEBUG torsion mismatch]` / scatter OOB 提示），正式跑分时可选择静默。

## 8. 关键文件与代码位置索引

| 位置 | 内容 |
|------|------|
| `scripts/experiment1_diffdock_baseline.py` | 主脚本：NaN 过滤（~756-830 行）、`--crop-beyond` 与 crop_beyond 关闭逻辑（~561-571 行） |
| `scripts/experiment1_nan_diag.slurm` | 5 案例诊断运行（已完成） |
| `scripts/experiment1_full.slurm` / `_rerun_failed.slurm` | 完整 / 重跑失败案例的 SLURM 模板 |
| `../DiffDock/models/cg_model.py` | `[NaN DIAG]`、cross-edge 硬编码 `tr_sigma*3+20`、tor_bonds[0] 修复 |
| `../DiffDock/models/tensor_layers.py` | 空 edge_index 守卫（e3nn 卷积层） |
| `../DiffDock/utils/sampling.py` | deepcopy + NaN score 回退 |
| `results/diffdock_baseline_v5.json` | 修复前完整基线（5 报错） |
| `results/nan_diag_test.json` | 诊断运行结果（5 全恢复） |

## 9. Git 状态

- **CoopDiff**（分支 `deepternary-paper-analysis`）：提交 `8c6e6e6`，含脚本 + 5 个 slurm 文件。
- **DiffDock**（分支 `nan-stability-fixes`）：提交 `47eb91a`，含 `cg_model.py` / `tensor_layers.py` / `sampling.py`。