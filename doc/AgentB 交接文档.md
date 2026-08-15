# Agent B 交接文档 — model.py identity-rotation 修复 + POI 指标

> 由 Agent A（采样 + 评估闭环）写给 Agent B。本文只覆盖 B 的职责范围：
> **（1）修 `model.py` 的 E3→POI 单位旋转占位；（2）在 `evaluate.py` 里补齐 POI / DockQ 指标。**
> A 已把采样闭环（`sampler.py` + `evaluate_entry.py`）跑通，B 的工作不要动 A 的文件（见文末契约）。

## 0. 当前仓库状态（B 接手前先读）

- 分支 `mvp-integration`。A 新增/改动：`tricoopdiff/diffusion.py`（新增 `reverse_step_torsion` + 修 `reverse_step_protac` 温度采样 tensor 化）、`tricoopdiff/sampler.py`（新增）、`tricoopdiff/evaluate.py`（实现 `evaluate_model`）、`scripts/evaluate_entry.py`（新增）、`scripts/evaluate_entry.slurm`（新增）。
- `model.py` **尚未被 A 改过**，identity-rotation 问题仍在，这正是 B 的活。

## 1. 任务目标

1. `model.py:forward` 里 PROTAC 坐标变换到 POI 帧时，把硬编码的 `torch.eye(3)` 换成**真实的 E3→POI 相对旋转**，并在采样循环里正确追踪、传入该旋转。
2. 在 `evaluate.py` 里实现 POI 结构质量指标（POI RMSD、DockQ / 界面接触恢复），并把结果接进 `evaluate_model` 的返回 dict。
3. 不改动 A 的文件、不破坏 `model.forward` 的现有输出契约（B 只加一个**默认 identity 的可选入参**）。

## 2. 背景与坐标约定（关键，先吃透再动手）

- 参考系：E3 固定在原点。PROTAC 坐标一次性采样在 **E3 全局帧**，subgraph1（E3-L）直接用；subgraph2（POI-L）要把它变换到 POI 帧。
- 变换函数在 `tricoopdiff/graphs.py:207`：

```python
def transform_ligand_to_poi_frame(ligand_coords, t_e3_to_poi, r_e3_to_poi):
    return (ligand_coords - t_e3_to_poi) @ r_e3_to_poi   # 右乘旋转
```

- 调用点在 `tricoopdiff/model.py:113-122`：

```python
e3_center  = e3_data['coords'][b].mean(dim=0)
poi_center = poi_data['coords'][b].mean(dim=0)
t_e3_to_poi = poi_center - e3_center
ligand_coords_poi = transform_ligand_to_poi_frame(
    ligand_data['coords'][b], t_e3_to_poi,
    torch.eye(3, device=device),      # ← BUG：硬编码单位旋转
)
```

- **问题**：SE3 扩散对 POI 施加的是 6 自由度的刚体运动（3 平移 + 3 旋转）。但 `model.forward` 重建 subgraph2 时，只用了平移 `poi_center - e3_center`，旋转写成单位阵。一旦 POI 被 `reverse_step_se3` 转过、或 `initialize_poi` 给过随机初始朝向，PROTAC 在 POI 帧里的**朝向就是错的**（差一个 SE3 旋转），subgraph2 的 FiLM/score 输入就不一致。

### 旋转是怎么产生的（B 要追踪的就是它）

- `tricoopdiff/diffusion.py:562` `initialize_poi`：`rot = R.random(B)`，`poi_pos = poi_template @ rot + translation`。
- `tricoopdiff/diffusion.py:429` `reverse_step_se3`：每步 `rot_matrix = so3_exp(rot_perturb)`，绕 POI 质心作用：`poi_pos = poi_center + poi_centered @ rot_matrix`。

即 POI 相对 E3 的**累积旋转** = 初始随机旋转 与 每步 SE3 逆扩散旋转 的组合。B 需要：
1. 在 `sampler.py` 的采样循环里维护一个累积旋转矩阵（初始化 = `initialize_poi` 用的那个随机旋转；每步乘上 `reverse_step_se3` 的 `so3_exp(rot_perturb)`）。
2. 把它作为新的可选参数传给 `model.forward`，再一路传给 `transform_ligand_to_poi_frame`。

> 注意右乘/转置的合成顺序：`transform_ligand_to_poi_frame` 是 `@ r_e3_to_poi`（右乘），而 POI 更新是 `poi_centered @ rot_matrix`。两处的旋转方向约定相反，**合成顺序和是否转置必须用数值实验核对**（见第 5 节验收）。别凭直觉写，用一个已知朝向的 toy 向量验证：`transform_ligand_to_poi_frame` 之后，PROTAC 与 POI 的相对几何应当和 native 复合物一致。

## 3. 第一部分：修 identity rotation（最小改动）

- `model.py:forward` 签名加一个可选参：`r_e3_to_poi: torch.Tensor = None`（`[B,3,3]`，默认 `None` → 内部回退 `torch.eye(3)`，保证训练/smoke 现有调用不改也能跑）。
- 循环里用它替换 `torch.eye(3, device=device)`。
- `sampler.py` 追踪累积旋转并传入。注意：A 的 `sample_single` 目前只调 `model(e3_data, poi_d, lig_d, t=t, t_se3=t_se3)`，B 加参数后要保证**不传时的默认行为 == 现在的 identity**（这样 A 的零-FiLM 基线不受影响）。
- `initialize_poi` 返回的随机旋转目前没暴露出来，B 要么让 `initialize_poi` 同时返回 `rot`，要么在 `sample_single` 里单独采样同一个随机旋转。选一种，保持单一真相。

## 4. 第二部分：POI 指标（跨帧参考是难点）

MVP 阶段 A 只评了 PROTAC（E3-L）pose，POI 指标**故意没写**，原因是**参考帧不一致**：

- `e3_data['coords']` 和 `poi_data['coords']` 是 `scripts/featurize_common.py` 里 `featurize_subgraph` 对 E3、POI **各自独立**跑 `InferenceDataset` 得到的，各自在 DiffDock 自己的蛋白预处理参考系里，**不在同一个全局帧**。
- 所以不能直接 `compute_rmsd(poi_pred, poi_data['coords'])` —— 两个 POI 坐标的帧对不上。

### 建议方案（B 自选其一，写在 docstring 里说明理由）

- **方案 1（推荐）**：用 case 目录里的 native 三元复合物 PDB（`DeepTernary/output/protac22/<case>/complex.pdb` 或 `gt_complex.pdb`）解析出 ground-truth 的 E3-POI 刚体变换；把预测的 E3 帧 POI 坐标对齐到该 native 帧，再算 POI RMSD / DockQ。
- **方案 2**：只评**旋转不变 / 帧无关**的量 —— 例如预测 POI 与 native POI 各自对 PROTAC 的**界面接触图**的重合（DockQ 本质就是界面接触恢复率），绕开全局帧对齐。

### 要写/接的指标

- `evaluate.py` 已有 `compute_dockq`（`evaluate.py:20`）、`compute_rmsd`（`evaluate.py:60`）、`compute_rmsd_by_region`（`evaluate.py:82`）、`compute_clash_rate`（`evaluate.py:130`）。B 复用它们，新增一个「POI 相对 PROTAC 的 pose 评估」入口（比如 `compute_poi_dockq(pred_e3_frame_poi, pred_ligand, native_complex_pdb)`）。
- 把结果填进 `evaluate_model` 的 `structure` 字典（当前 `evaluate.py:389` 只有 `protac_*` 键，加 `poi_*` / `dockq` 键）。**A 的文件只允许在这里加字段，别改 A 已有的 `protac_*` 逻辑。**

## 5. 验收标准（B 自测）

1. 数值 sanity check：构造两个已知朝向的 E3/POI，验证 `transform_ligand_to_poi_frame` 用累积旋转后，PROTAC 在 POI 帧的相对几何正确（对照 native 复合物的界面接触）。
2. 零-FiLM 基线（`sbatch scripts/evaluate_entry.slurm`，`--tribrain-ckpt` 不传）里，SE3 head 零初始化 → `s_se3≈0` → 累积旋转≈恒等。此时 B 的改动**必须让 `evaluate_model` 的 `protac_rmsd_best` 与 A 记录的基线一致**（即 identity 默认路径不改变结果）。
3. POI 指标能在 `evaluate_entry.py` 里正常产出、非 NaN。

## 6. 与 Agent A 的契约（不要破坏）

- `model.forward` 的返回 dict 结构不变：`{'s_q': {'tr','rot','tor'}, 's_se3': [B,6], 's_se3_components': {...}, 'confidence': [B,1], 'state': [B,2]}`。
- B=1 数据 dict 格式不变（`featurize_common` 的 `coords`/`features`/`edge_index`/`edge_attr`/`edge_mask`/`mask_rotate`）。
- 新增入参必须**默认 identity**，让 `scripts/mvp_smoke_test.py`、`scripts/train_entry.py` 等 A/训练侧的现有调用无需改动。
- **B 只改 `model.py`、`evaluate.py`、`sampler.py`（追踪旋转那一小块）、可能 `graphs.py`（若需）**；不要碰 `diffusion.py` 的采样数学、`train.py`、`scripts/evaluate_entry.py` 的 CLI。

## 7. 已知相关占位（上下文，非本次必做）

- `model.py:forward` 仍是 B=1 单复合物模式（多复合物 batch 需 DataLoader/padding，另立任务）。
- `train.py:_physical_loss` 的 3.4Å 硬截断、`reflect.py:local_minimization` 的简单梯度下降，都不在 B 范围。
