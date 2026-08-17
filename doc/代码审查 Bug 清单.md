# 代码审查 Bug 清单

> 审查日期：2026-08-16 | 范围：`tricoopdiff/` 核心模块（model / train / tribrain / diffusion / sampler / evaluate / graphs / merging / film）
>
> 目的：只记录问题，不改代码。按严重度分三级：🔴 严重 / 🟡 中等 / 🟢 轻微+待验证。

---

## 〇、2026-08-17 复审（post debug-se3-loss 合并，`bug-fix` 分支）

对当前 `master`（含 debug-se3-loss 的 11 提交）逐条重验。结论：

| 状态 | Bug |
|------|-----|
| ✅ **已修**（debug-se3-loss 带入） | BUG-3：`add_noise_protac`/`add_noise_se3` 已改**全区间**（docstring 明写 "no λ_q/λ_SE3 cutoff"），`reverse_step_*` 也全 σ 去噪，前后一致。`get_lambda_q/se` 仅剩诊断脚本引用，生产路径已死代码。 |
| ✅ **误报**（确认正确） | BUG-6（rot_score 无需 adjoint）、BUG-7（`r_e3_to_poi` 约定一致）、BUG-9（DockQ d 值 = 标准 iRMS 1.5 / Lrms 8.5，`evaluate.py:457-459` 正确） |
| 🔴 **仍存在** | BUG-18（featurization 独立居中，相对几何被抹）、BUG-20（`../DiffDock` 不存在，实为 `./DiffDock`） |
| 🟡/🟢 **仍存在** | BUG-1、2、4、5、8、10、11、12、13、14、15、16、17、19（16 个） |

行号漂移（仅标题里的行号过期，结论不变）：BUG-2 `224→225`、BUG-17 `84→99`、BUG-5 `335→364/392`、BUG-16 `116→121`。

**优先修**：🔴 BUG-18（训练监督信号无意义，模型学会"三分子堆原点"）+ 🔴 BUG-20（本机训练/评估入口跑不起来）。

**复审补充（逐函数读正文后新发现，均为 🟢 文档/死代码级）**：

- **N-1** `diffusion.py:1-15` 模块 docstring 仍描述"λ_q（t=0.7）/ λ_SE3（t=0.5）multi-rate masking"，但该 masking 已被 debug-se3-loss 删除 → docstring 过时。
- **N-2** `get_lambda_q`/`get_lambda_se3`（`diffusion.py:126-139`）+ `MultiRateDiffusion.__init__` 的 `lambda_q_threshold`/`lambda_se3_threshold` 参数已成**死代码**（生产路径不再调用，仅 `diagnose_loss_se3.py`/`verify_se3_sigma_consistency.py` 诊断脚本引用）。
- **N-3** `train.py:6-8` docstring 的 `s_tri_residual` 是旧设计残留（已记于《参考系与数值梯度专项分析.md》§2.2），且"direct path, not through FiLM"与现状相反。
- **N-4** BUG-8 加深：`MultiScalePooling` 的"short-range(<20Å)/mid-range(<50Å)"距离分层**根本没实现**——三次 pooling（`tribrain.py:121-141`）都作用在**同一个全节点集**上，只换线性投影/query，无距离 cutoff、无 Cα 粗粒化，`node_coords` 参数未被使用（与 BUG-4 同源：坐标盲）。

---

## 🔴 严重

### BUG-20 `featurize_common.py:20` / `evaluate_entry.py:35` 引用 `../DiffDock`（兄弟目录），但 DiffDock 实际在 `./DiffDock`（仓库内）

```python
# featurize_common.py / evaluate_entry.py:
_DIFFDOCK_DIR = _PROJECT_ROOT.parent / "DiffDock"   # ← ../DiffDock（仓库外兄弟）
```

- 实测（本机）：`./DiffDock`（仓库内，git-ignored）**存在**，且含 featurization 所需 `datasets/process_mols.py`、`utils/inference_utils.py`；而 `../DiffDock`（兄弟）**不存在**。
- `.gitignore` 里写的是 `DiffDock/`（仓库内），说明 canonical 位置是**仓库内**，但代码却找仓库外。
- 后果：`featurize_common.py` 会 `raise RuntimeError("DiffDock checkout not found")`，`evaluate_entry.py` 会 FATAL 退出；`train_entry.py` 经由 `featurize_common` 也被连累。**训练/评估入口在本机直接跑不起来。**
- 根因：路径改过（`../` ↔ `./`），`featurize_common`/`evaluate_entry` 没跟上，且新旧 CLAUDE.md 对此说法互相矛盾（docs/reorg 版写 `./DiffDock`，mvp 版写 `../DiffDock`）。

### BUG-1 `evaluate.py:20` `compute_dockq` 实现错误，且是死代码

```python
pred_dist = cdist(pred_coords, pred_coords)   # 分子内自距离
ref_dist  = cdist(ref_coords, ref_coords)
ref_contacts  = ref_dist  < interface_cutoff
pred_contacts = pred_dist < interface_cutoff
recovered = (ref_contacts & pred_contacts).sum()
dockq = recovered / ref_contacts.sum()
```

- **问题**：这里算的是**分子内 contact-map 相似度**（每个坐标集对**自身**求 pairwise 距离，看哪些原子互相靠近），**不是 DockQ**。DockQ 的 Fnat 是「两个蛋白**之间**的界面接触恢复率」，需要分子间（如 E3↔POI）的接触对，且还要叠加 iRMS/Lrms 两项。
- docstring 自己也写了「one from each protein」，但代码里 `cdist(coords, coords)` 是单分子自距离，根本没有「跨分子」。
- **影响**：`evaluate_model` 其实**没调用**这个函数（用的是 `compute_e3_interface_dockq` / `compute_poi_dockq`），所以是**死代码 + 错误实现**，一旦有人误用会得到无意义的分数。

---

## 🟡 中等

### BUG-2 `model.py:225` 批量堆叠用 `.squeeze(0)`，对 B>1 是错的

```python
s_q = {
    k: torch.stack(v, dim=0).squeeze(0) if len(v[0].shape) <= 2
    else torch.cat(v, dim=0)
    for k, v in all_s_q.items()
}
```

- 每个 `v[b]` 是 `[1, 3]`。`torch.stack(v, dim=0)` → `[B, 1, 3]`，`.squeeze(0)`：
  - B=1 → `[1, 3]` ✅（侥幸）
  - B>1 → `.squeeze(0)` 不动（dim0 是 B≠1）→ `[B, 1, 3]` ❌（应该是 `[B, 3]`）
- 当前 B=1 是唯一模式，所以暂时没炸，但这是埋着的雷。

### BUG-3 ~~多速率 λ 在 forward 有 mask、reverse 没有，前后不一致~~（✅ **已修**，debug-se3-loss 合并）

- **forward**：`diffusion.add_noise_protac` / `add_noise_se3` 用 `λ_q` / `λ_se3` 缩放噪声（`tr_update = randn * σ * λ`）。
- **reverse**：`reverse_step_protac` / `reverse_step_se3` 用**全 σ**（`tr_sigma = t_to_sigma(t)`，不乘 λ）。
- 结果：t < 0.7（λ_q=0）时 forward **不加噪**，但 reverse 仍在**继续 denoise**——PROTAC 在已经该收敛的阶段还在被拖动，可能过denoise。
- 模型文档明确写了「去噪时：更新步长乘以对应的 λ²」，代码没实现。

### BUG-4 `tribrain.py:243` `EquivariantTransformer` 名不副实——坐标/特征根本没被用

```python
def forward(self, pooled, node_coords, node_features, batch=None):
    ...
    s = self.scalar_proj(pooled)   # 只用 pooled
    v = self.vector_proj(pooled)
    for layer in self.layers:
        s, v = layer(s, v)
```

- `node_coords` 和 `node_features` 是参数但**函数体里完全没用到**。
- 「等变 transformer」既不读坐标、也不读逐节点特征，只处理一个 `[B, node_dim]` 的 pooled token。向量通道只是 `Linear` 投影 + 标量门控，**没有 docstring 声称的 cross-product，也没有任何几何交互**。
- 后来的 `SE3GeometryEncoder` 就是为弥补这个「坐标盲」才加的（代码注释也承认了），但 `EquivariantTransformer` 这个类名本身有误导性。

### BUG-5 `tribrain.py:364/392` 向量更新的 `LayerNorm(nv*3)` 破坏等变性

```python
v_flat = v.reshape(B, self.nv * 3)
v_flat = self.vector_norm(v_flat)   # nn.LayerNorm(nv*3)
```

- 对展平的所有向量分量做 LayerNorm，会**混合不同空间方向的分量**，O(3) 等变性被破坏（真正等变的做法是只对模长归一化、保留方向）。

### BUG-13 `reflect.py:324,343` `tricoop_reflect` 收了 `bonds` 却不传给 `detect_clashes`

```python
def tricoop_reflect(coords, atomic_numbers, bonds=None, ...):
    ...
    clash_mask = detect_clashes(coords, atomic_numbers)   # ← 没传 bonds
```

- `tricoop_reflect` 的签名里有 `bonds` 参数，但两次调用 `detect_clashes`（初始检测 + 最小化后复查）都**没传 `bonds`**。
- 结果：冲突检测永远走「共价半径 1-2 启发式」，会**把环上的 1-3 原子误判成 clash**（`detect_clashes` 自己的 docstring 也写了这个后果）。
- 对比：`evaluate.py:compute_clash_rate` 是正确地把 `bonds` 传下去的。`tricoop_reflect` 的 `bonds` 参数是死的。

### BUG-17 `train.py:99` `compute_loss` 无条件 `self.model.train()`，把 validate 的 eval 模式顶掉了

```python
# train.py compute_loss 开头：
self.model.train()  # 注释说 "still need train mode for dropout consistency"
```

- `train_entry.py:validate` 先 `trainer.model.eval()`，然后调 `validate_step` → `compute_loss` → **又被 `self.model.train()` 切回 train 模式**。
- 结果：**验证阶段在 train 模式下跑**，TriBrain 里的 `nn.Dropout`/`MultiheadAttention(dropout)` 全开，val_loss 是带 dropout 噪声的，`best_model` 的挑选也因此受影响。
- `@torch.no_grad()` 只挡梯度，挡不住 dropout。正确做法是 `compute_loss` 不碰模式，由 `train_step` 显式 `train()`、`validate_step` 显式 `eval()`。

### BUG-19 `scripts/inference.py` + `scripts/eval_entry.py` 是过时的并行入口，跑起来会产垃圾

- `inference.py` 用 `parse_protein`（`features = zeros(n_res, 1)` 占位零特征）和 `parse_ligand`（原子序数当特征），自己的 `load_model` 走 `from models.cg_model import CGModel`（`../DiffDock` 老路径，非 vendored `expert_factory`），自己的 `run_reverse_diffusion` 循环（非 `sampler.py`）。
- `eval_entry.py` 依赖 `inference.load_model`，同样过时。
- 真正的当前入口是 **`evaluate_entry.py` → `featurize_common.featurize_complex`（真实 ESM2 特征）→ `expert_factory.build_diffdock_expert`（vendored, strict）→ `evaluate_model`/`sampler.sample_ensemble`**。
- 后果：`inference.py`/`eval_entry.py` 与真实 pipeline **完全分叉**——若有人按 CLAUDE.md 或文件名误跑，会拿到零特征喂进模型，静默产出无意义结果。要么删掉，要么加「legacy，勿用，请用 evaluate_entry.py」的醒目提示。

---

## 🟢 轻微 / 待验证

### BUG-6 ~~`model.py:201-203` frame fix 的 rot_score 旋转~~（**误报，已核对正确**）

```python
r_poi_to_e3 = r_e3.transpose(-1, -2)
scores_2['tr_score'] = scores_2['tr_score'] @ r_poi_to_e3
scores_2['rot_score'] = scores_2['rot_score'] @ r_poi_to_e3
```

- ~~rot_score 是 so(3) 向量，需 adjoint~~。**已澄清**：SO(3) **真旋转**（det=+1）下，so(3) 轴向量与极向量变换**完全相同**，都按普通向量转；反射才区分两者，而这里不涉及反射。故 `rot_score @ r_poi_to_e3` 与 `tr_score @ r_poi_to_e3` 一样正确。详见 `doc/参考系与数值梯度专项分析.md` §1.3。

### BUG-7 ~~`train.py:118` `r_e3_to_poi` 约定~~（**误报，已核对正确**）

- ~~与 sampler 的 R_cum.transpose 约定是否一致需验证~~。**已澄清**：`se3_rot_noise`（= `add_noise_se3` 加到 POI 的旋转矩阵）的转置，与 `sampler.py` 的 `R_cum.transpose(-1,-2)` 是**同一约定**（都是「POI 实际朝向的逆」），一致。详见 `doc/参考系与数值梯度专项分析.md` §1.3。

### BUG-8 `tribrain.py:141` MultiScalePooling 注释与代码不符

- 注释说「concatenate 3 scales」「short=43/mid=43/global=42≈128」，但代码是**求和** `short + mid + (global_mean+global_max)/2`，三个都是 128 维。要么改注释，要么确认求和就是设计意图。

### BUG-9 ~~`evaluate.py:456` DockQ 的 d 值需核对标准~~（✅ **误报**，`evaluate.py:457-459` = 标准 DockQ：iRMS d=1.5、Lrms d=8.5）

- `dockq = (fnat + 1/(1+(irms/1.5)²) + 1/(1+(lrms/8.5)²)) / 3`。标准 DockQ 的缩放（iRMS 用哪个 d、Lrms 用哪个 d）需对照 Basu & Wallner 2016 原文确认是否写反。

### BUG-10 `train.py:252` `_physical_loss` 用固定 3.4 Å，非真实 VDW

- `F.relu(3.4 - dist)²` 对所有原子对一视同仁，没有按元素 VDW 半径。CLAUDE.md 已标注为已知局限，但仍是「物理损失」名不副实的一处。

### BUG-11 `film.py` 的 `FiLMSharedMLP` / `FiLMLayerHead` 是死代码

- 实际用的 FiLM 是 `tribrain.py:FiLMHead`；`film.py` 这两个类没被任何地方 import 调用，只剩 `apply_film` 在用。

### BUG-12 `evaluate.py:912` torsional strain 代理过于粗糙

- `1 - sin²(angle)` 把所有接近 0°/180° 的二面角都当 eclipsed 罚，不区分 staggered 构型、不涉及真实力场。

### BUG-14 `reflect.py:232` `detect_linker_strain` 把 anti(180°) 当应变

```python
return (abs_deg < strain_min) | (abs_deg > strain_max)   # strain_min=30, strain_max=150
```

- `|φ| > 150°` 就是 ~180°（anti），而 anti 恰恰是**最不 strained** 的 staggered 构型。把它和 `|φ| < 30°`（eclipsed）一起标为应变，物理上可疑。真正该罚的是近 0°（eclipsed），180° 不该罚（或用正规 torsional 势能）。

### BUG-15 `expert.py:17` `from torch_scatter import scatter_mean` 是未使用导入

- `expert.py` 里 `scatter_mean` 没有被任何地方用到（`compute_scores` 调的是 model 内部方法）。
- 顺带一个隐患：如果 `expert.py` 被**绕过 `__init__.py`/`expert_factory` 直接 import**，`torch_scatter` 是坏 stub 时这条导入会失败。依赖 `diffdock_compat` 先打补丁。

### BUG-16 `diffdock_compat.py:121` `_scatter_pytorch` 的 mean 对 1-D src 会广播错

```python
count = torch.zeros(dim_size, ...); count.index_add_(0, index, ones)
return out / count.clamp(min=1).unsqueeze(-1)
```

- `out` 是 `[dim_size, *src.shape[1:]]`，`count` 是 `[dim_size]`。当 `src` 是 1-D `[N]` 时，`out` 是 `[dim_size]`，`count.unsqueeze(-1)` 是 `[dim_size,1]`，广播会得到 `[dim_size,dim_size]`（错）。当前 `scatter_mean` 都作用于 2-D 特征所以没触发，属于边界雷。

### BUG-18 🔴🔴 **确证**：featurization 把每个分子**独立居中**，三元相对几何被彻底抹掉

已读到源码（`DiffDock/utils/inference_utils.py:231-237`），确凿：

```python
protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
complex_graph['receptor'].pos -= protein_center        # 受体（E3 或 POI）→ 自己的质心 = 0
ligand_center  = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
complex_graph['ligand'].pos  -= ligand_center           # 配体 → 自己的质心 = 0
```

- `featurize_complex` 对 `e3l`、`poil` 各自调 `InferenceDataset`，于是：
  - `e3_data`：E3 居中（e3_center = 0）
  - `poi_data`：POI 居中（poi_center = 0）
  - `ligand_data`：PROTAC 居中（ligand_center = 0）
- **三个分子被各自独立平移到原点，E3↔POI、E3↔PROTAC、POI↔PROTAC 的相对平移全部丢失**（只保留了各自朝向，因为只平移、不旋转）。
- 于是 `t_e3_to_poi = poi_center - e3_center = 0`，`transform_ligand_to_poi_frame` 的平移项恒为 0。
- **训练后果（致命）**：`compute_loss` 里「干净」的 PROTAC/POI 坐标都是原点，score matching 的目标是"把配体/POI 去噪到**原点**"，而不是去噪到它们在三元复合物里相对 E3 的真实位置。模型学到的是"三分子堆在原点"这个物理上无意义的分布。

**根因**：DiffDock 的 `InferenceDataset` 是给**推理**用的——推理时配体平移是随机初始化的扩散变量，所以初始位置无所谓、可以独立居中。但 TriCoopDiff 的**训练**需要保留配体/POI 相对 E3 的真实位姿作为监督信号，用了推理数据集就把信号抹了。正确做法是用 DiffDock 的**训练** featurization（`datasets/pdbbind.py` 那套，保留配体在受体坐标系里的位置），而不是 `InferenceDataset`。

> 这一条比之前的「差一个 e3_center」严重得多：不是差一个偏移，而是**整个相对几何不存在了**。之前 §1.4 专项分析里"差 e3_center"的结论是在"坐标为晶体框架"的假设下推出的，实测源码后推翻——实际是各自居中、相对平移归零。

---

## 附：确认没问题的部分（避免重复排查）

- `train.py` Tweedie 去噪（`x̂ = x_t + σ²·score`）与 score target `-tr_noise/σ²`（其中 `tr_noise = σ·ε`，故 target = `-ε/σ`）自洽 ✅
- `diffusion.sigma_schedule`、`so3_exp`（Rodrigues）、`so3_score_norm`/`so3_score_vec`（表查找）与 vendored DiffDock 对齐 ✅
- `kabsch_align` 的旋转矩阵 `R = U @ D @ Vt`（修过 transpose bug）✅
- `FiLMHead` 零初始化（adaLN-Zero）✅
- `sampler.sample_single` 的 `R_cum = R_cum @ rot_matrix` 累积顺序 ✅
