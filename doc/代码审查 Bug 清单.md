# 代码审查 Bug 清单

> 审查日期：2026-08-16 | 范围：`tricoopdiff/` 核心模块（model / train / tribrain / diffusion / sampler / evaluate / graphs / merging / film）
>
> 目的：只记录问题，不改代码。按严重度分三级：🔴 严重 / 🟡 中等 / 🟢 轻微+待验证。

---

## 🔴 严重

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

### BUG-2 `model.py:224` 批量堆叠用 `.squeeze(0)`，对 B>1 是错的

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

### BUG-3 多速率 λ 在 forward 有 mask、reverse 没有，前后不一致

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

### BUG-5 `tribrain.py:335` 向量更新的 `LayerNorm(nv*3)` 破坏等变性

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

### BUG-17 `train.py:84` `compute_loss` 无条件 `self.model.train()`，把 validate 的 eval 模式顶掉了

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

### BUG-6 `model.py:201-203` frame fix 的 rot_score 旋转需验证

```python
r_poi_to_e3 = r_e3.transpose(-1, -2)
scores_2['tr_score'] = scores_2['tr_score'] @ r_poi_to_e3
scores_2['rot_score'] = scores_2['rot_score'] @ r_poi_to_e3
```

- `rot_score` 是 so(3)（李代数）向量，换坐标系应当用**伴随作用**（adjoint），且行/列向量约定（`v @ R` vs `R @ v`）需要跟 DiffDock 的 rot_score 语义核对。`tr_score`（平移向量）直接 `@ R` 大概率对，`rot_score` 存疑。

### BUG-7 `train.py:118` `r_e3_to_poi = se3_rot_noise.transpose(-1,-2)` 约定需验证

- 把 SE3 旋转噪声的转置当成「E3→POI 帧旋转」传给 model，与 `sampler.py:120` 的 `R_cum.transpose(-1,-2)` 是两条不同的路径，两者约定是否一致需要跑通验证。

### BUG-8 `tribrain.py:141` MultiScalePooling 注释与代码不符

- 注释说「concatenate 3 scales」「short=43/mid=43/global=42≈128」，但代码是**求和** `short + mid + (global_mean+global_max)/2`，三个都是 128 维。要么改注释，要么确认求和就是设计意图。

### BUG-9 `evaluate.py:456` DockQ 的 d 值需核对标准

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

### BUG-16 `diffdock_compat.py:116` `_scatter_pytorch` 的 mean 对 1-D src 会广播错

```python
count = torch.zeros(dim_size, ...); count.index_add_(0, index, ones)
return out / count.clamp(min=1).unsqueeze(-1)
```

- `out` 是 `[dim_size, *src.shape[1:]]`，`count` 是 `[dim_size]`。当 `src` 是 1-D `[N]` 时，`out` 是 `[dim_size]`，`count.unsqueeze(-1)` 是 `[dim_size,1]`，广播会得到 `[dim_size,dim_size]`（错）。当前 `scatter_mean` 都作用于 2-D 特征所以没触发，属于边界雷。

### BUG-18 `featurize_common.py` E3/POI 分开 featurize，相对几何是否一致待验证

- `featurize_complex` 对 `e3l` 和 `poil` 分别调 `featurize_subgraph`（各自走 DiffDock `InferenceDataset`，各自做 recentering）。若两个子图的 recentering 中心不同（比如 e3l 以 E3 COM 为中心、poil 以 POI COM 为中心），则 E3 和 POI 落在**两个不同的坐标系**里，`ligand_data` 取的是 e3l 的（E3 系），而 `poi_data` 是 poil 系的——E3-POI 相对位姿就错了。
- 需要确认 `../DiffDock` 的 `process_mols`/`InferenceDataset` 是否对两个子图用**同一个原点**（比如都以 ligand COM 为中心），否则三元几何从一开始就带一个系统误差。这跟 `model.py` 的 frame fix / `AgentB 交接文档` 是同一类「参考系」问题。

---

## 附：确认没问题的部分（避免重复排查）

- `train.py` Tweedie 去噪（`x̂ = x_t + σ²·score`）与 score target `-tr_noise/σ²`（其中 `tr_noise = σ·ε`，故 target = `-ε/σ`）自洽 ✅
- `diffusion.sigma_schedule`、`so3_exp`（Rodrigues）、`so3_score_norm`/`so3_score_vec`（表查找）与 vendored DiffDock 对齐 ✅
- `kabsch_align` 的旋转矩阵 `R = U @ D @ Vt`（修过 transpose bug）✅
- `FiLMHead` 零初始化（adaLN-Zero）✅
- `sampler.sample_single` 的 `R_cum = R_cum @ rot_matrix` 累积顺序 ✅
