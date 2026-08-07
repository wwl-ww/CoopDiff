# DiffDock 审计文档（完整版）

> 源码 + 论文 (Corso et al., ICLR 2023, arXiv:2210.01776) 综合分析 | 2026-08-07

---

## 一、论文核心

### 1.1 问题建模

分子对接 → 生成建模。配体位姿映射到**产品流形**：

\[
\mathcal{P} = \mathbb{T}(3) \times SO(3) \times SO(2)^m
\]

- \(\mathbb{T}(3)\)：平移（3维欧几里得空间）
- \(SO(3)\)：整体旋转（3维旋转群）
- \(SO(2)^m\)：m 个可旋转键的扭转角（每个是 2D 旋转群）

在这个产品流形上定义前向扩散 + 逆向去噪，而不是直接在 3D 原子坐标上做。

### 1.2 两阶段推理

1. **Score model**（20M）：逆扩散生成候选位姿
2. **Confidence model**（5M）：二分类器（RMSD<2Å），对候选排序 → top-1

PDBBind 结果：top-1 成功率 38%（RMSD<2Å），远超传统方法（23%）和之前 DL 方法（20%）。

---

## 二、Score Model 架构（已验证源码）

### 2.1 整体结构

```
输入: HeteroData(receptor, ligand)
  │
  ├─ Embedding ────────────────────────
  │   rec_node_emb (AtomEncoder + ESM-2)
  │   rec_edge_emb (GaussianSmearing)
  │   lig_node_emb (AtomEncoder + sigma_emb)
  │   lig_edge_emb (bond features + sigma_emb + distance)
  │
  ├─ 交叉图 ───────────────────────────
  │   build_cross_conv_graph: receptor↔ligand
  │   动态截断：10 + 3·σ_tr (或 20 + 3·σ_tr)
  │
  ├─ 卷积层 × 6 ────────────────────────
  │   TensorProductConvLayer:
  │     Σ h_b ⊗ ψ_{ab} · Y(r̂_{ab})
  │     其中 ψ_{ab} = MLP(e_{ab}, h_a⁰, h_b⁰)
  │     BatchNorm + residual
  │
  ├─ Score 头 ─────────────────────────
  │   center_conv → tr_pred [B,3], rot_pred [B,3]
  │   bond_conv  → tor_pred [B,K]
  │
  └─ 输出: (tr_score, rot_score, tor_score, sidechain)
```

### 2.2 TensorProductConvLayer 详解

```
edge_attr → FC(hidden, weight_numel) → tp_weights    (ψ_{ab} MLP)
node_attr[dst] ⊗ Y(r̂_{ab}) ⊗ tp_weights → messages   (TensorProduct)
scatter(messages, src) → updated_node_attr              (消息聚合)
BatchNorm(out_irreps) → out
residual: out = out + pad(node_attr)
```

- 球谐：输⼊ l≤2，输出 l≤1（"odd_parity" 模式）
- irreps 序列：`0e → 0e+1o → 0e+1o+1e → 0e+1o+1e+0o`
- **FiLM 注入点**：residual 之后、下⼀层之前

### 2.3 DiffDock-L 参数（论文 + 训练脚本确认）

| 参数 | 值 | 来源 |
|------|-----|------|
| `ns` | **48** | 训练脚本 |
| `nv` | **10** | 训练脚本 |
| `num_conv_layers` | **6** | 训练脚本 |
| `sigma_embed_dim` | **64** | 训练脚本 |
| `distance_embed_dim` | **64** | 训练脚本 |
| `cross_distance_embed_dim` | **64** | 训练脚本 |
| `dropout` | **0.1** | 训练脚本 |
| `receptor_radius` | **15** | 训练脚本 |
| `c_alpha_max_neighbors` | **24** | 训练脚本 |
| `sh_lmax` | 2 | 源码默认 |
| `scale_by_sigma` | True | 源码 |
| `separate_noise_schedule` | False | 源码默认 |

### 2.4 节点特征

**配体**：16 类别特征（原子序数、手性、度、形式电荷、隐式价、H 数、自由基、杂化、芳香性、环成员）+ sigma 嵌入

**受体 Cα**：氨基酸类型（21 类）+ ESM-2 语言模型嵌入（1280 维，可选）

---

## 三、扩散过程

### 3.1 噪声调度

指数调度：\(\sigma(t) = \sigma_{\min}^{1-t} \cdot \sigma_{\max}^t\)

| DOF | σ_min | σ_max | 流形 |
|-----|-------|-------|------|
| Translation | 0.1 | **19** | R³ |
| Rotation | **0.03** | **1.55** | SO(3) |
| Torsion | **0.03** | **1.55** | Torus |

### 3.2 SO(3) 采样

不是简单高斯——是 SO(3) 上的各向同性分布：
- `sample_vec(eps)`：随机单位向量 × interp(cdf, eps)
- `score_vec(eps, vec)`：插值预计算的 score_norm
- 预计算缓存到 `.so3_*.npy`（⾸次运⾏后持久化）

### 3.3 逆扩散（推理）

20 步 Euler-Maruyama SDE：
\[
x \leftarrow x + g(t)^2 \cdot \nabla_x \log p_t(x) \cdot dt + g(t) \cdot \sqrt{dt} \cdot z
\]
最后⼀步不加噪声。可选**温度采样**（`temp_sampling`, `temp_psi`, `temp_sigma_data`）提⾼精度。

---

## 四、损失函数

### 4.1 Denoising Score Matching

三个 DOF 各⾃计算，权重均为 1.0：

**平移**（R³）：
\[
L_{tr} = \langle (s_\theta^{tr} - s^{tr})^2 \cdot \sigma_{tr}^2 \rangle
\]
其中 \(s^{tr} = -\Delta r / \sigma_{tr}^2\)（DSM ⽬标）

**旋转**（SO(3)）：
\[
L_{rot} = \langle ((s_\theta^{rot} - s^{rot}) / \text{score\_norm}(\sigma_{rot}))^2 \rangle
\]

**扭转**（Torus）：
\[
L_{tor} = \langle (s_\theta^{tor} - s^{tor})^2 / \text{score\_norm}^2(\sigma_{tor}) \rangle
\]

### 4.2 噪声采样

训练时 t ~ Beta(α, β)，⽤ `get_t_schedule('expbeta')` 采样噪声级别。

---

## 五、训练配置（DiffDock-L）

| 参数 | 值 |
|------|-----|
| Batch size | **16** |
| Learning rate | **1e-3** |
| LR scheduler | **Plateau** (patience=30) |
| Optimizer | Adam + **EMA** |
| Epochs | **850** |
| Warmup | Layer-wise (逐层解冻) |
| Validation | 每 5 epochs 跑实际推理（500 复合物）|
| Loss weights | tr=1, rot=1, tor=1 |

---

## 六、TriCoopDiff 集成验证清单

| # | 检查项 | 状态 | 备注 |
|---|--------|------|------|
| 1 | CGModel 参数匹配 (ns=48, nv=10, sigma_embed=64) | ✅ | `fix/diffdock-params` 已修 |
| 2 | 噪声调度匹配 (0.1→19, 0.03→1.55) | ✅ | `diffusion.py` 已修 |
| 3 | 训练超参匹配 (lr=1e-3, batch=16, epochs=850) | ✅ | `train_entry.py` 已修 |
| 4 | HeteroData 格式兼容 | ✅ | 源码验证 |
| 5 | set_time / complex_t / node_t 格式 | ✅ | 源码验证 |
| 6 | Conv 循环 + FiLM 注入 | ✅ | 源码验证 |
| 7 | Score head 逻辑复制 | ✅ | 源码验证 |
| 8 | timestep_emb (GaussianFourierProjection(64)) | ✅ | 已替换 placeholder |
| 9 | **损失函数权重匹配** | ⚠️ | 我们的 loss 未做 sigma 加权 |
| 10 | **SO(3) 采样精确性** | ⚠️ | 用近似 `so3_score_norm`，完整版需 .so3_*.npy |
| 11 | **ESM-2 嵌入** | ⚠️ | 当前未⽤，⽤原始氨基酸类型代替 |
| 12 | **温度采样** | ⚠️ | 推理未实现温度参数 |
