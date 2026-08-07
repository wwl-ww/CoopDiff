# DiffDock 审计文档

> TriCoopDiff Phase 0 审计 | 2026-08-07

---

## 模型架构

### CGModel (`models/cg_model.py`)

三层结构：**embedding → conv_layers → score heads**

| 阶段 | 内容 |
|------|------|
| Embedding | rec_node_embedding + rec_edge_embedding + ligand_embedding (lig_emb_layers) |
| 交叉图 | build_cross_conv_graph: receptor↔ligand radius edges, 动态截断 |
| Conv | conv_layers × N (TensorProductConvLayer)，拼接 ligand/lr/rec/rev_lr 四种边 |
| Score | center_conv → tr/rot scores, bond_conv → torsion scores |

### TensorProductConvLayer (`models/tensor_layers.py`)

每个 conv layer: `edge_attr → FC → tp_weights → node_attr[dst] ⊗ edge_sh ⊗ tp_weights → scatter → BatchNorm → residual`

**FiLM 注入点**：residual 之后、下一层之前。在 CGModel.forward() 的 conv 循环中（L339-349）。

### 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ns | 16 | 标量通道 |
| nv | 4 | 向量通道 |
| num_conv_layers | 2 | 卷积层数（DiffDock-L 可能用6） |
| sh_lmax | 2 | 球谐最大 l |
| scale_by_sigma | True | 缩放分数 |

---

## 扩散过程

- **噪声调度**: σ(t) = σ_min^(1-t) × σ_max^t（指数）
- **DOF**: tr σ∈[0.1,30], rot σ∈[0.1,1.6], tor σ∈[0.2,2.0]
- **SO(3)**: 预计算 score_norm 缓存到 `.so3_*.npy`
- **逆向**: 20步 Euler-Maruyama SDE，最后一步无噪声

---

## 图构建

`HeteroData` 含 `receptor`、`ligand` 节点，`(receptor,receptor)`、`(ligand,ligand)` 预计算边。交叉边每次前向动态构建。

---

## TriCoopDiff 集成点

| 特性 | 状态 |
|------|------|
| CGModel 包装为 FrozenDiffDock | ✅ |
| HeteroData 格式兼容 | ✅ |
| set_time 嵌入格式 | ✅ |
| conv 循环 + FiLM 注入 | ✅ |
| score head 逻辑复制 | ✅ |
| timestep_emb (GaussianFourierProjection) | ✅ |

**风险**：DiffDock-L 实际参数（ns, num_conv_layers）需从权重文件确认。
