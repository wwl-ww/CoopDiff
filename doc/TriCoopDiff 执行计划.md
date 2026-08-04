# TriCoopDiff 执行计划

> 合并版：技术细节（代码/函数/文件）+ 项目管理（验收标准/Go-No-Go/风险排查路径）

---

## 总览

| 阶段 | 时间 | 核心目标 | 验收标准 | Go-No-Go |
|------|------|---------|---------|-----------|
| **Phase 0** | 第 1 周 | 环境+数据+DeepTernary审计 | DiffDock跑通、审计报告出、数据量确认 | 数据 < 1000 → 调整预期 |
| **Phase 1** | 第 2-3 周 | MVP跑通 | 端到端生成三元结构，无硬物理错误 | 跑不通 → 回退排查 |
| **Phase 2** | 第 4-5 周 | 训练+结构验证 | FT版 DockQ ≥ 0.5，冲突率 < 2% | DockQ < 0.4 → 排查架构 |
| **Phase 3** | 第 6 周 | 系综质量验证 | 能量-概率 ρ > 0.3 | ρ < 0.2 → 跳过功能，写结构论文 |
| **Phase 4** | 第 7 周 | 功能排序验证 | α排序 ρ > 0.4，优于baseline | ρ < 0.3 → 降级为结构工具 |
| **Phase 5** | 第 8-9 周 | 消融+论文 | 6个核心消融完成，初稿完成 | — |

**总计：约 9 周。MVP 在第 3 周末出结果，关键决策点在第 4/5/6 周末。**

---

## Phase 0：准备阶段（第 1 周）

### 0.1 环境搭建与 DiffDock 验证（2 天）

```bash
conda create -n tricoopdiff python=3.10
pip install torch e3nn torchdiffeq rdkit biopython
git clone https://github.com/gcorso/DiffDock.git
# 下载 DiffDock-L 预训练权重
```

**需要读懂的关键文件**：

| 文件 | 内容 | 优先级 |
|------|------|--------|
| `datasets/pdbbind.py` | 图构建——双子图构图的参照 | ⭐⭐⭐ |
| `models/diffusion.py` | SDE/ODE 实现——我们扩散过程的基础 | ⭐⭐⭐ |
| `models/score_model.py` | EGNN score 网络——逐层注入的插入点 | ⭐⭐⭐ |
| `utils/sampling.py` | 推理逻辑——多 walker 采样的参照 | ⭐⭐ |

**验收**：DiffDock 推理 10 个蛋白-配体对，结果和官方报告一致。

### 0.2 DeepTernary 审计（3 天，与 0.1 并行）

```bash
git clone https://github.com/youqingxiaozhua/DeepTernary.git
# 下载预训练权重，在 PROTAC benchmark 上跑预测
```

**审计指标**：全局 DockQ、按 linker 长度分层 DockQ、空间冲突率、Linker 应变能 (MMFF94)、40 seed 收敛性和多样性。

**验收**：完整的 DeepTernary 审计报告，确认它的强项和弱项。

### 0.3 数据准备（3 天，与 0.1 并行）

**通用三元复合物**：
- PDB 查询脚本 → 筛选：两条蛋白链 + 一个配体（≥20 重原子）+ 界面面积 > 50 Å² + 分辨率 < 2.5Å
- 40% 序列相似度去冗余
- **⚠️ 先跑查询确认实际数量**。如果 < 1000 → 降低预训练预期或启用伪数据补充

**PROTAC benchmark**：
- ~50 个结构，统一格式（E3链、POI链、PROTAC分子）
- 按 E3 类型划分训练/测试（避免同 E3 的数据泄漏）

**产出**：`data/ternary_train.json`、`data/ternary_val.json`、`data/protac_benchmark.json`

### 0.4 伪三元数据生成（可选，2 天）

如果通用三元数据不足，写脚本从 PDB 蛋白-配体共晶池随机配对 + 标准 linker 连接 + 刚体调整 + 物理过滤。小批量试产 1000 个，肉眼检查。

**验收**：Phase 0 结束时，确认通用三元数据的实际规模，决定是否需要伪数据补充。

---

## Phase 1：MVP 实现（第 2-3 周）

> **目标**：能端到端生成三元结构。不求好，先求通。

### 1.1 双子图构图（2 天）

```python
# 新建: tricoopdiff/graphs.py

def build_subgraph_e3l(e3_coords, e3_features, ligand_coords, ligand_features):
    """E3-L 子图：E3残基 + PROTAC原子，❌ 无POI节点"""

def build_subgraph_pol(poi_coords, poi_features, ligand_coords, ligand_features):
    """POI-L 子图：对称"""

def build_tribrain_graph(e3, poi, ligand):
    """TriBrain 完整三分子图：所有节点 + 所有边（含E3-POI）"""
```

**验证**：两子图分别跑 DiffDock 原生推理 → 输出和标准二体对接一致（证明零 OOD）。

### 1.2 TriBrain 基础版（3 天）

```python
# 新建: tricoopdiff/tribrain.py

class MultiScalePooling(nn.Module):
    """近程<20Å(注意力池化,d=64) + 中程<50Å(Cα粗粒化+注意力池化,d=64) + 全局(平均+最大,d=64) → token序列[N,128]"""

class EquivariantTransformer(nn.Module):
    """3层，单头交叉注意力，32标量+4向量通道"""

class TriBrain(nn.Module):
    def __init__(self):
        self.pooling = MultiScalePooling()         # ~66k
        self.transformer = EquivariantTransformer() # ~1.1M
        self.se3_head = SE3ScoreHead()              # → s_SE3 [6]
        self.judge_head = JudgeHead()               # → 冲突掩码、置信度
        self.state_head = StateHead()               # → ΔΔG、生产性概率
        self.film_shared = FiLMSharedMLP()          # → h_cond [256]
        self.film_heads = nn.ModuleList([           # 12个独立head
            FiLMLayerHead() for _ in range(12)
        ])
```

**验收**：随机输入 → 所有输出维度正确，参数量 ~2.2M。

### 1.3 逐层 FiLM + DiffDock 封装（3 天）

```python
# 新建: tricoopdiff/expert.py, tricoopdiff/film.py

class FrozenDiffDock(nn.Module):
    """加载 DiffDock 权重，冻结，暴露每层 Conv 输出"""
    def forward(self, subgraph, t):
        # 返回: layer_outputs [h1..h6], scores {trans, rot, torsion}

def apply_film(layer_output, gamma, beta):
    """h' = h * (1+gamma) + beta, gamma/beta来自FiLM head, 零初始化"""
```

**注入流程**：
```
Conv1 → h1 → FiLM_1(γ1,β1) → h1'
...逐层...
Conv6 → h6 → FiLM_6(γ6,β6) → h6' → score heads
```

**验证**：零初始化时，加了 FiLM 的输出和原生 DiffDock 输出**数值精确相等**。

### 1.4 扩散推理 pipeline（3 天）

```python
# 新建: tricoopdiff/diffusion.py, tricoopdiff/model.py, tricoopdiff/merging.py

class TriCoopDiff(nn.Module):
    def forward(self, e3, poi, ligand, t):
        # 1. 构图
        sub1, sub2 = build_subgraphs(e3, poi, ligand)
        tri_graph = build_tribrain_graph(e3, poi, ligand)
        
        # 2. TriBrain → FiLM参数 + s_SE3
        tri_out = self.tribrain(tri_graph, t)
        
        # 3. 二体专家逐层前向 + FiLM注入
        sub1_layers = self.expert.get_layer_outputs(sub1, t)
        for l in range(6):
            sub1_layers[l] = apply_film(sub1_layers[l], film_params[l])
            sub2_layers[l] = apply_film(sub2_layers[l], film_params[l+6])
        
        # 4. Score输出 + 合并
        s_q1, s_q2 = self.expert.score_heads(sub1_layers[-1]), ...
        s_q = {k: s_q1[k] + s_q2[k] for k in s_q1}  # 直接加和
        
        return {'s_q': s_q, 's_se3': tri_out['s_se3']}, tri_out
```

**初始化**：E3 固定原点。PROTAC 在 E3 口袋附近球壳 ±10Å 随机。POI 在 E3 周围球壳 30-60Å 随机。多速率 λ_q(t=0.7归零)、λ_SE3(t=0.5归零)。

**验收**：给定三个分子 → 端到端推理 → 输出一个三元复合物结构。

### 1.5 TriReflect 后处理（1 天）

```python
# 新建: tricoopdiff/reflect.py

def detect_clashes(coords, vdw_radii):        # KD-tree, O(N log N)
def detect_bond_anomalies(ligand, bonds):      # 键长 > 标准×1.5
def detect_linker_strain(ligand, torsions):    # |二面角| < 30°或 > 150°
def local_minimization(coords, clash_atoms):   # MMFF94, 最多3轮
```

**验收**：输出结构无穿模/断键/高能二面角。

### Phase 1 总验收

| 检查项 | 标准 |
|--------|------|
| 端到端生成三元结构 | ✅ |
| 初始 FiLM 透明（输出和原生 DiffDock 一致） | ✅ |
| TriReflect 后结构无硬物理错误 | ✅ |
| 参数量 ~2.2M 可训 | ✅ |

**如果跑不通，按顺序排查**：
1. 双子图构图是否正确？（分别跑 DiffDock 验证零 OOD）
2. FiLM 嵌入位置是否正确？（逐层检查输出维度）
3. 扩散积分是否正确？（对比单步更新和理论预期）
4. 最坏情况：退化为"二体对接 + 加和"的 L0 baseline，先保证能出结构

---

## Phase 2：训练 + 结构验证（第 4-5 周）

### 2.1 训练 pipeline（3 天）

```python
# 新建: tricoopdiff/train.py

def train_step(model, batch, optimizer):
    t = torch.rand(batch_size)          # 随机采样扩散时间
    x_noisy = add_noise(x_clean, t)     # 从真实结构加噪
    score_pred, _ = model(x_noisy, t)
    
    loss_score = MSE(score_pred, noise_target)              # 权重 1.0
    loss_phys = physical_loss(x_denoised)                    # 权重 0.5
    loss_se3 = MSE(s_se3_pred, se3_target)                   # 权重 0.3
    
    # 物理损失梯度路径: loss_phys → x_denoised → s_tri_residual → TriBrain
    loss = loss_score + 0.5*loss_phys + 0.3*loss_se3
    loss.backward()  # 二体专家冻结，梯度只到 TriBrain+FiLM
```

> 物理损失对每步去噪后坐标计算，不走 BPTT。梯度通过 s_tri_residual 直接回传（不走 FiLM 间接路径）。

**验证**：小数据试训（100 样本），确认 loss 正常下降，无 NaN。

### 2.2 通用三元预训练（4 天）

| 参数 | 值 |
|------|-----|
| 数据集 | Phase 0 产出的通用三元数据 |
| Epochs | 50-100, early stopping (patience=15) |
| Batch size | 32 |
| 学习率 | 1e-4, AdamW, 余弦退火 |
| 训练范围 | TriBrain + FiLM头（~2.2M），DiffDock 冻结 |
| 多任务策略 | 阶段1-2 冻结裁判头和协调头，只训注入头+刚体头 |

**保存**：`checkpoints/tricoopdiff_zero.pt`

### 2.3 PROTAC 微调（2 天）

| 参数 | 值 |
|------|-----|
| 数据集 | PROTAC benchmark (~40 train / ~10 test) |
| Epochs | 30-50, early stopping |
| Batch size | 8 |
| 学习率 | 2e-5 |
| 正则化 | LoRA (rank=4) on TriBrain MLP 层, 或强 weight decay |

**保存**：`checkpoints/tricoopdiff_ft.pt`

### 2.4 结构预测验证（3 天）

```python
# 新建: tricoopdiff/evaluate.py

# 结构指标
compute_dockq(pred, ref)
compute_rmsd(pred, ref)           # 全局 + warhead + linker 分别报告
compute_clash_rate(ensemble)
compute_linker_strain(ensemble)   # MMFF94

# 按 linker 长度分层（短<8 / 中8-15 / 长>15）
```

**对比**：Zero 版 vs FT 版 vs DeepTernary 审计结果。

### Phase 2 验收标准

| 指标 | 及格线 | 优秀线 |
|------|--------|--------|
| FT 版全局 DockQ | ≥ 0.5 | ≥ 0.6 |
| 长 linker 子集 DockQ | ≥ DeepTernary | > DeepTernary +0.05 |
| 空间冲突率（TriReflect后） | < 2% | < 1% |
| Linker 应变能 | 低于 DeepTernary 20% | 低于 50% |

### 🔴 Go-No-Go 决策点（第 5 周末）

```
DockQ ≥ 0.5 → ✅ 继续 Phase 3
DockQ 0.4~0.5 → ⚠️ 排查后决定是否继续
DockQ < 0.4 → 🔴 排查架构，可能需升级 TriBrain 容量

排查路径:
  1. 数据质量？（通用三元是否真的有桥接模式？）
  2. TriBrain 容量？（3层32通道 → 升级到6层64通道）
  3. FiLM 调制？（消融 L0 vs L1，确认 FiLM 确实有增益）
  4. 训练是否充分？（loss 曲线、过拟合检查）
  5. 最坏情况：即使 DockQ 不行，检查物理合理性是否达标 → 叙事转向"物理更鲁棒"
```

---

## Phase 3：系综质量验证（第 6 周）

> **前提**：Phase 2 结构预测及格。**目的**：验证系综有物理意义——这是功能排序的前提。

### 3.1 多 Walker 采样（1 天）

32 个随机 seed → 32 条独立轨迹 → 32 个构象。基础统计：RMSD 分布、聚类。

### 3.2 系综质量评估（4 天）

| 指标 | 定义 | 及格线 | 优秀线 |
|------|------|--------|--------|
| 能量-概率 Spearman ρ | 32 构象 MMFF94 能量 vs 采样频率 | > 0.3 | > 0.5 |
| Top-1 能量排名 | 采样最多构象在 32 个中排第几 | 前 10 | 前 5 |
| 系综多样性 | RMSD 标准差 | > 1Å | > 2Å |
| 重采样一致性 | 两次独立采样统计量差异 | < 30% | < 20% |

**和 DeepTernary 32 seed 对比**：多样性、能量-概率相关性。

### 3.3 MD 对比（可选，2 天）

选 1-2 个有文献 MD 轨迹的 PROTAC 体系，聚类后比较主要构象态排名。

### 🔴 Go-No-Go 决策点（第 6 周末）

```
ρ > 0.3 → ✅ 系综有意义，进入 Phase 4（功能排序）
ρ 0.2~0.3 → ⚠️ 可以做但预期降低
ρ < 0.2 → 🔴 系综基本无物理意义
  → 跳过 Phase 4，论文重心放在结构预测+物理合理性
  → 功能排序叙事降级为"讨论中的未来方向"
```

---

## Phase 4：功能排序验证（第 7 周）

> **前提**：Phase 3 系综 ρ > 0.3。**公平 baseline = 传统对接打分（Vina/Glide/MM-PBSA），不是 DeepTernary。**

### 4.1 实验数据整理（2 天）

~200 个有协同性 α 值的 PROTAC，按实验方法、E3类型分层标注。按质量分层评估。

### 4.2 协同性排序（3 天）

从系综布居比 → ΔΔG 估计 → 排序。Spearman ρ vs 实验 α，和 baseline 对比。

| 指标 | 及格线 | 优秀线 |
|------|--------|--------|
| α 排序 Spearman ρ | > 0.4 | > 0.6 (p<0.05 vs baseline) |
| Top-20% 命中率 | > 40% | > 60% |

### 4.3 辅助功能指标（2 天）

| 指标 | 及格线 | 数据需求 |
|------|--------|---------|
| 生产性构象判别 AUROC | > 0.7 | ~20-30 对 |
| Linker 长度-活性趋势 | 5系列对3个 | ~3-5 系列 |
| Hook 效应拐点排序 ρ | > 0.4 | ~20-30 个 |

### 🔴 Go-No-Go 决策点（第 7 周末）

```
ρ > 0.4 且优于 baseline → 🟢 大成功，护城河建立
ρ > 0.3 → 🟡 部分成功，功能排序"有趋势但不够精确"
ρ < 0.3 → 🔴 功能排序不行
  → 模型降级为"结构预测+物理更合理"
  → 叙事调整为"系综提供了结构多样性但热力学精度待提高"
```

---

## Phase 5：消融 + 论文（第 8-9 周）

### 5.1 核心消融（7 天）

在 MVP 基础上，6 个必须做：

| # | 消融 | 预计耗时 | 测量指标 |
|---|------|---------|---------|
| A | L0（无FiLM）vs L1（逐层FiLM） | 1天 | DockQ、冲突率 |
| B | 加和 vs Score协调层 | 1天 | DockQ、linker RMSD |
| C | 单速率 vs 多速率 | 0.5天 | DockQ、收敛速度 |
| D | 口袋引导 有 vs 无 | 0.5天 | 长linker DockQ、收敛率 |
| E | TriReflect 有 vs 无 | 0.5天 | 冲突率、键长异常率 |
| F | 原始特征 vs +二体专家特征 | 1天 | DockQ |

> 补充消融（G/I/H）只在核心消融全部完成且效果达标后做。

### 5.2 论文初稿（7 天）

**结构**：Introduction（问题+贡献）→ Method（架构+扩散+采样）→ Experiments（结构 vs DeepTernary + 物理合理性 + 系综质量 + 功能排序 + 消融）→ Discussion（局限+未来）

关键图表：架构图（已有 mermaid）、DockQ 分层对比柱状图、能量-概率散点图、α 排序 vs baseline 对比表。

### 5.3 最终成功矩阵

| | 第一层过（结构好） | 第一层不过 |
|---|---------|-----------|
| **第二层过（功能好）** | 🟢 **大成功** | 🟡 偏科 |
| **第二层不过但第三层过** | 🟡 基本成功 | 🔴 失败 |
| **第三层也不过** | 🟠 部分成功 | 🔴 彻底失败 |

---

## 并行任务

以下任务可以和主线并行：

| 任务 | 时机 | 说明 |
|------|------|------|
| DeepTernary 审计 | Phase 0 | 和环境搭建并行 |
| 伪三元数据生成 | Phase 0-1 | 如果真实数据不足 |
| 实验 α 数据整理 | Phase 2 | 训练时整理，不占关键路径 |
| 论文图表 | Phase 3-4 | 边做实验边画图 |
| MD 对比 | Phase 3 | 1-2个体系即可，不要求全量 |

---

## 完整 Go-No-Go 决策链

```
第 2 周末: MVP 跑通？ → 通→继续 | 不通→排查架构
第 5 周末: DockQ ≥ 0.5？ → 是→继续 | 否(<0.4)→排查架构/升级容量
第 6 周末: 系综 ρ > 0.3？ → 是→做功能排序 | 否(<0.2)→跳过，写结构论文
第 7 周末: α排序 ρ > 0.4？ → 是→大成功 | 否(<0.3)→降级为结构工具
第 9 周末: 论文初稿完成
```

**核心原则：先求跑通，再求跑好；先验证地基，再盖楼。每一步都有退路，不在错误方向上走太远。**
