# Score 合并双重计数 + 零-FiLM 基线不透明 交接文档

> 由 Agent（本会话）写给下一个接手「score merging」任务的 Agent。本文只覆盖**问题 #1：
> `s_q = s_q1 + s_q2` 的直接相加破坏了零-FiLM 基线透明性**。问题 #2/#3/#4（eval 侧）已在
> commit `773c3bd` 修完，不在此范围。**正在跑的 Phase1 训练别动**（见第 7 节契约）。

## 0. 当前状态（接手前先读）

- 分支 `phase1-train`，最新 commit `773c3bd fix(evaluate): drop meaningless poi_rmsd, fix
  clash_rate bond mask, dockq-best selection`（+177/−110，只改 `tricoopdiff/evaluate.py`、
  `tricoopdiff/reflect.py`）。
- Phase1 训练**正在用当前合并方式跑**（B=1，100 epoch）。所以本任务的修复**不能改
  `merging.py` 的默认行为让训练中途崩掉**——要么默认参数向后兼容，要么先跟用户确认要不要
  停训再改。
- Agent B 的 identity-rotation 修复**已经落地**：`model.py:forward` 已有 `r_e3_to_poi` 可选
  入参（默认 `torch.eye(3)`），`sampler.py:sample_single` 已追踪累积旋转 `R_cum` 并传
  `r_e3_to_poi=R_cum.transpose(-1,-2)`。本任务不碰这条链路。

## 1. 问题一句话

**零-FiLM 基线（不传 `--tribrain-ckpt`）应该复现纯 DiffDock-L 二体（E3-L）的 PROTAC 分布
（这是 CLAUDE.md 里明确写的 Go/No-Go 参考）**，但实测它不：E3-PROTAC 界面 DockQ ≈ **0.153**，
而纯 DiffDock E3-L 同 case 是 **0.844**。根因是 `merging.py:merge_direct_addition` 把两个子图
的完整 score 直接相加。

## 2. 根因分析（两个独立问题，别混为一谈）

`tricoopdiff/merging.py:39`：

```python
merged[key] = scores_1[key] + scores_2[key]   # tr / rot / tor 三类都直接加
```

其中 `scores_1` = E3-L 子图输出，`scores_2` = POI-L 子图输出。两个子图各自由**同一个**冻结
DiffDock expert 跑一遍，各自输出完整的 PROTAC score（`tr`[3] / `rot`[3] / `tor`[K]）。

### 问题 A：帧不一致（tr / rot 是几何错误）

- PROTAC 坐标在采样循环里**只在 E3 帧**存一份（`sampler.py` 的 `ligand_pos`，E3 帧），
  `reverse_step_protac` 用 `s_q['tr']/s_q['rot']` 更新它。
- 子图 1（E3-L）：ligand 直接用 E3 帧坐标 → `tr_score`/`rot_score` 在 **E3 帧**。
- 子图 2（POI-L）：`model.py:130` 先 `transform_ligand_to_poi_frame(...)` 把 ligand 旋到
  **POI 帧**再喂 expert → `tr_score`/`rot_score` 在 **POI 帧**。
- 两个不同帧的 3 维平移 score / so(3) 旋转 score 直接相加是**维度错误**：sub2 的平移/旋转
  贡献必须先用 `R_cum`（累积 SE3 旋转）旋回 E3 帧再加，否则方向错。注意 `so(3)` 旋转 score
  是轴角向量，旋回 E3 帧同样要乘旋转矩阵。
- `tor`（扭转）是内部自由度，帧无关，相加**几何上合法**——但仍放大一倍的量级（问题 B）。

### 问题 B：双重计数破坏了零-FiLM 透明性（量级/语义错误）

- 零-FiLM 时 FiLM 参数全 0 → `scores_1` = 原生 E3-L expert 输出、`scores_2` = 原生 POI-L expert
  输出。`s_q = s_q1 + s_q2` 是两个**独立二体力场之和**，≠ 任何单一二体 expert。这就是
  `0.153 ≠ 0.844` 的来源。
- 即使把问题 A 的帧修对、把相加改成平均，零-FiLM 基线仍是「E3-L 与 POI-L 两个预测的平均」，
  **依然不等于纯 DiffDock E3-L**。透明性基线要求的是「不引入三体协调时 = 纯 E3-L」，所以
  光靠归一化救不回来。

### 语义矛盾（这是设计决策，不是纯 bug）

架构的本意（CLAUDE.md 数据流）：**协调信号走 TriBrain→FiLM**（adaLN-Zero，零初始化 → 透明），
s_q1/s_q2 只是两个二体 expert 的中间量。但 `merge_direct_addition` 又引入了一条**独立的、非
零初始化**的协调路径（把 POI-L 的完整 score 加进来）。这两条路径冗余，且后者破坏了
"零初始化 → 透明 → 基线可复现纯二体" 的核心卖点。

## 3. 候选方案（下一步 Agent 先复现测量，再和用户定选哪个）

先跑一遍复现（第 4 节命令），确认 0.153 vs 0.844 稳定可复现，再动代码。三档：

### 方案 A：只修帧（几何正确性，最小改动）

把 sub2 的 `tr`/`rot` score 用 `r_e3_to_poi` 旋回 E3 帧再合并（`tor` 不变）。解决 A 不动 B。
**副作用**：零-FiLM 基线仍 ≠ 纯 E3-L（只是从「方向也错」变成「方向对但量级/语义错」）。
适合作为第一步的「中间修复」，但**不足以让 Go/No-Go 基线通过**。

### 方案 B（推荐）：让零-FiLM 时 `s_q = s_q1`，协调只走 FiLM

- 合并改成加权 `s_q = α·s_q1 + (1−α)·s_q2`，**α 由 TriBrain 预测，且初始化成 α=1**（等价于把
  `merging.py:LearnedCoordinationLayer` 的 `weight_net` 改成零偏置 + 最后加 `logit` 初始化为 1，
  或 `Sigmoid` 前加常数），这样零-FiLM 时 α=1 → `s_q = s_q1` = 纯 E3-L，透明性基线天然成立。
- 这同时把「是否引入 POI 二体力场」也变成一个**可学习、从纯 E3-L 起步**的量，和 FiLM 的
  adaLN-Zero 哲学一致。
- **要处理的点**：α 从 1 起步意味着早期梯度主要经 FiLM（E3-L 路径），POI-L 的 FiLM head 梯度
  会很小（`(1−α)` 小）。需要确认 POI-L 分支仍有梯度、不塌死——必要时给 `(1−α)` 一个下限或
  单独的 POI-L score 辅助损失。
- 帧问题（A）在此方案里依然要修：`(1−α)·s_q2` 里的 `s_q2` 必须旋回 E3 帧再加，否则一旦 α<1
  又错。两者**都要做**。

### 方案 C：先不改，等 Phase1 跑完

默认方案（本会话已定）：Phase1 当前正在用 `s_q1+s_q2` 训练。若马上改合并，已训的 checkpoint
和训练动态会不连续。让 Phase1 先跑完/或到下一个自然暂停点，再决定 A/B。**若选 C，本任务只做
「复现测量 + 写清诊断」，不提交代码改动。**

> 我的建议：**先做第 4 节的复现 + 精确诊断，提交一份「测量记录」，然后等 Phase1 完再上方案 B。**
> 方案 A 可以顺带做一个「帧修正」的独立小 PR，因为它不管方案 B/C 都必须修（sub2 的 tr/rot 帧
> 错了，这是硬伤）。

## 4. 复现与验收命令

```bash
# 零-FiLM 基线（当前 s_q1+s_q2）——应得 ~0.153
python scripts/evaluate_entry.py --case 5T35_D_A_759 --n-walkers 32 \
    --out results/eval_zero_film_mergebug.json

# 纯 DiffDock E3-L 二体基线（对照，~0.844）——用 ../DiffDock 自己的 inference 跑同一 case
```

验收标准（方案 B 落地的 Go/No-Go）：

1. **零-FiLM 基线（不传 ckpt）的 E3-PROTAC DockQ 与纯 DiffDock E3-L 二体基线一致**（差异在采样
   随机误差内，同 seed 下 <0.02）。
2. 数值 sanity：构造一个已知朝向的 POI，验证 sub2 的 `tr`/`rot` score 旋回 E3 帧后，`s_q` 的
   `tr`/`rot` 与「手动把 PROTAC 用同一 `R_cum` 旋回」的几何一致（对照 native 三元复合物）。
3. 训练路径不崩：`scripts/mvp_smoke_test.py` 两个 gate 仍过（GATE1 零-FiLM 透明、GATE2 梯度
   到 TriBrain）；`train_entry.py` 跑 3-epoch smoke 无 NaN。

## 5. 交接契约（不要破坏）

- `model.forward` 返回 dict 结构不变：`{'s_q': {'tr','rot','tor'}, 's_se3': [B,6],
  's_se3_components': {...}, 'confidence': [B,1], 'state': [B,2]}`。
- **只改 `merging.py`（合并逻辑）、可能 `model.py`（把 `r_e3_to_poi` 传进 merge 做帧修正）、
  `sampler.py`（若合并需要累积旋转，但已有 `R_cum` 可直接复用）**。
- **不要碰**：`diffusion.py` 的采样数学、`train.py`、`scripts/evaluate_entry.py` 的 CLI。
- 新参数/新行为必须**默认退化为当前行为**（或经用户确认停训后再切），避免打断正在跑的 Phase1。

## 6. 相关占位（上下文，非本次必做）

- `loss_se3 ≈ 0`（见 `doc/训练入口 SLURM 与 Phase1 smoke 交接文档.md` §2）：SE3 head 零初始化
  输出≈0，若 loss 仍≈0 说明 σ 加权或 target 量级不对——与 score 合并无关，另立任务。
- `loss_phys ≈ 0`：`train.py:_physical_loss` 用固定 3.4Å 硬截断，良分离复合物无 clash，物理项
  无梯度——换成真实 VDW 半径才有效。
- B=1 单复合物模式、checkpoint 只存/读 `tribrain_state_dict` 等，仍在 CLAUDE.md 的 open items。

## 7. 一句话给下一个 Agent

先复现 0.153 vs 0.844 确认问题 A/B 都存在 → 写测量记录 → 帧修正是硬伤先修（方案 A 的小 PR）
→ 零-FiLM 透明性靠方案 B（α 初始化 =1）落地，等 Phase1 自然暂停后再切。
