# TriBrain → Coordinator 改名方案

> 提案日期：2026-08-14 | 状态：待确认后执行
>
> 范围：把唯一可训练模块 `TriBrain` 改名为 `Coordinator`。纯命名重构，不改任何数值/逻辑。

---

## 一、背景与动机

`TriBrain`（`tricoopdiff/tribrain.py`，~2.2M 参数）是 TriCoopDiff 里**唯一可训练**的模块：池化三元图 → 等变 Transformer → FiLM 调制参数 + SE(3) 刚体 score + 置信度 + 状态。它在整个架构里扮演"三元调控中枢"的角色——协调两个物理断开的二体专家（E3-PROTAC / POI-PROTAC）。

问题：
- "Brain" 这个名字**俗**，且不准确——它不做"大脑"式的决策，做的是"协调/调制"。
- 与项目名 **TriCoopDiff** 的"Coop（协同）"主题脱节——这个模块恰恰是"协同"的实体。
- 与算法原理脱节：数学上（见 `公式推导参考.md` 主定理）它学的是 E3-POI **耦合力** $g=-\beta\nabla U_{EP}$，并把它通过 **FiLM 调制**注入冻结专家。

## 二、为什么是 Coordinator

| 维度 | 贴合点 |
|------|--------|
| 项目 | 呼应 Tri**Coop**Diff 的"协同"；模块就是"协同"在代码里的载体 |
| 算法原理 | 它**协调**两个冻结二体专家，注入 FiLM + SE(3) 耦合 score（对应推导里的 $g$） |
| 学术环境 | "coordination network / coordinator" 是标准说法（区别于 ControlNet 的 control、adapter 的 adapt） |
| 代码 | `self.coordinator`、`coordinator.py`、`from .coordinator import Coordinator` 清晰自然 |

## 三、改名范围（精确清单）

### 3.1 文件与类（核心）

| 现在 | 改为 |
|------|------|
| `tricoopdiff/tribrain.py` | `tricoopdiff/coordinator.py` |
| `class TriBrain(nn.Module)`（tribrain.py:372） | `class Coordinator(nn.Module)` |
| `self.tribrain`（model.py:57 等） | `self.coordinator` |

### 3.2 构图函数

| 现在 | 改为 | 说明 |
|------|------|------|
| `build_tribrain_graph`（graphs.py:122） | `build_ternary_graph` | 它建的是"完整三元图"，"ternary" 比 "tribrain" 更准；也顺带消除了 "brain" 字样 |
| 变量 `tri_graph`（model.py:150） | `ternary_graph`（可选） | `tri` 本义是 ternary，不强制改；为一致可顺带改 |

### 3.3 Checkpoint 序列化键

| 现在 | 改为 | 决策 |
|------|------|------|
| `tribrain_state_dict`（inference.py:301-302；train.py save/load） | `coordinator_state_dict` | **建议改**——目前还没有训练出的 checkpoint，无兼容性负担，早改早干净 |

> ⚠️ 若未来已有 checkpoint，改键会导致旧权重读不出来。趁现在没训练，一次性改掉。

### 3.4 引用点清单（需同步）

**代码（Python）：**

| 文件 | 引用内容 |
|------|---------|
| `tricoopdiff/__init__.py` | `from .graphs import ... build_tribrain_graph` → `build_ternary_graph` |
| `tricoopdiff/model.py` | `from .tribrain import TriBrain`、`self.tribrain`、`build_tribrain_graph`、`tri_graph`、`zero_film_verification(..., tribrain, tri_graph)` 及 docstring |
| `tricoopdiff/train.py` | `model.tribrain.parameters()`、docstring/注释里的 "TriBrain" |
| `tricoopdiff/film.py` | 仅 docstring/注释 "TriBrain coordination / pooled condition from TriBrain" |
| `tricoopdiff/merging.py` | 仅 docstring "TriBrain conditioning vector" |
| `tricoopdiff/graphs.py` | `build_tribrain_graph` 定义 + docstring |
| `scripts/inference.py` | `from tricoopdiff.tribrain import TriBrain`、`tribrain = TriBrain(...)`、`tribrain_state_dict`、注释/help |
| `scripts/train_entry.py` | `from tricoopdiff.tribrain import TriBrain`、`tribrain = TriBrain(...)`、arg help "TriBrain scalar/vector channels" |
| `tests/test_model_build.py` | `from tricoopdiff.tribrain import TriBrain, MultiScalePooling, ...`、`tribrain = ...`、`test_tribrain_build()`、`build_tribrain_graph` |

**文档（Markdown）：**

| 文件 | 处理 |
|------|------|
| `CLAUDE.md` | 模块表 + 数据流图里的 "TriBrain" → "Coordinator" |
| `doc/TriCoopDiff 模型交接文档.md` | 大量 "TriBrain" 引用，需全局替换 |
| `doc/TriCoopDiff 执行计划.md` | 同上 |
| `doc/TriCoopDiff vs DeepTernary 竞争定位与实验计划.md` | 同上 |
| `doc/公式推导参考.md` | "TriBrain" → "Coordinator"（公式里的 $\mathrm{TriBrain}(\cdot;\theta)$ → $\mathrm{Coordinator}(\cdot;\theta)$） |
| `doc/三分子初始化策略方案.md` | 同上 |
| `doc/README.md` | 索引里的引用 |

### 3.5 不改的东西（边界）

| 保留 | 原因 |
|------|------|
| `TriCoopDiff`（项目/顶层模型名） | 项目名，不动 |
| `MultiScalePooling` / `EquivariantTransformer` / `SE3ScoreHead` / `JudgeHead` / `StateHead` / `FiLMHead` | 都是功能描述名，与 "brain" 无关 |
| `build_subgraph_e3l` / `build_subgraph_pol` | 二体子图，语义不变 |
| `s_tri_residual`（train.py） | 这是 "three-body residual"（三体残差），不是 "tribrain"，`tri` = ternary，不改 |
| `TriReflect`（后处理） | 独立模块，"tri" = ternary |
| `FiLM` / `MultiRateDiffusion` / `TernaryInitializer` | 标准术语/功能名 |

## 四、执行步骤

1. `git mv tricoopdiff/tribrain.py tricoopdiff/coordinator.py`
2. 改类名 `TriBrain` → `Coordinator`、`self.tribrain` → `self.coordinator`
3. 改 `build_tribrain_graph` → `build_ternary_graph`（定义 + 所有调用点）
4. 改 checkpoint 键 `tribrain_state_dict` → `coordinator_state_dict`
5. 更新 `__init__.py` 导出
6. 同步所有 import / 变量 / docstring / 注释
7. 同步文档（CLAUDE.md + doc/*.md 全局替换）
8. 跑冒烟测试 `python tests/test_model_build.py`
9. 校验无残留：`grep -rn "TriBrain\|tribrain" .` 应只剩本方案文档自身

## 五、风险与验证

- **纯重构**：不改数值、不改逻辑、不碰权重初始化，冒烟测试应全绿（`test_tribrain_build` → `test_coordinator_build` 等函数名也同步改）。
- **checkpoint 键**：改键后旧 checkpoint 失效——目前无训练出的 checkpoint，安全。
- **文档一致性**：文档里 "TriBrain" 引用量大，漏改会导致前后不一致；第 9 步 grep 兜底。
- **`tri` vs `tribrain` 的区分**：改名时只动 "tribrain/brain"，保留 `tri`（ternary）语义的符号（如 `s_tri_residual`、`tri_graph`、`TriReflect`），避免误伤。

## 六、待确认

1. 是否把 `build_tribrain_graph` 一并改成 `build_ternary_graph`？（建议：是）
2. 是否把 checkpoint 键 `tribrain_state_dict` 一并改？（建议：是，趁无 checkpoint）
3. 文档是否全量同步替换？（建议：是，否则不一致）

确认后即可执行。
