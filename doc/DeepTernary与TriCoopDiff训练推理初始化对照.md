# DeepTernary 与 TriCoopDiff 训练/推理初始化对照

> 审计范围：DeepTernary 本仓库实现，以及 TriCoopDiff `phase1-run` 提交
> `8d684382f5e498005715e7f9b0259807efaf0f42`。DeepTernary 同时保留了历史批量评测
> `predict_cpu.py` 和较新的用户入口 `predict.py`；两者的 MGD 初始化不同，本文分别记录。

## 结论

DeepTernary 的历史评测路径在训练和推理中都采用组件级初始化：固定 Protein 1，将 Ligand 和 Protein 2 分别做任意 SO(3) 旋转，并平移到公共原点附近 0–5 Å。TriCoopDiff 则在训练中使用从真实三元结构出发的连续 DiffDock forward noise，却在推理中默认使用 `coupled5a` 组件级 proposal，并将其视为 `t=1`。两者表面上都有“5 Å”，但只有 DeepTernary 在训练中直接学习了该初始化分布。

此外，DeepTernary 的 MGD 和 PROTAC 并不是同一个条件任务：MGD 主要依赖三元训练数据；PROTAC 还显式接收两个 warhead、两个二元配体构象和两个蛋白口袋。

## 总体对照

| 模型与任务 | 训练初始化 | 推理初始化 | 主要先验 | 对齐情况 |
|---|---|---|---|---|
| DeepTernary MGD | P1 固定；Ligand、P2 独立随机旋转和平移 0–5 Å | 历史评测执行相同随机化；新简化入口保留 bound 刚体布局 | 无 PROTAC 六项已知口袋输入；native 界面用于监督 | 历史路径基本对齐；新入口更容易 |
| DeepTernary PROTAC | 与 MGD 相同的组件级随机化 | 相同随机化，通常运行 40 seeds | 两端身份、二元端点构象、P1/P2 口袋 | 较好对齐 |
| TriCoopDiff MGD | 晶体三元结构 + 双界面连续 DiffDock noise | 默认 `coupled5a`，从 `t=1` 运行 20 步 | 无任务专属口袋输入 | 不严格对齐 |
| TriCoopDiff PROTAC | 与 MGD 使用同一训练算法 | 与 MGD 使用同一初始化器 | 尚无 warhead mask、二元构象和口袋先验 | 分布与信息均不对齐 |

## DeepTernary 的共同训练初始化

`TernaryPreprocessedDataset` 首先保存原生 Ligand、P1 和 P2 作为监督目标，然后构造模型输入：

1. P1 固定；训练时可随机交换 P1/P2 角色。
2. Ligand 先去质心，再施加任意旋转和平移。
3. P2 独立去质心，再施加另一组任意旋转和平移。
4. 平移方向随机，半径为 `Uniform(0, 5 Å)`，不是在 native pose 上做 5 Å 局部扰动。
5. 对当前 MGD/PROTAC decoder，模型训练阶段还加入随 epoch 衰减的坐标与特征高斯噪声：`2 × 0.5^(epoch+1)`；推理时关闭。

实现位置：[`ternary_pdb.py`](../DeepTernary/deepternary/models/ternary_pdb.py)、[`geometry_utils.py`](../DeepTernary/deepternary/models/geometry_utils.py)。MGD 和 PROTAC 配置均设置 `freeze='protein1'`、`random_flip_proteins=True`、`translation_distance=5.0`。

## DeepTernary MGD

MGD 使用 `align_method='p1-p2'`，没有启用 PROTAC 的六项 `known_pocket_info`。数据集仍从 native 三元结构计算 Ligand–P1、Ligand–P2 和 P1–P2 界面，并将相应坐标用于训练损失；区别是这些信息不作为六项已知口袋特征直接注入 decoder。历史批量路径 `predict_cpu.py::predict_one_bound` 虽然读取 bound complex，但仍会独立随机移动 Ligand 和 P2，且默认只运行 1 seed；这与训练组件级初始化基本一致。

仓库中的新入口 `predict.py::_run_mgd` 是另一种口径：它明确标为 bound setting，不对 P2 或 Ligand施加上述显式随机刚体变换。Ligand 输入仍是 RDKit 生成的内部构象，但 `get_lig_graph_revised` 会通过 Kabsch 将其对齐到输入 bound ligand 的刚体位置；P2则直接保留输入共享参考系。因此，报告 DeepTernary MGD 指标时必须同时说明所用入口，不能混用两种难度。

相关配置与入口：[`glue.py`](../DeepTernary/deepternary/configs/glue.py)、[`predict_cpu.py`](../DeepTernary/predict_cpu.py)、[`predict.py`](../DeepTernary/predict.py)。

## DeepTernary PROTAC

PROTAC 推理使用 unbound Protein 1/2、完整 PROTAC、两个 unbound 端点配体以及 `lig1_mask`/`lig2_mask`。模型显式消费：

- `p1_mask`、`p2_mask`：两侧蛋白口袋；
- `lig1_mask`、`lig2_mask`：完整 PROTAC 中的两端原子身份；
- `lig1_coord`、`lig2_coord`：两端二元配体构象。

训练时这些 mask/坐标由 native 三元结构的界面生成；unbound 推理时则由用户提供的两个端点配体及其对应蛋白坐标生成。程序先将二元端点构象映射回完整 PROTAC、推导口袋约束，再随机移动 Ligand 和 P2。随机摆放并未消除这些强先验，因此其任务不是 blind pocket search。这里还有一个版本差异：历史 `predict_cpu.py` 默认 `FIX_TWO_ENDS=False`，即 RDKit 生成完整构象时不固定两端，但仍用两个 unbound 端点构造 pocket mask/coordinate 条件；较新的 `predict.py` 默认 `use_fix_two_ends=True`，会在 RDKit 构象生成阶段也约束两端。报告结果时必须记录所用入口和该开关。配置见 [`protac.py`](../DeepTernary/deepternary/configs/protac.py)，输入构造见 [`predict.py`](../DeepTernary/predict.py)。

## TriCoopDiff 当前训练

当前 Phase 1 从真实三元坐标开始，以 E3 为固定参考系：

1. 采样 `t ~ Uniform(0,1)`，生产训练令 `t_pol = t`。
2. E3–Ligand 与 POI–Ligand 分别采样独立 DiffDock 噪声。
3. 平移标准差随时间从 0.1 Å 增至 19 Å；旋转使用 IGSO(3)；Ligand 另加 torsion noise。
4. 使用 Kabsch 将 POI 侧 noisy Ligand 对齐到 E3 侧 proposal，并将同一刚体变换同步应用于 POI。
5. 可随机交换两个蛋白角色，但没有 MGD/PROTAC 专属初始化分支。

实现见 [`tricoopdiff/train.py`](../tricoopdiff/train.py) 和 [`tricoopdiff/diffusion.py`](../tricoopdiff/diffusion.py)。在审计的 `phase1-run` 中，正式 Phase 1 wrapper 调用索引生成器时未传 `--task`，而索引生成器默认 `MGD`；因此该分支交付的 Phase 1 checkpoint 是 MGD 训练结果，不应视作已完成 PROTAC 专项训练。

尤其在 `t=1` 时，训练平移是标准差 19 Å 的高斯噪声；它不是“半径不超过 5 Å”的分布。训练构造还通过双 Ligand proposal 的 Kabsch 组合，使 POI–Ligand 分支对应一个合法的 DiffDock forward marginal。

## TriCoopDiff 当前推理

MGD 与 PROTAC 共用 [`scripts/evaluate_entry.py`](../scripts/evaluate_entry.py) 和同一个 sampler。默认 `coupled5a`：

- E3 固定；
- Ligand 随机 torsion、任意旋转，COM 放到 E3 COM 周围 0–5 Å；
- POI 独立任意旋转，COM 也放到 E3 COM 周围 0–5 Å；
- 从 `t_anchor=1.0` 开始20步反向扩散。

因此默认推理把半径至多 5 Å 的有界 proposal 标记为训练中的 `t=1`，但训练的 `t=1` 平移尺度是 19 Å 高斯分布，POI位置又来自双界面噪声的 Kabsch 组合。两者在半径分布和组件相关性上都不同。

当前 indexed 数据由三元晶体结构拆分，因此输入仍保留 E3、POI 与 Ligand 各自的 native 内部构象。`coupled5a` 会丢弃它们原来的全局相对位置和朝向，但不会把蛋白变成真正的 unbound 构象；Ligand 也从晶体内部构象出发后再随机 torsion。这一点与“仅提供独立 unbound 组件”的真实部署条件不同。

其他模式：

| 模式 | Ligand | POI | 用途与风险 |
|---|---|---|---|
| `native` | DiffDock `N(0,19²)` 平移 | 30–60 Å shell | 二体基线；POI–Ligand 容易 OOD |
| `coupled5a` | E3 COM 附近 0–5 Å | E3 COM 附近 0–5 Å | 类似 DeepTernary 组件摆放，但不是本模型训练 forward marginal |
| `anchor_local` | 观测界面 anchor 附近 | native POI pose 附近 | 局部/oracle 诊断，不能用于 blind 公平评测 |

初始化实现见 [`tricoopdiff/diffusion.py`](../tricoopdiff/diffusion.py)；时间和反演路径见 [`tricoopdiff/sampler.py`](../tricoopdiff/sampler.py)。

## 差距与建议

MGD 应先明确评测口径：若复现 DeepTernary 历史指标，应增加训练和推理一致的 `component5a` 模式，同时匹配其 RDKit ligand 构象、任意组件旋转和 0–5 Å 平移；若采用新 bound 入口，应保留输入蛋白共享参考系并单独命名，不能与 blind docking 混报。

PROTAC 应实现 `dual_binary_prior`：训练与推理都提供端点 mask、两个二元配体构象、两个口袋及其局部关键点。标准 DiffDock forward noise 可以保留为辅助分布，但未经相同 proposal 训练的 `coupled5a` 不应被称为严格匹配的 `t=1` 初始化。
