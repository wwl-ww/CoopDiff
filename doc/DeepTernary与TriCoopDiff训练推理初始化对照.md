# DeepTernary 与 TriCoopDiff 训练/推理初始化对照

审计对象为本仓库当前代码和本地 DeepTernary 源码。TriCoopDiff 的初始化实现以 [sdpi 规范](初始化方案_gpt6.md) 为准；旧组件球壳与 5 Å 初始化操作说明已移除。

## 对照口径

| 对象 | 训练 | 推理 | 额外输入 |
|---|---|---|---|
| DeepTernary MGD | 固定 P1，Ligand/P2 组件级随机旋转和平移 | 历史 `predict_cpu.py` 做组件随机化；`predict.py` 的 bound 路径保留输入 P2 布局 | 两入口的约束不同，不能混报 |
| DeepTernary PROTAC | 组件随机化并带已知口袋信息 | 端点和二元口袋约束下的组件随机化 | warhead 原子、两端二元构象、两侧口袋 |
| TriCoopDiff MGD/PROTAC | 真实三元结构上的双界面 forward noise，加一次共享 torsion noise | 唯一 `sdpi`，完整 ligand 共享构象、两侧尺寸自适应 prior、精确装配，从 t=1 开始 | 当前未接入 warhead/口袋 sidecar |

`sdpi` 保留两侧局部初始化的所有相对距离，并与训练共用刚体组合函数。这不意味着其经验终端分布严格等于训练的 t=1 forward marginal，也不意味着已经获得 DeepTernary 的二元端点先验。

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


## TriCoopDiff 当前训练与推理

训练使用 [train.py](../tricoopdiff/train.py) 的共享时间和双界面独立刚体噪声；平移 σ 为 0.1→19，旋转使用 IGSO(3)，torsion σ 为 0.0314→3.14。训练保存实际采样变换，通过 [rigid.py](../tricoopdiff/rigid.py) 的解析群组合重建 POI，不再从 noisy ligand 用 Kabsch 猜回训练刚体变换。

推理从完整索引化化学图生成新 ETKDGv3 构象，随机 torsion 只执行一次。A/B 分别计算自身 receptor RMS radius，再独立采样整体 SO(3) 和高斯平移；A 固定，POI 由两侧变换组合确定。旧 `native`、`coupled5a`、`anchor_local` 不再是可运行的模式。

训练池是 [Phase 1 数据协议](Phase%201%20数据协议.md) 定义的 PROTAC/Molecular Glue 联合交集。不能再根据旧 wrapper 的默认参数推断当前训练池或现有 checkpoint 的训练来源；具体权重必须检查其版本、配置和数据记录。

当前诊断输入的蛋白来自晶体拆分，仍保留 bound 蛋白内部结构；重新生成 ligand 并不把蛋白变成 unbound 输入。因此这些结果是初始化/采样工程验证，不是官方 unbound PROTAC benchmark。

## 尚未实现的比较条件

二元 warhead pose、口袋、候选置信度与 prior adapter 仍是[后续设计](双二元强先验接入方案.md)。当前不能宣称已达到 DeepTernary PROTAC 的同输入条件。MGD 也需要单独固定 bound/unbound 口径和入口；不再建议在当前采样器恢复组件级 5 Å fallback。

完整评估必须固定数据、蛋白构象来源、先验来源、候选数和种子。真实 DockQ 选出的 oracle-best 与不依赖真值的 top-1 分开报告。
