# PROTAC 信息建模交接

## 分支与职责

### 当前实现更新：二元姿态参与自主生成（2026-09-17）

`conditioned_residual_consensus_v7_interface_geometry` 已接入逐步训练和采样。
在原 role/pocket/radial 输入之外，`binary1/binary2` 分别保存
`[N_maps, N_ligand, N_protein, 3]` 的距离、距离副本和置信度。蛋白节点顺序保留；
只计算各自二元复合物内部距离，没有跨蛋白的真实三元距离。
化学对称映射保留完整原子对应，以当前候选的整体距离残差选择每侧一个合法映射；
不再使用逐对独立区间，避免接受任何合法映射都无法解释的混合距离。
映射按化学原子序规范排序，重复映射去重，不读取真实三元结构进行选择。

`binary_geometry.py` 在当前候选上重算约束偏差，编码配体—残基消息，并沿
配体化学键传播。两侧的恢复方向和力矩通过可学习系数进入刚体 velocity；
两端约束还会投影到 sampler 的实际可旋转键及 moving-atom mask，直接进入
共享 torsion 更新。linker 构象沿原有 torsion 自由度更新，保留既有化学拓扑。
`structure_velocity.py` 统一 torsion 轴方向与 Kabsch 去刚体分量，并给出
与实际 sampler 一致的笛卡尔运动切向量。结构损失匹配同一 flow 状态/目标下
的原子运动及本侧二元距离变化率；不把带噪中间态直接强制拉到终态距离。
`--structure-loss-weight` 默认0.1，并记录/校验于checkpoint，恢复时不得静默改变。

`interface_geometry.py` 补充当前E3–POI、E3–ligand、POI–ligand界面消息，
复用现有蛋白/配体化学特征编码。边包含12 Å内的全部节点对和每个节点的
最近邻，以覆盖碰撞位置及分离状态；距离RBF、粗粒度重叠特征和化学信息
生成局部消息与方向，进入刚体和共享torsion输出。该通道不读取真实界面。
相应结构损失监督三个界面的距离变化率，目标仍为同一flow coupling的运动。
这一通道在二元条件缺失时仍可读取当前几何；只有二元条件通道随dropout清除。

训练数据构建、条件 dropout、manifest 加载、checkpoint 和训练/采样入口均已
连接。缺失二元条件会清除全部对应距离约束。冻结 DiffDock 保持原输入通道。
新架构从 C 模型迁移，门控零初始化；旧径向特征 checkpoint 无法直接加载为
新架构。新模块需要训练，旧实验指标不代表本架构的精度。

实现验收使用 `tests/test_binary_geometry.py` 和
`scripts/smoke_binary_geometry.py`；后者直接运行 SDPI/BiTSeed，记录实际来源
与 fallback，不读取真实三元结构评分，不需要 DeepTernary 预测结果。
当前训练链路烟测位于 `results/interface_geometry_build_smoke_20260917/`。
`tests/test_structure_velocity.py` 核对完整映射选择、采样器有限差分及刚体/torsion梯度。
距离约束在蛋白Cα节点分辨率工作，不等价于全重原子碰撞模型；独立top-1质量正在
PROTAC22上评估。两个训练seed已完成160轮，并分别按内部验证选定
epoch77与epoch41。v5/v6烟测目录保留为历史记录，权重不能直接载入v7。
full/pocket-only/absent结构损失包含不同数量的条件约束项，因此这些loss不能
直接相减解释为条件带来的预测收益。

- 分支：`protac`
- worktree：`/mnt/disk/wl/MoRE-tri/CoopDiff-protac`
- 基线：`model-v3` 的合并提交 `9002dda`
- 唯一主问题：**哪些 native-free、部署时真实可得的 PROTAC 信息，能够提高 TriCoopDiff 精修后的 DockQ？**

本分支负责条件资产、条件编码、训练—部署一致性和结构级验证。候选池排序、256→64 selector、post-ranker 与 walker 间排序通信属于 `rank` 分支，不在这里并行修改。需要共享接口时先写清张量或 manifest 合同，再协调合并。

## 已知结论，不要重复

当前候选与精修基线为 Hybrid-56/8、64 walkers、20步、seed 31000，父模型为 C40 EMA。PROTAC22 上三组严格配对结果为：

| 条件训练 / 推理输入 | Final | Union | Final ≥0.23/0.49 |
| --- | ---: | ---: | --- |
| default / pocket | 0.3066 | 0.3289 | 15/1 |
| default / absent | 0.3244 | 0.3374 | 17/2 |
| control / absent | 0.2936 | 0.3175 | 16/0 |

同一权重下 `pocket - absent` 的 Final 为 `-0.0178`（9胜/13负）。完整256池的 case-best 也没有改善高质量尾部。因此：

1. 6 Å Cα pocket-only mask **不能作为默认条件**。
2. default-absent 相对 control 的提升只能说明训练方式可能有益，不能声称模型利用了 pocket。
3. PROTAC22 中严格双侧 warhead 化学映射为 0/22；禁止用坐标最近邻或宽松子图结果伪造 atom map。
4. v2 pocket manifest 有修饰残基节点错位，已废弃；只能使用与 DiffDock/ProDy 节点顺序一致的 v3 资产。

完整证据见 `todo.md`、`Q.md` 和 `RANKING_PROTAC_RESEARCH.md`。不要从失败或 partial 目录拼接结果。

## 当前后台作业

原仓库 `/mnt/disk/wl/MoRE-tri/CoopDiff` 正在运行 P90/P100 多 seed 的训练与配对评估，使用封存 runtime。它们检验 pocket-only 训练—部署匹配，不代表新方案。不要停止、覆盖或从本 worktree 重启这些任务；等统一审计报告落盘后只补充结论。

生成数据、checkpoint 与缓存仍位于原仓库的 `data/`、`results/`、`checkpoints/`。本 worktree 不复制大文件；需要时使用绝对路径，并把新输出写入带协议和日期的独立目录。

## 下一阶段优先级

### P0：恢复可信的两端角色

先解决 warhead/靶蛋白侧的身份问题，再训练模型。优先使用有明确化学来源的片段 SMILES、键级/电荷修复和人工不可知的确定性消歧规则。任何 atom map 必须验证 canonical ligand identity、原子数、立体化学、唯一子图、连接键和两侧来源。无法唯一确认时标记 missing，不猜测。

### P1：同侧局部二元几何

在每个 protein–warhead 自己的局部坐标系中编码信息，例如：

- warhead 原子角色、linker attachment bond 与 linker atom mask；
- protein–fragment 距离/RBF、接触和方向性局部边；
- DiffDock 置信度或多姿态不确定性；
- 条件存在性、来源和可信度 mask。

不得输入 native ternary 坐标、两个蛋白的真实相对变换或由其派生的跨蛋白距离。条件应对每侧共同刚体变换保持不变或等变。

### P2：先做“信息是否被使用”的小实验

在正式长训前至少比较正确条件、全缺失条件、侧别交换、错误配对条件和几何扰动条件。要求同 checkpoint、同候选、同初态和同 diffusion seed。若正确条件与错误或缺失条件没有可重复差异，不进入大规模训练。

## 实验门槛

所有结构实验必须报告 Initial、Final、Union，以及 ≥0.23/0.49；同时给出逐例胜/平/负。主要因果量是**同一权重下正确条件减 absent**，而不是新权重减旧权重。进入默认协议至少要求：

- 全部预注册 case 完成，不能只统计成功子集；
- proposal pool hash、selected candidate ID、初态坐标/DockQ 和随机种子完全一致；
- Final 均值提高，Final ≥0.23 不减少，胜数不少于负数；
- 至少两个训练 seed 方向一致；
- 条件打乱或侧别交换后收益明显下降。

PROTAC22 是开发集，只能用于机制筛选；最终结论需要独立的 PROTAC holdout。case-best/oracle 只能判断上限，不能写成部署 top-1。

## 代码入口与检查

优先复用：

- `tricoopdiff/protac_conditions.py`：化学映射与本侧口袋审计；
- `tricoopdiff/condition_manifest.py`：严格部署 manifest；
- `tricoopdiff/condition_features.py`：条件张量；
- `tricoopdiff/conditional_consensus.py`：条件融合；
- `scripts/train_conditioned_entry.py`：配对训练；
- `scripts/evaluate_selected_refinement.py`：固定候选结构评估。

最小回归：

```bash
python tests/test_protac_conditions.py
python tests/test_condition_manifest.py
python tests/test_condition_features.py
python tests/test_conditioned_training.py
python tests/test_selected_refinement_contract.py
```

运行入口必须先导入 `tricoopdiff.diffdock_compat`。冻结两个 DiffDock 专家，不恢复 LoRA，不引入 SLURM，不让测试自动下载模型。提交前记录命令、checkpoint SHA256、manifest 指纹、数据 split 和输出路径。

## 首个交付物

先提交一份“可用信息资产审计”：逐 case 列出两侧 fragment 来源、严格映射状态、attachment/linker 定义、局部二元几何是否可用及失败原因。只有审计证明存在足够覆盖率后，才实现新 encoder；否则应明确报告数据缺口，而不是用 pocket-only 重跑替代。

## v3 二元先验实现（2026-09-17）

已实现 DeepTernary 同类的高可信输入链，但不复制其坐标近邻原子映射：

- `prepare_protac_binary_chemistry.py` 可从 CCD 恢复实际观测的裁剪片段，并按严格化学、仅电荷标准化、保守 MCS 分级记录角色映射。MCS 仅用于审计，不进入首轮部署 manifest。
- PROTAC22 实际审计恢复 44/44 侧；34 侧角色原子集合明确，11 例双侧不重叠。高可信门槛下 27/44 侧可用、7/22 例双侧完整。
- `make_protac_binary_manifests.py` 只接纳严格或仅质子化差异的映射，输出两端原子角色、linker、两侧口袋和各自 protein–fragment 二元坐标。生成和加载均不读取 ternary native。
- 条件 v3 将单一最近距离升级为 ligand/protein 双向径向距离轮廓；所有对称等价 atom map 共同平均。该表示对每侧独立刚体变换及化学对称置换不变。
- 新架构标识为 `conditioned_residual_consensus_v4_binary_radial_profile`。旧条件 checkpoint 维度不兼容，必须由同一 C40 parent 重新训练；旧 pocket-only 结果不能作为新模型成绩。

资产生成示例：

```bash
python scripts/prepare_protac_binary_chemistry.py --index <protac22.json> \
  --ccd-dir <ccd_dir> --root /mnt/disk/wl/MoRE-tri/CoopDiff --out <new_audit_dir>
python scripts/make_protac_binary_manifests.py --index <protac22.json> \
  --chemistry-report <new_audit_dir>/report.json \
  --root /mnt/disk/wl/MoRE-tri/CoopDiff --out <new_manifest_dir>
```

下一实验应先在这 7 个双侧 case 做同 checkpoint 的 correct / absent / side-swap / wrong-pair 配对机制检查。超过 DeepTernary 仍需独立结构评估证明，当前实现只补齐输入能力和可信资产合同。
