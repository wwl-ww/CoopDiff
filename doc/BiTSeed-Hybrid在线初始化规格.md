# BiTSeed-Hybrid 在线初始化规格

更新时间：2026-09-13。本文件定义下一轮实现和审计的候选协议；在固定预算比例消融通过
前，`192/64` 是**待审计默认值**，不是已经证明最优的最终超参数。整个生成路径必须
native-free、在线且确定性；不使用 LoRA、人工 pose bank 或 MGD94 DockQ 进行选择。

## 默认预算

每例先生成恰好256个 proposals，再选择32或64个 CoopDiff walkers。

| 来源 | 配置 | 原始计算 | 进入候选集 |
|---|---|---:|---:|
| BiTSeed | `C=4, M=8` | 两侧共64条 frozen DiffDock trajectories | 192 |
| broad SDPI | 64个独立 `sdpi_v1` samples | 不调用 DiffDock | 64 |

BiTSeed 对每个 ETKDGv3 conformer 生成 `8×8=64` 个同构象 binary-pose 组合，再用
确定性均衡配对保留48个；每个 marginal pose 恰好出现6次，四个 conformer 共192个。
DiffDock 使用20点 schedule（19次更新）、rigid-only 更新和冻结 checkpoint
`checkpoints/score_model/best_ema_inference_epoch_model.pt`。SDPI 的每个 proposal 独立
采样 conformer、torsion、双侧 Haar rotation 和受体尺度自适应 translation。

基础 seed 为 `S=31000`。BiTSeed 继续使用现有 conformer/trajectory seed 公式；所有
消融组的 SDPI 候选 `k` 共享独立命名空间 `S + 1,000,000 + k`，因此32/64/128/256
预算严格构成同一候选池的前缀。候选 ID 必须带来源：
`("bitseed", conformer, e3_pose, poi_pose)` 或 `("sdpi", k)`。

## 确定性失败处理

1. 单条 binary trajectory 出现 NaN/Inf 时换新 seed 短补采；每个 conformer/side
   最多尝试 `2M` 次，连续3次失败则提前停止。
2. 不强制收满 `M×M`：保留两侧部分成功 pose 的笛卡尔积，用确定性边际均衡规则最多
   选48个；若不足48个，仅以同数目的新 SDPI proposals 补齐缺口。
3. 所有 conformer 均失败时，退化为256个 SDPI proposals。
4. SDPI 也按固定 seed 顺序补采；达到请求数四倍仍不足时才允许 case 失败。

任何补位都必须记录失败阶段、draw 数、异常类型、来源计数和最终候选数。禁止读取 native
决定是否 fallback。当前候选生成和 oracle CLI 已实现纯 SDPI-256、Hybrid-192/64、
保留成功 conformer 并按缺口补位，以及所有 conformer 失败时的整例 SDPI fallback。

## 过滤、去重与 Walker 选择

硬过滤只删除非有限坐标、形状/atom-map 错误和确定无效的刚体变换。现有
`acceptable` 几何标记暂作软排序特征，未经好姿态召回审计不得直接 hard reject。

所有候选在固定 E3 坐标系中全局聚类，不额外对齐。若两个候选同时满足 ligand
heavy-atom RMSD `<2 Å` 和 POI Cα RMSD `<5 Å`，暂视作同一 basin。阈值必须在 internal
split 上审计并冻结，不得依据 MGD94 调整。簇内按现有 native-free geometry penalty
选代表，再按簇轮询选择，避免一个高密度模式占满预算。没有 learned ranker 时固定配额为：

| CoopDiff walkers | BiTSeed | SDPI |
|---:|---:|---:|
| 32 | 24 | 8 |
| 64 | 48 | 16 |

若某来源或独立簇不足，按确定性顺序由另一来源补齐；最终 walker 数必须保持32或64。

该确定性 selector 已在 `tricoopdiff/expert_anchors.py` 实现。5T35真实回归得到188个
独立簇；x32与x64分别严格选择24/8和48/16个 BiTSeed/SDPI 候选，全部来自不同簇且
无 hard-invalid 候选。该例的 approximate oracle retention 为0.832，未命中全池最佳，
因此只能证明实现正确，不能证明筛选质量达标。

## 现有证据与待验证项

MGD94 的 exact proposal oracle 显示：纯 SDPI-256 为 mean/median `0.257/0.209`，
DockQ `>=0.23/0.49` 为 `39/10`；B1 (`C4M8`) 加 case-level SDPI fallback 为
`0.313/0.309`、`61/12`。C1 (`C16M4`) 仅提高到 mean `0.315`、`64/12`，却使用两倍
binary trajectories，因此选择 B1 作为结构化分支。DiffDock 成功的57例上 B1 mean
`0.270`，同例 SDPI mean `0.178`；但 SDPI 仍逐例赢16例，支持保留 broad 分支。

`B1∪SDPI-256` 的512候选 oracle 为 mean `0.328`、DockQ `>=0.23/0.49` 为 `65/13`，
只证明互补性，不证明固定256预算下的最佳比例。实现前后必须比较 `224/32`、`192/64`、
`128/128`，保持256候选、相同 seed 和 exact DockQ；默认值只有在 oracle coverage、去重后
basin 数和后续 retention@32/64 均不过度退化时才能冻结。

### PROTAC22 unbound 三例 smoke-test

2026-09-13 使用 `5T35_D_A_759`、`6BN7_B_C_RN3`、`7KHH_C_D_WEP` 做固定256候选的
exact proposal oracle。网络只读取 unbound 蛋白和 ligand；同目录 bound 蛋白仅作为
DeepTernary 原生方式的全原子 DockQ 计分模板。

| 方法 | exact DockQ（三例） | mean | DockQ >=0.23 | `acceptable` 总数 |
|---|---|---:|---:|---:|
| SDPI-256 | 0.249 / 0.128 / 0.284 | 0.220 | 2/3 | 34 |
| Hybrid-192/64 | 0.247 / 0.421 / 0.286 | 0.318 | 3/3 | 127 |

6BN7 的一个 conformer 在32次 draw 后只得到6/8个有效 binary poses；该 conformer 的
48个名额被自动替换为 SDPI，case 未失败。该三例 pilot 的纯 SDPI 使用历史 seed schedule，
因此只作功能性证据；正式比例消融改用上述共享前缀 seed。此结果验证了 PROTAC 输入兼容性、局部 fallback
和候选密度收益；由于只有三例且提升主要由6BN7驱动，不能据此冻结192/64比例或宣称总体
精度提升。

2026-09-13 完成完整 PROTAC22 exact oracle。四组固定256候选、共享 SDPI seed 前缀，
并使用 `2M`/连续3次失败的短补采协议；全部22例完成且零失败。

| 候选比例 | mean | median | DockQ >=0.23 | 相对SDPI win/tie/loss |
|---|---:|---:|---:|---:|
| SDPI-256 | 0.1868 | 0.1726 | 4/22 | — |
| BiTSeed/SDPI 224/32 | 0.3381 | 0.3223 | 20/22 | 20/2/0 |
| BiTSeed/SDPI 192/64 | 0.3341 | 0.3125 | 20/22 | 20/2/0 |
| BiTSeed/SDPI 128/128 | 0.2991 | 0.2837 | 18/22 | 16/5/1 |

224/32与192/64在21/22例的oracle完全相同，仅7Q2J由224/32额外保留的一个配对提高
0.0887；总体mean差0.0040且命中数相同。考虑MGD94上SDPI逐例互补证据，候选比例预冻结
为192/64；128/128已显示结构化候选预算不足，不再作为默认值。

另在 GPU 5 运行 Hybrid-192/64 的完整22例 x32/x64 native-free selector exact
retention 审计。初始化协议只在以下条件同时满足后冻结：完整比例消融无失败；192/64
相对224/32没有实质 oracle coverage 退化且优于128/128；x64 selected-oracle/full-oracle
的逐例平均比值至少0.90、x32至少0.80；来源配额、候选总数和确定性复跑全部通过。
若 retention 门禁失败，只修改 selector，不回看 native 调整候选生成或聚类阈值。
GPU 7另行运行共享同一候选与exact标签的预注册selector套件，包括组件、配额、RMSD阈值
和10个确定性随机配额基线；该测试用于解释默认规则，不按PROTAC22逐例结果连续调参。

## 审计清单

- 生成函数签名和 trace 中不存在 native、DockQ 或外部候选路径。
- 两类 seed 空间无碰撞；重复运行候选坐标 hash 完全一致。
- BiTSeed 同 conformer 配对、48/64均衡边际度和 frozen expert 身份可验证。
- 任一失败路径最终仍生成256个有限候选，并准确记录实际来源比例。
- 去重在固定 E3 frame 中执行，聚类阈值未从 MGD94 调参。
- 32/64 walker 配额、回填顺序和 tie-break 均确定性且与 native 无关。
- 分开报告 proposal oracle、selector retention、refined oracle 和 ranked top-1。
