# BiTSeed-Hybrid 在线初始化与实验协议

更新：2026-09-15。BiTSeed = **Conformer-Diverse Binary-to-Ternary Basin Seeding**，只命名初始化模块，不包含 selector、CoopDiff 或 post-ranker。

## 已确定与尚未确定

当前工作候选池为 **192 BiTSeed + 64 SDPI = 256**，在线、native-free、固定配置生成，不用人工 pose bank 或 LoRA。它是后续实验的统一基准，不是已证明全局最优的比例。

64-walker selector 尚未冻结：保留 Geometry-64、Diversity-64、Hybrid-48/16和Hybrid-56/8。B40/C40及C81在PROTAC22上的全池对照均已完成，完整表与retention见[研究文档第2节](../RANKING_PROTAC_RESEARCH.md)；MGD94下表仍是旧factorial模型结果，不能移植为C组成绩。当前PROTAC精修工作基线保留C40 EMA，C81为训练validation-loss最佳对照。

## 候选生成与预算

| 来源 | 配置 | 无失败时的二元轨迹数 | 候选数 |
| --- | --- | ---: | ---: |
| BiTSeed | C=4 conformers，每侧 M=8 poses | 2CM=64 | 192 |
| SDPI | 独立完整初态 | 0 | 64 |

每个 conformer 两侧共享同一完整 ligand 内部构象，分别做 frozen DiffDock rigid-only 采样；20点 schedule 对应19次二元更新。每侧8个 pose 组成8×8=64对，确定性均衡保留48对，每个 marginal pose出现6次；四个 conformer共192。这里64条**二元**轨迹不是64条完整 CoopDiff walker。

基础 seed S=31000；conformer stride为100000007，具体 binary seed计算以 [expert_anchors.py](../tricoopdiff/expert_anchors.py) 为准。Hybrid中的 SDPI候选 k 使用 `(S+1000000+k) mod (2³¹−1)`，因此同一配置下 broad 预算共享前缀；不要套用独立 `generate_sdpi_candidates` 的默认 stride。ID为 `["bitseed",c,i,j]` 或 `["sdpi",k]`。

从原始输入在线生成和实验审计缓存要区分：缓存仅用于固定候选反事实对照，不能人工筛选内容后冒充在线预测。

## 失败与补位：当前代码的确切行为

每个 conformer/side 最多尝试2M条 binary trajectory，连续3次失败提前停止；失败重采消耗额外计算。保留部分成功 pose，从可用笛卡尔积均衡选取最多48对，不要求先收满8×8。

缺额由新的 SDPI候选补齐；全部 BiTSeed失败时可退化成256 SDPI。实际来源可能不是192/64，须记录 partial/failed conformers、draw次数、异常和实际来源。

**SDPI proposal生成没有“四倍预算重试”的循环**：当前实现逐个调用基础 initializer；SDPI本身失败或未被捕获的输入/化学异常仍可使 case失败。不能承诺任何错误都自动补成256个有限候选。二元补采不足不必整例退出，但fallback也不是通用异常吞掉机制。

## Selector：数值过滤、几何排序、去重与回填

`_valid_candidate` 仅检查 ligand/POI/rotation tensor形状、非空坐标与有限性；不额外验证 atom map、RᵀR或det(R)。这些更强校验是生成/输入契约与后续审计要求，不冒充已实现过滤。

几何排序是字典序：先 `acceptable`，再蛋白 Cα<3 Å对数，再两侧最小Cα–ligand距离偏离3.5 Å及蛋白最小Cα距离偏离5 Å的和，之后separation与candidate ID。它不是学习 rank模型。`acceptable` 要求：

- 蛋白间无Cα<3 Å；
- 两侧最小Cα–ligand距离均在[1.5,8] Å；
- 蛋白间最小Cα距离在[3,15] Å。

不满足者**仍可入选**，不是hard reject。Diversity按该顺序在固定E3 frame贪心聚类，不额外对齐；相对簇代表同时满足ligand RMSD<2 Å、POI Cα RMSD<5 Å时归入同簇，不是连通分量聚类。

有配额时先保护较小来源配额、优先不同簇；随后用未选簇代表补位，最后才用重复簇候选补满。来源/簇不足时实际配额可以偏离请求。有限候选总数不足walker预算则报错。代码默认32请求24/8、64请求48/16，不等于科学上已冻结默认selector。

| 榜单名称 | 从同一256池如何选64 |
| --- | --- |
| SDPI-64 | 使用该池64个broad SDPI初态 |
| Geometry-64 | 无来源配额，无RMSD去重，按几何排序 |
| Diversity-64 | 无来源配额，有RMSD去重 |
| Hybrid-48/16 | 有去重，请求48 BiTSeed +16 SDPI |
| Hybrid-56/8 | 有去重，请求56 BiTSeed +8 SDPI |
| BiTSeed-only-64 | 只从BiTSeed分支做diversity选择 |

这里 Geometry-64 **不是**历史 Geometry-A8×64；Consensus-64也不在这张selector榜中。不要混用旧anchor初始化方法和当前选择器名称。

## MGD94 全池精修：主要证据

固定bound MGD94、seed31000、上述256池、20步CoopDiff、同一frozen factorial EMA checkpoint。全池每个候选精修一次，各selector只离线选相同轨迹子集；94例、24,064条全部数值有效，18,048 BiTSeed/6,016 SDPI，无记录失败。

主产物：`results/mgd_full_pool_refinement_20260913/summary_v1.json`，精确case清单见 `integrity.case_files`；配置与复跑见[运行指南](训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md)。

下表为每例所选集合的最大真实DockQ，再对94例平均。Initial/Final可以由不同候选取得；Union=max(Initial,Final)，不含中间步，**全都是oracle，不是部署top-1**。

| 方案 | Initial | Final | Union |
| --- | ---: | ---: | ---: |
| SDPI-64 | 0.2077 | 0.2515 | 0.2602 |
| Geometry-64 | 0.2376 | 0.2665 | 0.2862 |
| Diversity-64 | 0.2414 | 0.2668 | 0.2887 |
| Hybrid-48/16 | 0.2445 | 0.2624 | 0.2867 |
| Hybrid-56/8 | 0.2406 | 0.2472 | 0.2740 |
| BiTSeed-only-64 | 0.2380 | 0.2367 | 0.2636 |
| 全池256上界 | 0.3044 | 0.3097 | 0.3412 |

Geometry与Diversity有90/94例Final相同，均值差0.00029，不据此认定Diversity显著更好。相对SDPI的均值提升是描述性结果，不能省略配对不确定性。56/8与纯BiTSeed-64在当前权重下不作优先方案，不宣称其在新模型必然失败。

Geometry / Diversity / Hybrid48 的最终逐例平均retention为83.32% /83.49% /82.54%，exact oracle候选保留34/31/30例；初态与末态全池赢家ID仅25/94相同。因此初始化oracle改善并不自动解决最终选择问题。

## 来源收益与几何限制

全部BiTSeed-192的每例best均值从0.2726降至0.2631，SDPI-64从0.2077升至0.2515。但单候选均值分别从0.03979→0.04479、0.03531→0.05649，提升比例75.2%与86.7%。“BiTSeed最好初态被破坏”和“多数BiTSeed候选改善”可以同时成立。

Geometry/Diversity实际平均来源约40.6/23.4、39.4/24.6，不能只由这些计数断言“多选SDPI导致更高分”；候选身份与质量也不同。

初态acceptable仅2,306/24,064（9.58%），native sanity为92/94通过。该Cα代理没有验证末态的全原子物理合理性；不能说未通过初态gate的最终赢家无效，也不能据此把所有近距阈值直接升级为hard safety。先在内部验证审计初/末态几何与高质量候选召回，再修改规则。

## 历史候选比例证据及边界

MGD94早期proposal-oracle：SDPI-256 mean约0.257；B1(C4M8)带case-level fallback约0.313，C1(C16M4)约0.315但二元轨迹翻倍；B1∪SDPI共512候选约0.328。它们支持互补性，但预算、seed及fallback版本不同，不与上表直接作配对增益。

PROTAC22完整unbound exact proposal oracle，固定256候选及共享SDPI前缀，四组均22例完成：

| BiTSeed/SDPI比例 | Initial oracle mean | DockQ≥0.23 |
| --- | ---: | ---: |
| 0/256 | 0.1868 | 4/22 |
| 224/32 | 0.3381 | 20/22 |
| 192/64 | 0.3341 | 20/22 |
| 128/128 | 0.2991 | 18/22 |

224/32与192/64在21/22例oracle相同，仅7Q2J驱动差异；据此并结合MGD互补性保留192/64作工作配置。它不是MGD固定预算比例最优的证明，也不反映精修后优劣。早期三例smoke与83.2%的单例**近似**retention已不作为完整审计的替代证据。

历史proposal/比例产物与旧脚本从[当前工作](../todo.md)和Git归档追溯。完整22例selector审计的具体结果若引用，必须附对应报告；不再保留“正在GPU5运行”等过期状态。

## 后续冻结门禁

候选生成、selector和精修分别版本化。内部验证固定同一池/初态，比较旧模型与dual-init共享模型；检查两来源能力、initial/final retention、几何有效性、实际配额及失败分母。通过后才冻结selector。旧0.90/0.80 retention目标是研究验收目标，不是当前已达标事实。

最终部署仍需不读取native的top-1规则。MGD94和已用于决策的PROTAC22不能再充当未触碰holdout。
