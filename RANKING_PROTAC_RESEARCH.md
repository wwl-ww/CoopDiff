# TriCoopDiff 下一阶段：Ranking 与 PROTAC 条件建模调研

调研日期：2026-09-15；执行状态更新至2026-09-16。范围：本仓库代码、C40/C81 已完成结果、DeepTernary 本地实现和原始论文/官方资料。设计章节仍是研究意见；第9节单独记录已经实现或启动的范围。

执行进展另见 [todo.md](todo.md)：32/8组采集及首轮排序已完成，160例训练扩展正在打标签并排队运行18项学习曲线；学习排序器此前仍无≥0.23的验证top1命中。
条件编码已推进到独立训练checkpoint、真实数据训练入口、四格配对验证与显式部署manifest；587/65簇的40-epoch正式条件/全缺失配对训练已经启动。P0化学恢复不等于二元来源验证。
以下整体架构仍是提案；已运行范围和当前限制以该执行记录及对应 manifest 为准。

## 1. 我的结论：先补齐信息和决策，再增加模型复杂度

推荐主线是：**冻结二元专家；以 C40 EMA 为当前精修基线；新增显式 PROTAC 条件；训练“当前质量”和“精修价值”两个不同的预测头；先做末端排序，再做前置分配，最后才研究 walker 交互。**

这里有两个不同瓶颈：

- **池里有没有好构象？** 主要由输入条件、候选生成、精修模型决定。排序不能找出池外不存在的构象。
- **能不能不看 native 找到池里的好构象？** 由 selector、post-ranker 和初末态选择策略决定。我们目前大多数成绩仍是 oracle，尚未解决部署 top1。

“大胆”应该体现在条件化搜索、未来价值监督和计算预算分配上；不应一开始把生成器、排序器、跨 walker 注意力、奖励梯度全部耦合起来。否则难以判断谁有效，也容易让不可靠的 ranker 把整个搜索带偏。

### 建议的最终系统

```text
蛋白 + PROTAC化学图 + 可选的两侧二元/口袋条件
  ↓ 条件来源、原子映射、局部坐标验证
在线候选生成（blind协议 / known-binary协议分别版本化）
  ↓ 数值检查 + 软几何特征 + 多样性描述
未来价值 pre-ranker + 集合多样性选择：256 → 64
  ↓
来源条件 + PROTAC条件的 TriCoopDiff 精修
  ↓ 保存初态及末态；可选少量中间态
结构质量 post-ranker：对保存的状态统一评分
  ↓
native-free top1 / top5 + 校准置信度 + 物理有效性报告
```

初版不做在线 walker 交换，不用 DockQ 选择实际输出。中间态是否纳入输出必须作为新协议记录，不能继续叫此前只含初末态的 Union。

## 2. 当前实验到底支持什么

### 2.1 C40 完整 PROTAC22 基准

固定 unbound 输入、192 BiTSeed +64 SDPI、seed31000、20步精修；22例、5632条全部数值有效。每例先取集合中最佳 DockQ，再跨例平均。以下均为 oracle，不是模型排序后的 top1。

| 方案 | 初始化均值 | 最终均值 | 初末态 Union | ≥0.23：初/终/Union | ≥0.49：初/终/Union |
| --- | ---: | ---: | ---: | --- | --- |
| SDPI-64 | 0.1313 | 0.1662 | 0.1809 | 3/5/6 | 0/0/0 |
| Geometry-64 | 0.2708 | 0.3041 | 0.3145 | 14/15/16 | 1/1/1 |
| Diversity-64 | 0.2722 | 0.3026 | 0.3144 | 14/15/16 | 1/1/1 |
| Hybrid-48/16 | 0.2628 | 0.3042 | 0.3100 | 13/16/16 | 1/1/1 |
| Hybrid-56/8 | 0.2745 | 0.3066 | 0.3188 | 15/16/17 | 1/1/1 |
| BiTSeed-only Diversity-64 | 0.2783 | 0.3047 | 0.3174 | 15/16/17 | 1/1/1 |
| 全池256 | 0.3168 | 0.3369 | 0.3512 | 19/18/20 | 1/1/1 |

来源：[C40 summary](results/protac22_dual_init_e40_20260914/c40_gpu7/summary.json)、同目录逐例 JSON 与 manifest。

三个直接判断：

1. 精修平均有收益，但仍会破坏好初态；全池 ≥0.23 从19例降到18例就是提醒。
2. 全池最终与 Hybrid-56/8 最终相差约0.0302；在固定 C40、固定池和末态集合下，任何只排序不改坐标的方法都不能超过0.3369这个均值上界。允许选择全池初末态时上界为0.3512。
3. 初态与末态的全池赢家 ID 只有 **7/22例相同**。因此“预测初始化 DockQ”不能代替“预测精修后价值”。这只是观察到的赢家切换，不证明 ranker 可以准确预测切换。

当前没有验证通过的 native-free top1，所以“post-ranking 可带来多少提升”还没有实测基线，不能把 oracle 差距直接当预计收益。

### 2.2 修正 retention 的表达

定义每例 `R_c = selected_best_c / full_best_c`，再平均；零分母需单独规定并报告。C40 这22例分母均可用。

| Selector | 初始化逐例平均 retention | 最终逐例平均 retention |
| --- | ---: | ---: |
| Geometry-64 | 84.06% | 88.90% |
| Diversity-64 | 84.50% | 88.53% |
| Hybrid-48/16 | 81.30% | 89.06% |
| Hybrid-56/8 | 85.40% | 89.77% |

此前说 Hybrid-56/8 保留约91.0%，是 `0.3066/0.3369` 的**均值之比**；与89.77%的逐例平均比值不是同一指标。今后以逐例口径为主，两者不能混标。

### 2.3 C81 best 现已全部完成，不再使用17例中途结论

原版非LoRA C best 为epoch81，选择依据是验证loss。22例、5632条均完成且有效；本次额外核对22个池hash、5632个初态hash和精修seed均与C40一致。

| 方案 | C40 最终 | C81 最终 | C40 Union | C81 Union |
| --- | ---: | ---: | ---: | ---: |
| SDPI-64 | 0.1662 | 0.1487 | 0.1809 | 0.1708 |
| Geometry-64 | 0.3041 | 0.2973 | 0.3145 | 0.3263 |
| Diversity-64 | 0.3026 | 0.2957 | 0.3144 | 0.3269 |
| Hybrid-48/16 | 0.3042 | 0.2915 | 0.3100 | 0.3219 |
| Hybrid-56/8 | 0.3066 | 0.2957 | 0.3188 | 0.3280 |
| 全池256 | 0.3369 | 0.3255 | 0.3512 | 0.3456 |

来源：[C81 summary](results/protac22_cbest_e81_20260915/summary.json)。

**最终榜 C40 更好；部分64候选 Union 榜 C81 更好。** 后者说明“破坏一些初态”和“在另一些初态上产生互补收益”可以同时发生，不能简化为 C81 全面无用。先用 C40 建立末端排序基线，再评估排序器能否兑现 C81 的 Union 优势；这不授权把 C40/C81 结果合池增加预算。

“训练勉强收敛即可”不是上述实验的结论。应按部署目标选择checkpoint：Final、可实现的初末态 top1、两来源保持和物理有效性，而非只看验证loss。

## 3. 原始研究给我们的启示

以下是可迁移思想，不是这些方法已在本项目奏效的证明；不横向拼接不同benchmark的成功率。

| 原始来源 | 与本项目有关的事实 | 我的取舍 |
| --- | --- | --- |
| [DiffDock，ICLR 2023](https://arxiv.org/abs/2210.01776) | 生成与置信度排序分开建模 | 首先建立独立 pose-quality ranker；二元置信度不等于三元质量 |
| [GNINA 1.0，2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8191141/) | 区分末端重排、精修中评分、全程使用CNN；默认采用末端重排 | “排名模型应该介入哪里”必须做消融，先从低耦合后排序开始 |
| [DiffDock-L / Confidence Bootstrapping，ICLR 2024](https://arxiv.org/abs/2402.18396) | 用置信度反馈改善生成，并关注未见口袋泛化 | 可作为远期闭环；不能拿未校准自评分直接制造真值 |
| [FlowDock，v3，2025](https://arxiv.org/html/2412.10966v3) | apo→holo flow 与结构置信度/亲和力输出分工；原结构置信度头来自预训练并保持冻结 | 支持复用表示，但“冻结且已训练的置信度头”与我们冻结未排序监督的头不是一回事 |
| [Set Transformer，ICML 2019](https://arxiv.org/abs/1810.00825) | 对集合元素关系建模，处理排列不变输出 | 适合候选间冗余/互补；不是第一版必须上的网络 |
| [FK steering，ICML 2025](https://proceedings.mlr.press/v267/singhal25b.html) | 用中间潜势对粒子重采样，进行推理期引导 | 给 walker 协作提供思路；不能把其理论保证直接移植到我们确定性flow与启发式扰动上 |
| [DeepTernary，Nature Communications 2025](https://www.nature.com/articles/s41467-025-61272-5) | PROTAC使用known-pocket unbound条件；多构象预测后按预测对齐误差排序 | 必须补齐输入条件合同，再比较生成与排序能力 |
| [PROflow，2024原始预印本](https://arxiv.org/html/2405.06654v1) | 给定或对接得到两侧warhead姿态，建模linker兼容的蛋白装配，并用伪三元数据训练 | 借鉴linker兼容空间与伪数据，不直接复制其优化/投影到现有flow |
| [DiffLinker，原始论文](https://arxiv.org/abs/2210.05274) | 根据三维片段条件生成连接分子 | 借鉴片段条件表示；我们已知完整PROTAC拓扑，不需要顺手改成新linker化学设计 |
| [PoseBusters，Chemical Science 2024](https://pubs.rsc.org/en/content/articlepdf/2024/sc/d3sc04185a) | 姿态精度之外还应检查化学和几何有效性 | DockQ、二元配体姿态和物理有效性分别报告 |

特别说明：DeepTernary论文报告的PROTAC均值0.65不能与本项目blind-like条件下的0.3512 Union直接当公平差距。但在当前固定状态集合内，排序无论多好都无法产生0.65的均值；若要达到明显更高的结构质量，必须同时改变候选/精修质量。这是集合上界推论，不是认定对方指标已经按本地口径复现。[论文来源](https://www.nature.com/articles/s41467-025-61272-5)

## 4. 代码审计：哪些东西已经有，哪些只是名字像有

### 4.1 TriCoopDiff当前能力

基于C40实验封存的 [consensus.py](results/protac22_dual_init_e40_20260914/runtime/tricoopdiff/consensus.py)：

- 第59行附近：C组有 `source_embedding`，区分SDPI/BiTSeed；它不是“PROTAC口袋条件”。
- 第71行附近：四个gate控制对运动建议的信任，代码明确不是校准置信度。
- 第79行附近：`judge_head` 和 `state_head` 被 `requires_grad_(False)`，本轮训练不对它们施加排序监督；不能仅凭返回字段名 `confidence` 当作已训练并校准的三元rank分数。
- 第135行附近：已暴露 `rank_features=context`，适合先做冻结表示上的probe。
- [pose_ranking.py](results/protac22_dual_init_e40_20260914/runtime/tricoopdiff/pose_ranking.py) 已有小MLP `PoseRanker` 与同case listwise loss，可复用骨架；类存在不等于已有可部署训练权重。
- [geometry_encoders.py](results/protac22_dual_init_e40_20260914/runtime/tricoopdiff/geometry_encoders.py) 已有类型/局部配对几何信息，不应误称模型完全不看化学和局部界面；真正缺的是显式端点、linker角色和二元条件输入。

[sampler.py](results/protac22_dual_init_e40_20260914/runtime/tricoopdiff/sampler.py) 第317行明确 `add_noise=False`。给定同初态与模型，反向过程是确定性的；没有一个每步自动探索的新噪声源。`return_trajectory` 可返回轨迹，但当前全池评估入口未保存完整坐标。

[evaluate_selected_refinement.py](scripts/evaluate_selected_refinement.py) 的逐候选JSON保留DockQ、几何代理、ID、hash、seed等；末态坐标在评分后未作为训练样本落盘。现有JSON可做标签/选择审计，**不是完整几何ranker训练集**。若要用真实末态图或 `rank_features`，需按冻结版本确定性重放并保存，不能根据DockQ反推特征。

### 4.2 DeepTernary给PROTAC额外输入了什么

本地DeepTernary checkout：`827821dccca31a5918bd0355e2d6bf70c072b6dd`；调研时工作树干净。以下判断来自代码，不把变量名当科学定义。

| 条件 | 本地证据 | 真正提供的信息 |
| --- | --- | --- |
| 两侧蛋白口袋mask | [protac.py](DeepTernary/deepternary/configs/protac.py) 的 `p1_mask/p2_mask` | 哪些残基参与对应二元界面 |
| 两端配体原子mask | 同配置的 `lig1_mask/lig2_mask` | 哪些原子属于各自结合端 |
| 二元端点三维坐标 | 同配置的 `lig1_coord/lig2_coord`；[predict.py](DeepTernary/predict.py) 第402行附近读取 `unbound_lig1/2` | 端点相对各自蛋白的结合姿态，而不只是口袋中心 |
| 节点条件嵌入 | [iegmn_gtval.py](DeepTernary/deepternary/models/iegmn_gtval.py) 第122–178行附近 | 将口袋/端点身份加到蛋白和配体特征 |
| 条件化坐标处理 | 同文件 `pocket/aligned_pocket` 模式、keypoint Kabsch装配 | 用端点条件组织linker构象及蛋白装配 |
| 预测结构误差 | [losses.py](DeepTernary/deepternary/models/losses.py) 第358–365行附近；`predict.py` 第663行附近排序 | 学习输出质量代理，低预测误差优先 |

这表明“PROTAC额外信息”优先不是更多ESM维度、泛泛的分子量或E3类别标签，而是**两端身份、两侧口袋、各自二元几何、连接关系**。

三个容易误读的实现细节：

1. `FIX_TWO_ENDS` 不宜翻译成“把两端在共同世界坐标中永久钉死”。[predict.py](DeepTernary/predict.py) 的 `get_rdkit_coords_with_fixed_ends` 调整的是各端子结构内部距离bounds；后续还有对齐。`predict_cpu.py` 默认 `FIX_TWO_ENDS=False`，较新 `predict.py` 的对应参数默认True，复现必须固定入口。
2. `p2_rmsd_pred` 不是标准AF式逐残基PAE矩阵。本地loss构造目标时用了坐标 `L1Loss`，不能不经核对就称为严格RMSD。我们可以学误差头的任务分工，不应照搬命名和损失当作正确的DockQ代理。
3. `replace_to_unbound_coords` 用输入完整配体与片段的空间近邻（<1 Å）做对应；另有对RDKit构象向输入配体Kabsch对齐的路径。这依赖预处理坐标关系。**不是凭此就宣判数据泄漏**，但新入口不能依赖目标三元布局来恢复原子映射或蛋白相对位置。

官方代码入口可交叉核查：[DeepTernary repository](https://github.com/youqingxiaozhua/DeepTernary)、[predict.py](https://github.com/youqingxiaozhua/DeepTernary/blob/master/predict.py)。本地上述commit用于固定审计对象；远端master会变化。

## 5. Ranking在哪些阶段参与：三个任务，而不是一个分数包打天下

令候选 `z_i` 为初态，`Rθ(z_i)` 为固定精修器的末态，真实质量分别为 `q_i^I`、`q_i^F`。训练标签可以读取训练集native；部署输入绝不包含native。

| 模块 | 输入时可知道什么 | 应预测什么 | 不该承担什么 |
| --- | --- | --- | --- |
| 当前质量头 Q | 当前几何、化学、可用二元条件 | 当前DockQ/可接受概率/界面误差 | 不把高质量自动解释为值得继续精修 |
| 未来价值头 V | 初态或中间态、时间、来源、条件、固定精修器身份 | 继续指定剩余计算后的质量或增益 | 不能读取尚未生成的末态特征 |
| 集合选择策略 A | 一组Q/V、相似度、预算、来源信息 | 哪64个互补候选值得计算 | 不将复制数或簇大小当作正确概率 |

### 5.1 先训练 post-ranker：解决真正的输出选择

第一版对64个初态和64个末态统一评分，共128个状态，输出top1/top5。另保留“只排64个末态”的消融，量化保留初态能否转化为真实收益。

推荐输出：

- `q_hat`：当前DockQ回归值，限定到[0,1]。
- `p23/p49`：DockQ达到0.23/0.49的概率；采用有序分桶或其他一致性约束保证 `p49 ≤ p23`。
- 可选辅助头：界面误差/接触质量、两侧端点偏离、物理有效性；不是亲和力或降解活性预测。

训练目标：同case pairwise/listwise排序 + 阈值分类 + 温和的质量回归。初末态配对差值可作辅助监督；不能只训练“末态是否改善”的分类器，因为从0.01升到0.02的构象未必值得输出。

同case优先排序是为了抑制蛋白大小、配体大小、靶点难易度这些跨case捷径。所有低质量候选也需要明确质量标签，不能只有相对顺序；几乎并列的候选不强迫产生很大的分数间隔。

初态/末态分布必须同时进入训练与校准。最终选择器优先用当前结构质量，阶段、来源等元信息做有/无消融，防止学成“见BiTSeed就加分”或“末态总优先”。对冻结表示的提取固定明确的诊断时钟；不能仅拿最后一次Euler更新前的隐藏量冒充更新后几何的特征。

### 5.2 再训练 pre-ranker：预测精修后的价值

初版用两个头，避免混淆部署策略：

```text
V_final(z)  ← q^F
V_union(z)  ← max(q^I, q^F)    # 这是训练标签，不是推理时读取真实值
G(z)        ← q^F - q^I        # 可选增益辅助头
```

若只交付末态，以 `V_final` 为主；若保留初态，`V_union` 更接近预算分配目标，但它仍是oracle潜力，真正top1能否兑现要由post-ranker验证。不要把 `V_union` 的预测称为已实现成功率。

更进一步，若全部256个初态都免费存入输出档案，预选择的目标就应是“精修对已有档案的新增价值”，而不是再次优先高 `q^I`。该版本的输出集合为256初态+64末态，预算/候选合同不同，要单列；第一版仍用所选64的初末态，保持可比性。

未来标签依赖精修器：C40、C81、未来加条件的模型不能无标签混用。初版锁定C40；换精修器至少重采校准集。当前确定性flow一次重放给出一个确定标签，不可伪装成多次独立rollout估计的期望。

### 5.3 集合选择：简单多样性优先于立刻Set Transformer

先比较：几何基线 → MLP/树模型未来分数 → 未来分数+贪心去冗余。质量分数描述候选本身；多样性奖励只决定是否值得额外分配一个walker。

候选相似度至少联合：固定参考蛋白frame中的另一蛋白姿态、双侧接触指纹、linker扭转/端点几何。不能分别对齐蛋白后把两个不同装配判成重复；也不宜仅用完整配体RMSD，让较长linker掩盖端点错误。

初版可以实现 `选择分数 = V_i − λ·max(sim(i,j), j∈已选集合)`，但λ需在内部验证集确定。这是启发式，不保证最大化expected-best。数据足够后再比较Set Transformer的集合上下文与learned marginal utility；只对候选级embedding做注意力，不对256份全部原子图做全连接跨候选attention。

不赞成在训练前就永久每簇只剩一个代表：近似簇内部可能具有不同精修结果。保存完整池与簇成员，选择时软限制；用最终label审计被删成员。来源配额保留无配额、48/16、56/8作为对照，避免从22例再调出一个新“最佳比例”。

## 6. Ranking输入：让模型看到三元几何，不让标签混进来

### 第一版的特征层次

1. **便宜物理/几何特征**：双侧接触距离分布、归一化clash、配体与各口袋的覆盖、蛋白相对姿态、linker构象/扭转、端点误差。接触数/BSA须按接触原子或面积尺度归一化，不能默认为越大越好。
2. **冻结学习表示**：复用现有 `rank_features`，与上述特征组合；验证它相比仅手工特征是否有增益。得到该表示可能仍需两个专家前向，不是零成本；若用于256预排序，明确计入expert调用数。
3. **轻量三元图质量网络**：当frozen probe不足时，单独训练局部蛋白重原子+完整ligand的等变/不变编码器。蛋白内部可缓存，跨分子边按当前候选重建；优先覆盖两侧二元与蛋白—蛋白界面，不一上来全蛋白全原子大模型。

可选轨迹特征包括位移、torsion变化、端点漂移、专家方向分歧和几何稳定性。**pre-ranker只能使用决策之前已产生的轨迹**；完整20步轨迹特征只可用于post-ranking或事后分析。

### 明确禁止/警惕

- `initial_dockq/refined_dockq/safe_dockq/fnat/lrms/irms`、native接触、`safe_stage`、基于真值的最佳ID不能作为输入。
- `fit_rmsd`须核查含义：BiTSeed内部装配拟合残差不等于与native的误差，也未必有足够区分度。
- 专家输出的score/速度范数不是confidence；当前仅加载score checkpoint，不能虚构现成的二元置信度。若另接DiffDock confidence模型，要核实权重、输入分布与大PROTAC校准。
- 原始候选ID、PDB ID、绝对序号、生成失败文本、文件路径不作为学习特征；hash只用于完整性审计。
- 物理分数不能替代降解活性，DockQ也不是DC50/Dmax标签。
- 数值 `valid=True` 不等于化学结构正确，当前Cα gate不等于全原子物理检查。先audit召回再把几何阈值升级为hard reject。

## 7. Ranking训练数据：最大工作量在标签生产和划分

### 7.1 先处理已知数据问题

本次直接读取当前 `data/ternary_train.json` 和 `data/ternary_val.json`：分别6834/253 entries、587/65 clusters，按case的四位PDB ID检查有 **83个交集**。这是当前磁盘索引的审计结果，不代替每个历史checkpoint的索引hash溯源。

不能直接拿现有val当新排序器的干净泛化验证。应生成新的版本化索引：同PDB、同蛋白对/同源簇、同配体或linker系列按所声明的泛化任务合组。将这些相似关系做组约束时可能出现大连通组，必须报告最终组数与分布，不能为了凑比例拆开泄漏组。

所有同一case的初态、末态、中间态、扰动、不同checkpoint输出及伪数据衍生物必须在同一split。若验证目标要衡量对未见复合物的生成泛化，生成器训练也要排除对应组；冻结C40在其已见训练case上生成ranker验证标签，只能说明有限范围内的排序泛化。

MGD94和PROTAC22已反复用于开发决策，不再是未触碰holdout。它们可用于当前诊断与公开对照，不用来拟合ranker或反复调整阈值。最终应增加按时间/蛋白家族/linker系列划分的独立评估。

### 7.2 数据记录与标签

每条记录至少保存：

```text
case/group/split + 输入与条件的来源hash + generator/checkpoint/runtime hash
candidate_id + 实际source + conformer/binary seed + fallback类型
初态坐标 + 末态坐标 + 可选中间态坐标 + 原子/残基映射
可推理特征（独立字段）
训练标签（独立文件/字段）：DockQ、阈值、物理有效性、初末态差值
实际耗时、expert调用数、轨迹/采样配置
```

生成器和特征提取进程不获得测试native路径；评分进程单独读取native并通过ID关联标签。测试时用“移除native仍可运行”的集成检查防止误用。训练生成路径需要训练native构造监督并不违规，但不能把这些坐标或理想化映射作为测试输入。

包括真实生成的坏姿态、单侧好/另一侧坏、双侧接触但蛋白界面错、好初态被破坏、差初态被救回、SDPI fallback、端点条件失真。不要只用在native附近加噪声的样本：它们不能覆盖真实推理错误分布。

合成近native姿态可补充稀缺高质量标签，但保持标识和比例控制；在验证集报告真实生成分布的校准，不把合成高分数量当系统召回提高。失败轨迹不能静默丢弃，可学习失败风险，最终失败case仍计入分母。

### 7.3 可控的采集规模

先做数据管线烟测，而不是一次性全7000例×256×多个checkpoint：

- 小烟测：16个训练组，每例32条，验证坐标重放、特征不泄漏、loss下降与source路由。
- 第一轮：约256个训练case，每例从完整256池中按来源/几何/簇分层抽64条精修；约32个干净验证case精修全256。它们是启动预算建议，不是已经存在的干净split。
- 没有被精修的候选不能标成“最终失败”。子采样会产生选择偏差；记录纳入规则/概率，训练时按case平衡，必要时做加权或补采。验证集必须全池审计pre-selector retention。
- 初版确认有泛化信号，再扩到更多簇与多个candidate seed。优先覆盖新的蛋白/linker类别，不优先堆同case的近重复候选。

按此次C40每条含生成摊销约4.53秒粗估，`256×64 +32×256=24576`条约31 GPU小时，仅指标签采集；还不含新增几何特征、ranker训练、条件模型训练。分子大小与共卡负载不同会显著改变耗时；先用烟测实测吞吐。不要用0.65“准确率”或论文单构象秒数推算我们的全池成本。

## 8. Walker是否应该交换信息

### 8.1 三个层级，按这个顺序做

| 层级 | 机制 | 建议 |
| --- | --- | --- |
| 共享输出档案 | 独立精修，共享初末态embedding，最后统一排序/去重 | 第一版就做；风险低、容易解释 |
| 共享计算分配 | 在少量检查点，根据未来价值停止低潜力轨迹，把预算用于其他轨迹 | 第二阶段，先训练中间态V再上线 |
| 共享运动/粒子繁殖 | 跨walker消息影响速度，或重采样并扰动优秀粒子 | 研究分支；必须有等预算收益和多样性证据 |

初末态组成输出档案不等于粒子之间修改了轨迹，也不应包装成已实现交互采样。

### 8.2 为什么“复制好walker，换seed继续”在这里不成立

当前确定性flow没有逐步噪声，相同状态、时间、来源和条件会产生相同后续更新。克隆会浪费计算；仅换seed不保证产生任何新构象。

若研究粒子重采样，需要明确定义rejuvenation：在当前时钟沿允许的linker扭转/局部刚体自由度扰动、保持source与条件身份、重新计算局部图与专家读数，并验证其处于模型训练支持的区域。若要可称为正确SMC，还需给出目标分布、过渡核及权重修正；否则就称“启发式自适应多walker搜索”，不能借FK论文替代证明。[方法启发](https://proceedings.mlr.press/v267/singhal25b.html)

维护精英档案、簇覆盖、来源覆盖与祖先ID；记录effective sample size和克隆坍缩。给探索分支留固定预算，不允许所有walker瞬间变成同一个高自评分但错误的basin。

### 8.3 首个等预算实验：先做筛减，不急着繁殖

原预算为 `64×20=1280` 次TriCoopDiff状态更新。可试：

```text
128候选各走4步 → 中间价值头选48个 → 各继续16步
128×4 +48×16 =1280次更新
```

比较相同1280更新预算的64×20；这里最终只有48条完整轨迹，必须明确说明。若拿 `256×4 +64×16=2048` 的方案比较64×20，那是1.6倍预算，不是免费加速。

分配计算还要计入ranking、原子图构建和两次expert前向，不能只计训练模块FLOPs。中间态V应针对相同数值schedule和剩余步数训练；不能用末态Q直接淘汰暂时几何较差但可恢复的轨迹。

### 8.4 真正大胆的方案放在哪里

后续可以做candidate-level Set Transformer：每个walker形成不变摘要，通过集合attention产生“继续/停止/扰动幅度/条件信任”控制量；先不输出跨walker平均坐标或平均旋转。后者会混合不兼容basin并破坏化学几何。

如果再做rank-guided velocity，单独记录新增引导项、梯度裁剪和几何约束，检查reward hacking。我们的flow速度并非自动是某个势能梯度，所以不应强制“运动场与rank能量共享一个标量势”作为默认架构。

## 9. PROTAC条件：应增加哪些信息，如何保证可用

### 9.1 输入合同分层

| 模式 | 输入条件 | 可以主张的任务 |
| --- | --- | --- |
| U：无已知口袋 | 两蛋白、完整PROTAC图，在线专家生成 | 当前blind-like基线 |
| F：片段已知 | U + 两端原子角色、linker、连接键 | 利用化学设计信息，不使用二元晶体姿态 |
| P：口袋已知 | F + 两侧口袋残基/软概率 | pocket-conditioned三元装配 |
| B：二元姿态已知 | P + 两侧片段相对各自蛋白的坐标/对应关系 | known-binary PROTAC装配，适合对齐DeepTernary条件 |

一个模型可支持缺失条件，但评估必须分模式。已知两侧二元结构通常不等于已知两蛋白相对位置；每个二元复合物在独立局部frame中输入，不能夹带目标三元的刚体布局。

建议每侧条件记录：

```text
fragment_atom_map, attachment_bond, pocket_residue_map
binary_fragment_coords_in_own_protein_frame
availability_mask, confidence/uncertainty
provenance = experimental_binary / template / docked / predicted / absent
```

`source`仍表示当前候选的生成来源；`provenance`表示条件来源；`availability`表示条件有无。三者不得压成一个0/1。原候选fallback为SDPI时，source=0，但它仍可具有可靠的口袋条件。当前v2将provenance保留为manifest/报告审计字段，但**不送入神经网络**：通用训练只见过native监督，无法校准template/docked/predicted的独立embedding。

### 9.2 原子映射与坐标检查是第一优先级

- 用化学图、原子映射编号与立体化学建立片段到完整PROTAC的对应；对对称多重匹配保留等价解或按不依赖native的规则处理，不能靠目标三元坐标最近邻选“正确映射”。
- 蛋白残基对应通过序列/结构映射取得，记录缺失残基、替换、链角色与覆盖率；已知口袋裁剪必须只用允许的二元条件。
- 对两侧蛋白—片段对分别做随机SE(3)变换，预测装配应满足协变性；改变无关输入文件的绝对摆放不应改变最终相对结构质量。
- 独立设置fragment geometry和protein-local binary pose：内部形状约束不等于在共同frame中钉死两端位置。
- 去掉bound/native文件后应仍能生成全部输入和候选。对方示例文件可用于复现实验，但新接口必须通过这一测试。

本地PROTAC22例目录有 `unbound_lig1/2.pdb` 等资产；目前只确认文件可见，不等于22例映射与来源都已审核通过。先生成条件资产manifest，不直接把这些文件作为训练标签混入新训练。

## 10. 如何让条件真正影响初始化和精修

### 10.1 条件初始化：不要拿已知二元姿态还重新盲搜完整PROTAC

无条件模式保留BiTSeed工作池。双二元条件模式新增独立版本：

1. 从完整化学图获取两端、linker与attachment方向。
2. 在保持端点内部合理性的前提下生成多种linker构象。
3. 将一端对齐其本地二元参考；依据另一端在完整配体上的位置，刚体放置另一蛋白及其绑定片段。
4. 使用全原子clash、端点适配、linker strain等软指标排序/去重；允许二元条件有误差，必要时保留多个二元姿态。

两端都已知时，不必人为维持M×M的blind搜索形式。主要采样维度会转为linker构象、连接键扭转、端点微扰及二元姿态不确定性。原192/64在该模式中只是待比较的预算参考，不再自动是最佳比例。

ETKDG的坐标/距离约束是可用起点；RDKit文档明确区分 `coordMap` 与 `boundsMat`，化学图固定时可做受约束构象生成，不必生成新分子。[RDKit官方文档](https://rdkit.org/docs/RDKit_Book.html)

**不能把两蛋白固定在任意相对位置，再要求linker强行闭合。** 至少一侧蛋白姿态需要随连接构象变化；遇到不兼容条件应放宽可信度或报告失败，而不是拉坏键长来满足约束。

### 10.2 条件精修：第一版最小改动

在C架构上加单独condition encoder，不修改冻结DiffDock内部权重：

- ligand节点加入 `E3端/POI端/linker/未知` role embedding，连接键单独标识；一般三元小分子允许双侧接触重叠，不能强迫分成PROTAC式三段。
- protein节点加入口袋软mask、条件可用性和来源embedding。
- 加两侧端点局部接触tokens与linker token，输入现有ternary协商模块；固定的source条件继续保留。
- 辅助几何：端点偏离的SE(3)一致向量/不变残差、连接方向、可达性、局部clash；旋转向量/叉积的极性处理沿用现有等变合同并测试。
- 输出仍为兼容现有运动参数化的刚体/torsion更新；不要同时换成完全自由的原子坐标生成器。

若只增加role embedding有效但幅度小，再加强局部重原子交互；单一全局token可能表达不出端点接触方向，这是比盲目加Transformer层数更明确的扩容理由。

### 10.3 为什么不能只把口袋mask塞进旧模型推理

训练时模型必须见过条件的含义、缺失、噪声和错误。新输入通道没有经过训练就启用，不保证产生有意义行为。添加几何投影也会改变数值轨迹，应先作为独立启发式消融，不能继续声称完全相同flow。

建议先soft-conditioning，再比较soft-restraint：

`L = L_flow + λ_endpoint L_endpoint + λ_geometry L_geometry + λ_contact L_contact`

这些是拟议辅助项：`L_flow`监督真实训练路径速度；端点/接触/化学约束主要作用于重建clean端点或短rollout，而非强迫所有带噪中间态已经满足最终接触。错误二元条件必须按不确定性降权。

若训练路径本身也改成受约束投影路径，需要重推/数值验证路径导数与目标速度；不能投影坐标后仍拿投影前速度当正确标签。训练与推理clock必须一致，本地定义是prior `t=1`、clean `t=0`，不要照抄别的flow论文相反的时间方向。

将来可用投影保证linker兼容或局部最小化，但要证明不会导致手性翻转、环畸变、端点过约束和评分偏好虚假高clash接触。PROflow提供了条件空间的研究先例，不替我们完成上述验证。[PROflow原文](https://arxiv.org/html/2405.06654v1)

## 11. 没有大量PROTAC三元晶体，怎么训练

### 三层数据，而不是拿22例反复微调

| 层次 | 数据/构造 | 教给模型什么 | 局限 |
| --- | --- | --- | --- |
| 通用三元监督 | 清洁划分的TernaryDB；训练native可导出双侧接触mask | 双界面几何与条件使用 | 多为较小ligand，不等于真实warhead–linker分布 |
| 二元/片段辅助 | 独立二元结构、映射后的片段与完整化学图；独立重新对接/构象采样 | 端点保留、口袋识别、条件噪声鲁棒性 | 没有三元native时不能直接给正确DockQ或最终蛋白相对姿态 |
| 伪三元/真实PROTAC补充 | 蛋白对结构+合法linker匹配的伪数据；独立于开发集的真实PROTAC结构 | linker兼容、连接方向、较长分子覆盖 | 伪装配不是已证明的真实结合/降解阳性，需标记与降权 |

训练native中提取的端点坐标属于监督允许的信息，但**不能以完美端点条件训练全部样本，再声称能处理未知/对接错误的条件**。将各蛋白—片段对独立变换，重新生成linker与初态，避免模型直接读取目标蛋白相对布局。

第一轮建议保留原双来源训练，同时加入condition dropout（无条件、单侧、双侧）和轻度二元姿态扰动；确切比例/噪声幅度在干净内部验证选择。条件缺失时给mask，不能用全零坐标冒充真实条件。加错误口袋/错二元姿态的压力测试，看模型是否知道降低信任而不是被强行牵走。

通用三元ligand可按训练接触定义两侧集合，允许交集；只对确有可分离端点/连接路径的分子监督PROTAC分段。不得把每个小分子机械切三段当作真实PROTAC。伪数据来源需按原蛋白组与linker系列分组，避免同一蛋白对生成的伪样本跨split。

PROflow明确使用蛋白对与linker构建伪三元训练数据，说明有可行的数据扩展方向；但“几何可达”不等于“生物学上会结合”，不应用这些伪样本直接训练降解活性标签。[原始数据构造](https://arxiv.org/html/2405.06654v1)

### 建议训练顺序

1. 新condition encoder先接到C40上做小规模监督，确认condition确实被使用；缺失条件不应灾难性退化。
2. 在清洁split上训练/微调条件化精修器，保持DiffDock冻结，训练路径包含两类初始化与条件可用性变化。
3. 用新精修器重新生成一部分rank标签；先校准post-ranker，再更新pre-ranker未来价值标签。
4. 确认信息利用仍受结构表达限制后，才考虑局部side-chain柔性；刚体蛋白模型不能修复真正的apo→holo内部构象变化，仅靠延长epoch无效。

LoRA“不采用”在这里是根据当前项目决策的工程默认。本次没有重新审计另一分支的全部LoRA配对结果，不能将其写成已证明普遍无效；新条件方案也不以LoRA为依赖。

## 12. 必做实验矩阵：把信息、模型、排名拆开

### 12.1 先做两组低耦合实验

| 编号 | 改动 | 固定什么 | 回答的问题 |
| --- | --- | --- | --- |
| R0 | 几何/随机排名基线 | C40、候选、初末态 | 我们当前native-free top1到底有多差？ |
| R1 | 冻结表示+小MLP当前质量头 | 同R0，不改轨迹 | 学习post-ranking能否兑现oracle潜力？ |
| R2 | 简单未来价值pre-ranker+多样性 | C40与已训练post-ranker | 选初始化质量还是精修后价值更好？ |
| P0 | 条件资产、原子映射、独立frame审计 | 不训练模型 | 是否真的有合规可用的两侧条件？ |
| P1 | known-binary条件初始化，暂不精修 | 条件合同、固定候选预算 | 额外信息能否抬高池的结构质量上界？ |

R1和P0/P1是两个可独立推进的研究工作包，但本文没有自动启动训练或另一个agent。

### 12.2 PROTAC信息的2×2消融

| 初始化 | 无显式条件精修（C40） | 新条件化精修 |
| --- | --- | --- |
| 当前U模式候选 | 已有基准 | 只测精修阶段条件收益 |
| B模式条件候选 | 只测条件初始化及旧模型适配性 | 条件初始化+条件精修完整方案 |

同一行共享初态；跨行初态变了，比较固定候选数/总计算预算，不能声称逐候选配对。B模式下比较有/无后排序时也使用同一池。先分别比较无精修、精修、初末态可实现top1；避免拿oracle提升代替部署收益。

追加小消融：role-only、pocket-mask、binary-coordinates逐级加入；条件dropout开关；正确/扰动/错误条件；PROTAC与MGD能力保持。不要第一轮做所有组合的指数级网格。

### 12.3 之后才考虑更大胆的比较

- R3：未来价值+贪心diversity vs Set Transformer；只有内部验证top1/retention稳定获益才保留后者。
- W1：64×20独立walker vs128×4→48×16预算重分配。
- W2：在W1确有收益后才加粒子rejuvenation或集合控制，记录祖先多样性与额外开销。
- 联合训练生成器与ranker、confidence bootstrapping、reward-gradient guidance：远期，不进入第一轮主协议。

## 13. 验收指标与停止标准

### 主指标：真正交付的结构

- native-free top1 DockQ均值/中位数，≥0.23、≥0.49成功数与比例；top5中是否至少有可接受结构。
- 同时报告相应候选集oracle，区分pool upper bound、pre-selection retention与post-ranking regret。
- `post-regret = 集合oracle − 模型实际选出的DockQ`；`pre-regret = 全池最终oracle − 所选集合最终oracle`。初末态混合另列，不与末态口径混合。
- 全原子clash、配体手性/键长/环几何、双侧端点吻合及linker strain；physical-valid且DockQ达标的联合成功率。
- 分来源/条件模式/蛋白家族/linker长度分组；数值失败、条件映射失败与fallback均保留分母。

### 排序器诊断

校准曲线/Brier分数、组内Spearman、精修变差检测、初末态选择准确率、源标签捷径消融；这些是辅助指标。高AUC或rank相关性不保证top1好，不能替代主指标。

不确定性按PDB或更严格的蛋白/化学组做配对bootstrap；22例中有同PDB重复链，不能把5632个候选当独立样本计算显著性。多个方案密集调参需要独立验证，不以0.000x的均值名次定赢家。

### 我建议的门禁

1. **R1不过关，不做在线reward引导。** 如果连末态排序都不可靠，把同一模型放进采样循环只会扩大错误。
2. **P1不能抬高上界，先查条件、映射、linker采样和可达性。** 不是立刻把网络加大。
3. **条件模型只有native-perfect输入才有效，不冻结协议。** 必须证明在真实二元输入/扰动和缺失条件上可用。
4. **新split/训练来源未审计，不宣称零样本泛化。** 当前开发集成绩可以汇报，但限定用途。
5. **比简单版本多花计算却无稳定下游收益，就保留简单版本。** 所有增益门槛和允许额外耗时应在内部验证之前写入协议，而不是看完PROTAC22再决定。

## 14. 优先级与落地位置

| 优先级 | 工作 | 预计代码落点（拟议，不代表已创建） |
| --- | --- | --- |
| P0 | 数据split与条件来源审计 | `scripts/` 新只读审计入口、版本化index |
| P0 | 保存初末态坐标、rank特征、标签分离 | 扩展 `scripts/evaluate_selected_refinement.py` 的新协议，不覆盖历史产物 |
| P1 | post-ranker冻结特征基线 | 复用 `tricoopdiff/pose_ranking.py` 骨架，新训练/评估入口 |
| P1 | PROTAC condition schema与受约束初始化 | 新 `protac_conditions` 模块，接入 `expert_anchors`/图构建 |
| P2 | pre-value+贪心多样性 | 独立于pose质量头，版本化generator依赖 |
| P2 | 条件精修训练 | 新模型/checkpoint版本，保持旧C40可加载 |
| P3 | walker预算重分配/集合策略 | 新sampler协议，不改变旧确定性基线 |

最小测试包括：全局SE(3)协变、两个二元frame独立扰动、原子重排/对称映射、单侧/无条件、错误条件、fallback来源、断开native仍可推理、ranker梯度不进入冻结生成器、未来特征不可提前读取、初末态标签不进入输入、恢复运行后坐标hash一致。

## 15. 2026-09-16 双线执行合同

### Ranking：明确拆成 pre-selector 与 post-ranker

- post-ranker 的封存档案只覆盖被抽中的64个候选及其末态；当前夜间队列扩展到160个训练case，固定32/80/160三档、geometry/frozen+geometry两类特征、三个seed，共18项。8例验证被用于选epoch，只能称内部验证。
- 另建 full-pool initial 协议，对每例全部256个初态重算14维native-free几何并独立打DockQ标签；训练目标是256→64计算分配，主指标为选64后保留的全池oracle、≥0.23命中和来源组成。它不读取末态，不能冒充未来价值R2。
- 两条rank任务使用不同checkpoint和报告。第一版不做walker在线交换：在post-ranker未证明可泛化前，不允许用其分数改变轨迹。
- 后台状态分别见 `results/model_v3_ranking_night160_20260915/queue_status.json` 与 `results/model_v3_initial_selector_20260916/status.json`。只有完整网格成功才生成总表。

### PROTAC：从C EMA做严格零扰动迁移

- `ConditionedConsensusFuse` 增加蛋白口袋mask、配体双端角色、端点—本侧蛋白局部距离和条件可用性/置信度；三个条件注入通道由一个零门控制，因此未训练时无论条件是否存在都逐位复现父C模型输出。provenance只审计、不学习。第一步先学习开门，随后梯度进入条件编码器。
- 通用三元训练的native坐标只生成6 Å二值接触mask与旋转平移不变的本侧最近距离，坐标不进入条件张量或checkpoint。默认condition dropout覆盖全缺失、单侧、口袋、配体角色和binary局部几何缺失。
- checkpoint独立为`conditioned_consensus_training_v1`，绑定父checkpoint SHA256并注明从`ema_trifuse_state_dict`迁移；旧v20加载器拒绝它。训练继续使用1:1 SDPI/BiTSeed与在线C=1/M=1端点。
- 优化器把成熟父模型参数保持在`1e-4`，新condition参数使用显式记录的`1e-3`；两组都进入同一plateau scheduler。该比例仍需pilot验证，不是最终超参数。
- 验证固定为`SDPI/BiTSeed端点 × full/absent`四格。同一case四格共享flow时间，同一来源的full/absent共享coupling；BiTSeed失败回退单独计数。
- 部署只接受`protac_condition_manifest_v1`：必须核对case、canonical isomeric SMILES、原子数、片段子图映射和蛋白节点顺序；schema没有native三元坐标字段。conditioned evaluator需显式opt-in，旧checkpoint不能静默读取manifest。
- PROTAC22两端严格warhead化学映射目前没有一例双侧同时通过，因此第一轮结构评估不伪造原子映射。新增的pocket-only协议直接由各侧unbound protein及同frame的`unbound_lig{1,2}.pdb`生成6 Å Cα口袋mask；22/22例完整，明确标作provided template，不读取native三元结构、不使用跨蛋白坐标，也不声称已提供linker/warhead条件。
- 这一pocket-only部署输入仍有明确分布差：当前默认训练中role与binary通道各独立以0.1概率丢弃，双侧同时只剩pocket的组合很少。第一轮可用于检验模型能否利用口袋，但若结果为负，不能据此否定完整PROTAC条件；下一版应显式加入pocket-only可用性模式并单列验证格，而不是事后提高测试条件。

真实单例烟测完成一次训练与保存/恢复。修正配对验证后，epoch1 EMA的full/absent loss分别为SDPI `11.7704687/11.7704687`、BiTSeed端点 `1.8034111/1.8034112`；这只证明迁移与验证合同正确，不是条件模型已学到收益。8/4例pilot据此确定新condition参数使用`1e-3`而父参数保持`1e-4`。

首轮formal作业在epoch1结束前审计出“训练只见native provenance、推理却激活未训练template列”的缺陷，已主动终止且不作结果。v2从模型张量删除provenance后以相同40-epoch协议在物理GPU5/7重启：默认条件dropout对全条件缺失控制；587训练簇、65验证簇、父C40、seed和双来源端点协议相同。完成前不把中途loss差写成结论。训练结束后固定Hybrid-56/8的64个walker，比较`condition训练+pocket`、`condition训练+absent`和`absent训练+absent`；候选seed、20步精修和DockQ口径不变。只有这组结构级比较才能回答额外条件是否改善最终DockQ。

## 16. 最终观点

**先把“能生成好构象”与“能识别好构象”分别做实，再用未来价值把二者连接。**

短期押注：C40冻结精修 + 独立结构质量ranker + 初末态档案；同时补齐与DeepTernary可对齐的PROTAC二元条件，验证其对候选上界的提升。

中期核心贡献可以是：**条件化双端初始化、兼容双来源的条件精修，以及面向最终结构质量的预算分配与排序。** 这比“一个更复杂的rank网络”更完整，也比“仅增加初始化DockQ”更有说服力。

不建议现在押注：把冻结judge当confidence、只用初态DockQ训练pre-ranker、只按簇大小选walker、确定性轨迹复制换seed、对未校准rank分数做强引导、用目标三元布局提供两侧二元条件，或声称ranker可以突破固定池oracle。

**一条具体的起跑线：先做R1和P0/P1；R1证明我们会选，P1证明我们拿到了更好的几何信息；再做R2和条件化精修，最后决定跨walker协作值不值得。**
