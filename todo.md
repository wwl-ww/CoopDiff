# CoopDiff 接手清单

更新时间：2026-09-14。本文是新负责人唯一需要先读的执行入口；历史解释见
[`doc/低DockQ根因排查交接.md`](doc/低DockQ根因排查交接.md)，文档状态见
[`doc/README.md`](doc/README.md)。实验事实以 `results/` 中完整报告为准。

## 当前状态

- 当前开发分支：`feat/deterministic-online-anchors`，从 `model-v2` 的 `0d7bd67`
  创建；交接前远端同步点为 merge commit `3753c4e`。
- `192 BiTSeed + 64 SDPI` 固定池、per-conformer 缺口补位、统一 selector 和
  full-pool refinement 已实现；MGD94 为 94/94、24,064/24,064 trajectory、零失败。
  正式结论已合并进 [`BiTSeed-Hybrid 在线初始化规格`](doc/BiTSeed-Hybrid在线初始化规格.md)。
- 已结束分支：`investigate/model-v2-followup`。不要继续该分支的 loss sweep。
- 主目标是 Molecular Glue（MGD）。PROTAC 后续会加入 warhead/口袋信息，不能用
  PROTAC 的容易提升替代 MGD 的真实困难。
- `data/`、`checkpoints/`、`results/`、`logs/` 不进 Git；换机器接手必须另行传输。

## 一句话结论

低 DockQ 的核心是从宽 SDPI prior 搜索正确的联合位置与朝向，而不是一个隐藏实现
错误或 `0.000x` 级 loss 差异。当前路线统一为“广泛生成 → 智能筛选 → 昂贵精修 →
最终排序”：**BiTSeed**（Conformer-Diverse Binary-to-Ternary Basin Seeding）用双
frozen DiffDock 专家负责搜索完整 SE(3) basin，native-free selector
选择值得投入计算的初始化，再由 CoopDiff 精修；不继续长训旧目标。

## 已完成且可引用的结果

### MGD94 覆盖上界

- teacher epoch 2 EMA，x32：oracle mean/median `0.21922/0.17427`；DockQ
  `>=0.23/0.49/0.80` 为 `33/94、7/94、1/94`；walker mean 仅 `0.05397`。
- 第二套 x32 的 acceptable 为 34/94；两套独立 x64 为 38/94 与 37/94。
- 两套 x64 池化 acceptable 为 `43/94=45.7%`。两套 x32 + 两套 x64 的统一
  192-walker oracle mean/median 为 `0.30152/0.23137`，acceptable/medium/high
  为 `48/94、17/94、1/94`。
- 以上全是 native 事后选择的 oracle，不是 model-ranked top-1。
- 新的固定256 proposal oracle 中，纯 SDPI-256 为 mean/median `0.25722/0.20901`，
  DockQ `>=0.23/0.49/0.80` 为 `39/10/0`。B1 (`C4M8`) 加 case-level SDPI fallback
  为 `0.31310/0.30885`、`61/12/0`；C1 (`C16M4`) 为 `0.31488/0.30885`、
  `64/12/0`，但 binary 成本是 B1 两倍。

### 根因与生成器门禁

- 化学恢复、缓存、图方向、checkpoint 加载和完整原子 DockQ 重建已通过检查；五例
  native 重建均为 DockQ 1.0。
- 五例 × 8 seeds：baseline mean/median `0.04058/0.02613`；translation-only
  oracle `0.06021/0.04600`；rotation-only `0.06617/0.03108`；读取 native 的
  joint-rigid oracle 为 40/40 DockQ 1.0。位置或朝向单独修复都不充分。
- 20-case `8×8` geometry-anchor 对同 seed broad x8：oracle mean/median
  `0.06717/0.06348 → 0.16487/0.12156`，acceptable `0/20 → 5/20`，case
  15 胜/3 平/2 负，平均提升 `+0.09770`，anchor 接受 92/160。
- 同面板 consensus-anchor：oracle mean/median `0.12956/0.08772`，acceptable
  5/20，case 15 胜/2 平/3 负。它有效但弱于 geometry，适合做筛选特征或补充分支。
- 该 20-case 是索引前 20 条，其中 15 个为 A8S；它证明 A8S-heavy 子群上的效果，
  不能写成 MGD94 全体泛化结论。

### 已否决或后置

- 20-case DiffDock conformer 低于 ETKDG：oracle mean/median
  `0.10023/0.07406` 对 `0.12059/0.08985`，acceptable 0/20 对 3/20；池化无新增
  acceptable。保持 `etkdgv3_random` 默认。
- windowed、soft-time、direction、agreement、joint-translation、prior-replay、
  增大 gradient accumulation 和单纯增加积分步数均未过稳定 DockQ/命中门禁。
- teacher 与 base 的 MGD94 差异近似噪声；标准 Reflow 只能后续做少步蒸馏，不能
  解决 prior-endpoint 配对。
- 首版 128 维 latent listwise ranker 在独立 test 只保留 44.1% oracle，在 MGD
  五例只保留 15.2%；不要原样重训。

## 当前实现

- `tricoopdiff/expert_anchors.py`：`DualExpertAnchorInitializer`。同一 ligand conformer
  分别进行 E3–ligand 与 POI–ligand rigid-only DiffDock，Cartesian 组合后用 proper
  Kabsch 对齐共享 ligand；做蛋白碰撞、双侧接触和 POI outward relaxation，无可接受
  候选时显式回退 broad SDPI。
- 同文件的 `generate_candidates()` 是 BiTSeed 当前在线候选入口：支持多 conformer、
  conformer 内配对、确定性均衡 pair 子采样，以及持续补采直到每侧收集满 `M` 个有效
  binary poses。`binary_retry_limit=4` 表示每个 conformer/side 最多 `4M` 次 draws。
- `tricoopdiff/pose_anchors.py`：统一 proper rigid fit 和 joint surface SE(3) 原语。
- `scripts/evaluation_runner.py`、`scripts/evaluate_index.py`：支持
  `--init-mode expert_anchor`、`--binary-anchor-steps`、
  `--anchor-proposal-attempts`、`--anchor-selection geometry|consensus`，并记录 provenance。
- 当前 initializer 是实验验证版：每个 walker 都重新生成 proposal，尚未实现下一节的
  case 内确定性在线临时候选集、集合感知选择器和 32/64 walker 一一映射。
- binary expert 整例失败时回退 `sdpi_v1`、纯 SDPI-256 proposal 生成及其 CLI 已实现；
  部分 conformer 失败时保留成功 anchors 并只补齐缺口的逻辑尚未实现。
- `scripts/evaluate_selected_refinement.py` 统一评估候选 Initial/Final/Union；native 指标只
  用于事后审计，不能当作部署 top-1。

## 关键本地产物

- checkpoint：`results/paired_pilots/followup_seed28102_teacher_acc8_64cases_e16/epoch_002.pt`
- frozen expert：`checkpoints/score_model/best_ema_inference_epoch_model.pt`
- MGD94 index/cache：`results/mgd_gate/full94_v1/mgd94_full_v1.json`、
  `results/mgd_gate/full94_v1/cache_ccd_v7/`
- 完整 x32：`results/official_mgd94_teacher28102_ema_32walkers/cases/`、
  `results/mgd_followup/mgd94_teacher_x32_seed32000/`
- 完整 x64：`results/mgd_followup/mgd94_x64_seed31000/`、
  `results/mgd_followup/mgd94_x64_seed41000/`
- anchor 20-case：`results/mgd_followup/anchor_geometry_a8_20cases_seed31000/`、
  `results/mgd_followup/anchor_consensus_a8_20cases_seed31000/`
- conformer 20-case：`results/mgd_followup/conformer20_seed55000/`
- BiTSeed 配置与协议：`doc/BiTSeed-Hybrid在线初始化规格.md`
- BiTSeed v4：`results/proposal_oracle_ablation_20260912/`；正式目录后缀为 `_v4`，
  `recovery_smoke_v4` 仅作补采逻辑 smoke，所有 `_v3` 目录均已停止。

只有存在 `batch_status.json` 且 `complete=true` 的目录可进入正式汇总。以下任务被主动
停止，只能用于调试：`mgd94_x32_rankfeatures/` 为 74/94，
`anchor_geometry_a8_20cases_seed32000/` 为 6/20；二者没有完成状态。

## 当前统一方案：BiTSeed → select → CoopDiff → rank

### 初始化协议冻结任务（2026-09-13）

- [x] 实现固定256候选的 Hybrid-192/64、`2M`短补采、连续3次失败截断、部分pose池保留和
  按缺口SDPI补位。
- [x] 实现固定E3 frame的 ligand/POI RMSD聚类、x32的24/8配额和x64的48/16配额。
- [x] 完成PROTAC22的SDPI-256、224/32、192/64、128/128 exact proposal oracle；四组
  均22/22零失败，候选比例预冻结为192/64。
- [x] 完成PROTAC22的192/64 selector retention@32/64 exact审计。
- [x] 完成MGD94统一256池的全部 Initial/Final/Union 与 selector retention 审计。
- [x] 完成proposal `acceptable` 拒绝原因和native 94例 sanity check；生产阈值尚未改变。
- [ ] 在internal validation固定hard safety与soft compatibility拆分，并重算refined几何。
- [ ] 对冻结配置做候选坐标hash复跑，再锁定协议版本。

当前分支实现确定性在线多假设搜索，不建立或加载可人工编辑、可跨 case 复用的 pose
bank。正式推理从原始输入一条命令运行到底；中间候选仅在内存中存在，允许保存数量、
特征摘要和坐标 hash 作审计，默认不落盘全部结构。

其中在线搜索模块正式命名为 **BiTSeed**；与精修模型组成完整系统时写作
**BiTSeed-CoopDiff**。名称不覆盖 selector/ranker，oracle 也不是 BiTSeed 的部署组件。

### 1. 在线联合生成

1. 正式生成器只接收原始 E3、POI、完整 ligand、固定 seed、frozen DiffDock checkpoint
   和冻结配置；函数签名不得包含 native、DockQ 或人工候选路径，正式评估禁用 pose cache。
2. 对每个 `etkdgv3_random` ligand conformer，两侧各生成 `M` 个完整 ligand binary
   poses，仅在同一 conformer 内配对。当前固定256候选消融比较 `C1M16`、`C4M8`、
  `C16M4`，compute-matched 消融比较 `C1M64`、`C4M16`、`C16M4`；正式映射和 seed
   协议以 `doc/BiTSeed-Hybrid在线初始化规格.md` 为准。
3. `q(E,P|L)=q_E(E|L)q_P(P|L)` 仅是扩大覆盖的 proposal 分布，不声称真实 ternary
   条件分布可因子分解；后续 ternary compatibility 专门修正 binary-to-ternary gap。
4. 候选顺序、seed 派生、expert 调用数、outward relaxation、twist 与 broad 比例必须
   在 benchmark 前冻结。若 anchor 数不足，按固定规则回退 broad SDPI，禁止人工补选。

### 2. 物理过滤、聚类与 pre-ranker

筛选顺序为 hard filter → cluster/context → set-aware pre-ranker → diversity-constrained
Top-W。hard filter 只删除明确不可能的姿态，包括非有限坐标、严重蛋白穿插、ligand
完全脱离任一侧和刚体拟合失败；边界可疑项保留为软特征，避免过滤掉新 basin。

聚类先建立 basin 结构，但不能仅凭几何 medoid 过早把每簇压成一个候选。为每个候选
记录簇大小、局部密度、簇内/medoid 距离，再由 pre-ranker 判断簇内质量；最终用每簇
配额或确定性 MMR 防止一个高分错误模式占满 walker。pre-ranker 输入分为：

- **binary evidence**：两侧 expert 分数、逐步位移和轨迹稳定性，但不把 binary top-1
  当作 ternary 真值；
- **ternary compatibility**：E3–POI、E3–ligand、POI–ligand 多阈值接触/距离分位数、
  穿插与短程排斥、双侧平衡及界面互补；
- **candidate-set context**：共识密度、姿态簇、离群程度和与已选 basin 的距离。

模型优先使用小型 Set Transformer/DeepSets 或等价集合感知结构，不原样重训旧 128 维
latent MLP。当前单 conformer 下 strain 可能不具区分力；PROTAC warhead 距离/linker
strain 只有在可靠端点 mask 接入后才能使用，不能在当前 MGD 完整 ligand 模式中伪造。

### 3. 固定预算精修与最终排序

Top-W 必须恰好映射到 `W=32` 或 `W=64` 条 CoopDiff walker，而不是将正常预算缩成
8 条。初始探索配额建议为 x32 的 24 anchor + 8 broad、x64 的 48 anchor + 16 broad；
具体比例只在 internal split 消融一次并冻结。pre-ranker 决定哪些 basin 值得投入计算，
post-ranker 则在精修后的 32/64 个结构中输出部署用 top-1、top-k 和可信度，两者不合并。

### 4. 训练、门禁与报告

两个 DiffDock experts 始终冻结。当前 BiTSeed 生成模式直接在 MGD94 上做已声明的
development oracle ablation；该 benchmark 过去已反复用于方案比较，不再声称是未触碰的
zero-shot holdout。native 只由评分阶段读取，不进入生成器。后续 selector 仍只能用
internal 32/8/10 PDB-disjoint split 训练，不得用 MGD94 DockQ 作监督标签。门禁顺序固定为：

1. hard-filter 好姿态召回率与候选 oracle 上界；
2. pre-ranker oracle retention@32/64；
3. 固定预算 CoopDiff coverage；
4. post-ranker ranked top-1/top-k；
5. 在明确标注为 development benchmark 的完整 MGD94 上报告结果，并另寻未参与设计的
   外部数据作最终独立验证。

验收仍要求短面板 paired mean DockQ 至少 `+0.01`、median > 0、没有稳定坏亚组，并改善
DockQ `>=0.23` 或 top-k 命中；`0.000x`、单 case 拉动和只改善训练 loss 均不算进展。
同时报告二元专家调用数、原始/过滤/聚类后候选数、CoopDiff walker 数和 GPU 时间，避免
用隐藏计算预算制造优势。论文贡献定位为 native-free 的 ternary
search–select–refine framework，而不是“更大的 diffusion model”。

## 不可破坏的协议

- MGD94 已是开发 benchmark，不得再称独立 zero-shot 测试；oracle-best、native-reading
  oracle、ranked top-1 仍必须分开报告。
- MGD94 的 94 例分母固定；`3IQV_A_P_FSC` 使用显式 `pdb_inferred_fallback`。
- 实验 factorial-v1 checkpoint 必须显式 `--allow-experimental-checkpoint`，继续校验
  frozen expert、无 LoRA、expert identity 与 EMA。
- 旧 checkpoint 的 torus 表不可恢复；当前统一 deterministic quadrature，不能声称
  是旧轨迹逐位复放。
- GPU 物理卡 2–5 可用；不同卡运行不同的完整任务，不把同一测试的数据拆到多卡。
- 不创建 worktree；保护用户改动；减少 Git 次数，使用短 imperative Conventional
  Commit subject。所有命令从仓库根目录运行。

## 快速验证

```bash
/home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_model_build.py
PYTHONPATH=. /home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_joint_endpoint_experiment.py
/home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_module_boundaries.py
git diff --check
```
