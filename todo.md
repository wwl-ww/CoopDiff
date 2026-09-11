# CoopDiff 接手清单

更新时间：2026-09-11。本文是新负责人唯一需要先读的执行入口；历史解释见
[`doc/低DockQ根因排查交接.md`](doc/低DockQ根因排查交接.md)，文档状态见
[`doc/README.md`](doc/README.md)。实验事实以 `results/` 中完整报告为准。

## 当前状态

- 当前分支：`model-v2`；交接前远端同步点为 merge commit `3753c4e`。
- 工作区在本文修改前干净；本项目没有训练或评估进程，GPU 2–5 均已释放。
- 已结束分支：`investigate/model-v2-followup`。不要继续该分支的 loss sweep。
- 主目标是 Molecular Glue（MGD）。PROTAC 后续会加入 warhead/口袋信息，不能用
  PROTAC 的容易提升替代 MGD 的真实困难。
- `data/`、`checkpoints/`、`results/`、`logs/` 不进 Git；换机器接手必须另行传输。

## 一句话结论

低 DockQ 的核心是从宽 SDPI prior 搜索正确的联合位置与朝向，而不是一个隐藏实现
错误或 `0.000x` 级 loss 差异。当前最有效的改动是双 frozen DiffDock 专家生成完整
SE(3) geometry-anchor，再由 CoopDiff 精修；下一阶段应把初始化条件做到最好，并开发
严格无 native 的在线候选筛选，而不是继续长训旧目标。

## 已完成且可引用的结果

### MGD94 覆盖上界

- teacher epoch 2 EMA，x32：oracle mean/median `0.21922/0.17427`；DockQ
  `>=0.23/0.49/0.80` 为 `33/94、7/94、1/94`；walker mean 仅 `0.05397`。
- 第二套 x32 的 acceptable 为 34/94；两套独立 x64 为 38/94 与 37/94。
- 两套 x64 池化 acceptable 为 `43/94=45.7%`。两套 x32 + 两套 x64 的统一
  192-walker oracle mean/median 为 `0.30152/0.23137`，acceptable/medium/high
  为 `48/94、17/94、1/94`。
- 以上全是 native 事后选择的 oracle，不是 model-ranked top-1。

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
- `tricoopdiff/pose_anchors.py`：统一 proper rigid fit 和 joint surface SE(3) 原语。
- `scripts/evaluation_runner.py`、`scripts/evaluate_index.py`：支持
  `--init-mode expert_anchor`、`--binary-anchor-steps`、
  `--anchor-proposal-attempts`、`--anchor-selection geometry|consensus`，并记录 provenance。
- 当前 initializer 是实验验证版：每个 walker 都重新生成 proposal，尚未实现下一节的
  case 内确定性在线候选集合和学习式选择器。

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

只有存在 `batch_status.json` 且 `complete=true` 的目录可进入正式汇总。以下任务被主动
停止，只能用于调试：`mgd94_x32_rankfeatures/` 为 74/94，
`anchor_geometry_a8_20cases_seed32000/` 为 6/20；二者没有完成状态。

## 下一负责人第一项任务

建议从 `model-v2` 新建一个综合初始化分支，不创建 worktree。先实现“确定性在线候选
生成”，不要建立或加载可人工编辑的 pose bank：

1. 正式推理只接收原始蛋白、ligand、固定 seed 和 frozen checkpoint；候选生成函数
   不得接收 native 路径，正式评估禁用 pose cache。
2. 每侧在线生成 M 个二元 pose，以约 `2M` 次专家推理组成 `M²` 个 ternary anchor。
   候选配额、seed 推导、twist、geometry/consensus/broad 比例必须在测试前冻结。
3. 先在 PDB-disjoint internal split 比较当前 64 候选与扩大后的 256 候选 oracle；
   native 只能由候选封存后的独立评估进程读取。最终 CoopDiff top-K 预算保持不变。
4. 若候选 oracle 没有明显提升，停止做 ranker 并增加新的生成模式；若提升，再构建
   native-free pre-ranker，并按姿态簇选择 diversity-aware top-K 进入 CoopDiff。
5. pre-ranker 特征至少包含专家分数/轨迹稳定性、候选共识、三类接触距离分布、穿插
   与排斥、界面互补、双侧平衡和 conformer strain。只用 internal 32/8/10
   PDB-disjoint split 训练，MGD94 不参与训练或超参数选择。
6. 精修后再训练 post-ranker 输出 top-1。门禁顺序固定为：候选 oracle 上界 →
   pre-ranker oracle retention → 固定预算 coverage → post-ranker top-1。

验收门禁：短面板 paired mean DockQ 至少 `+0.01`、median > 0、没有稳定坏亚组，
并改善 DockQ `>=0.23` 或 top-k 命中；`0.000x`、单 case 拉动和只改善训练 loss
均不算进展。必须同时报告二元专家调用数、原始/过滤后候选数、CoopDiff 精修数和 GPU
时间，避免用隐藏计算预算制造优势。

## 不可破坏的协议

- MGD94 只作最终评估；oracle-best、native-reading oracle、ranked top-1 分开报告。
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
