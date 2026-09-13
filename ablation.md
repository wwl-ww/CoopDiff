# BiTSeed Proposal Oracle Ablation

## 名称与范围

**BiTSeed** 的正式展开为 **Conformer-Diverse Binary-to-Ternary Basin Seeding**。
它使用两个 frozen binary docking experts，在共享 ligand conformer 条件下组合 ternary
basin seeds；与 CoopDiff 联用时称为 **BiTSeed-CoopDiff**。BiTSeed 只指在线候选初始化，
不包含 native oracle、后续 selector、CoopDiff refinement 或 post-ranker。

## 实验目的

本实验测量初始化候选的 native-reading oracle 上界，不训练 ranker、不执行候选部署
筛选，也不运行 CoopDiff refinement。第一组固定256个完整 Cartesian proposals，比较
不同 conformer/pose 分配的质量–成本；第二组同时固定128条 binary trajectories 和256个
实际评分 proposals，比较相同 expert 预算下的分配。`C1` 与 `C2` 配置完全相同，共用
一次结果，因此六个实验标签对应五个独立任务。

## 数据与输入

- 数据集：完整 MGD94，共 94 cases，bound protein conformations。
- 索引：`results/reproduction_20260911/index_current/mgd94_full_v1.json`。
- 索引 SHA256：`f5b3196378e462b00d64b5a6743286b068e4920ac6f1cb738ddb0780327dd289`。
- ligand chemistry：93 例使用 CCD-restored SDF；`3IQV_A_P_FSC` 使用显式
  `pdb_inferred_fallback`。
- `protein1` 作为固定 E3，`protein2` 作为移动 POI；native complex 仅在全部候选生成
  完成后用于 DockQ 评分，不进入 DiffDock expert 或候选生成。

## 对照组与计算预算

| 方案 | conformers `C` | 每侧 poses `M` | binary trajectories `2CM` | 潜在组合 `CM²` | 评分候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BiTSeed-A1 (`C1M16`) | 1 | 16 | 32 | 256 | 256 |
| BiTSeed-B1 (`C4M8`) | 4 | 8 | 64 | 256 | 256 |
| BiTSeed-C1/C2 (`C16M4`) | 16 | 4 | 128 | 256 | 256 |
| BiTSeed-A2 (`C1M64`) | 1 | 64 | 128 | 4,096 | 256 |
| BiTSeed-B2 (`C4M16`) | 4 | 16 | 128 | 1,024 | 256 |

每条 binary trajectory 使用20点推理 schedule，实际执行19次 frozen DiffDock expert
更新。A1、B1 每例无失败时的 expert forward 数分别为608、1,216；C1/C2、A2、B2
均为2,432。非有限轨迹产生的补采调用单独报告。所有成功 case 均实际构造并评分256个
候选。

## Ligand Conformer 与 Binary Pose 生成

- 基础 seed 固定为 `31000`；conformer `c` 的 seed 为
  `(31000 + c × 100000007) mod (2³¹−1)`。
- conformer policy 为 `ETKDGv3 + useRandomCoords=true`。`no_torsion=false`，初始化时对
  可旋转键施加独立均匀 torsion；当前实验不做 conformer RMSD 去重。
- 同一 conformer 的 E3-side 和 POI-side expert 必须使用同一个 ligand 内部构象；禁止
  跨 conformer 配对。
- 二元 DiffDock 只更新 ligand 刚体位姿，所有 expert torsion update 强制为零，从而在
  一条二元轨迹中保持该 conformer 的内部构象。
- 每侧第0个 pose 使用该 conformer 的初始 SDPI binary placement，其余 pose 使用独立
  receptor-adaptive translation 和 Haar rotation。
- 每个 conformer/side 按固定 seed 序列持续抽取轨迹，成功结果依次填入 bank，直到收集
  满 `M` 个有效 poses；单个失败 draw 不再绑定并终止某个固定 slot。每侧总 draw 上限为
  `4M`。报告记录 `binary_draws` 和 `binary_retries`；达到全局上限仍不足 `M` 才将该
  case 标记失败。该恢复逻辑属于 v4 协议。

## 256个 Ternary Proposals 的构造

候选 ID 固定为 `(conformer_id, e3_pose_id, poi_pose_id)`，只允许同一 conformer 内
配对，不按 native 质量挑选组合。A1/B1/C1保留每个 conformer 的完整 `M×M` Cartesian
product，每个 conformer 分别产生256、64、16个候选。A2每个 conformer 从4,096对中
均衡取256对；B2每个 conformer 从256对中均衡取64对；C2保留全部组合。A2/B2/C2中
每条 E3-side 和 POI-side marginal pose 都恰好参与4个最终候选。均衡子集完全由固定
seed 决定，并在读取 native 前生成。

对每个 binary pair，先用 proper rigid fit 将 POI-side ligand 对齐到 E3-side ligand，
并将同一刚体变换施加到 POI。随后沿 E3→POI 质心方向测试13个 outward offsets：
`0, 0.5, …, 6.0 Å`，按以下非 native 字典序 penalty 选出一个 proposal：

1. acceptable 优先；
2. protein Cα pairs `<3 Å` 的数量越少越好；
3. `|d(E3,L)−3.5| + |d(POI,L)−3.5| + |d(E3,POI)−5.0|` 越小越好；
4. 相同时选择更小的 outward offset。

acceptable 定义为：没有 Cα pair `<3 Å`，两侧 protein–ligand 最近距离均在
`1.5–8 Å`，protein–protein 最近距离在 `3–15 Å`。该标记只作诊断；256个候选无论
acceptable 与否都进入 oracle 评分，不执行 fallback 或 hard rejection。

## Oracle 评分与汇总

每个候选均通过 `DeepTernary/DockQ/dockq_util.py::cal_dockq` 计算 full-atom exact
DockQ、Fnat、LRMSD 和 iRMSD。每例 primary result 是256个候选中的最高 exact DockQ；
这是 native-reading oracle，不代表可部署 top-1。另行保存 DeepTernary approximate
DockQ 作为诊断。

全量汇总必须使用相同94例进行 paired comparison，并报告：

- oracle-best DockQ 的 mean、median；
- DockQ `>=0.23` 和 `>=0.49` 的 case 命中数；
- A1/B1/C1 和 A2/B2/C2 的逐 case win/tie/loss 与 paired delta；
- acceptable 候选数、binary retry 数、失败 case；
- 每例 candidate generation、exact scoring 和 total wall time。

## 模型、软件与运行命令

- Frozen expert：`checkpoints/score_model/best_ema_inference_epoch_model.pt`。
- Expert SHA256：`64afbdc97f6c700df0309162eb533ee6f2754c71412aca7a3a534a7239667260`。
- Config：`checkpoints/score_model/model_parameters.yml`，SHA256
  `58c3d2a5da4194fe4ba968363a3f16ccb3c11f2c0898c9d27e62f4b06893f943`。
- 不加载 TriFuse/CoopDiff checkpoint。GPU为 RTX 5090；任务使用 `nohup setsid`
  脱离终端，并按相同 conformer 数绑定到同一物理卡串行执行。
- Python：`/home/wl/micromamba/envs/tricoopdiff/bin/python`（3.10.20）；PyTorch
  2.13.0+cu130；RDKit 2022.09.5；NVIDIA driver 580.142。
- 设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，ESM2-650M 从本地 snapshot
  `hf_cache/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c`
  加载。所有命令从仓库根目录执行。

三组命令的公共入口为：

```bash
python scripts/evaluate_proposal_oracle.py \
  --index results/reproduction_20260911/index_current/mgd94_full_v1.json \
  --conformers <1|4|16> --poses-per-side <16|64|8|4> \
  --candidate-budget 256 --binary-steps 20 --binary-retry-limit 4 \
  --seed 31000 --device cuda --out-dir <arm_output>
```

后台排程如下：

| 物理 GPU | 串行任务 | 预计总时间 |
| --- | --- | ---: |
| 2 | A1（约3.2h）→ A2（约6.2h） | 约9.4h |
| 3 | B1（约4.2h）→ B2（约6.2h） | 约10.4h |
| 4 | C1/C2 共用（约6.2h） | 约6.2h |

A2/B2 watcher 以“前项进程结束且已写出 `batch_status.json`”作为接棒条件，不要求前项
`complete=true`。因此少数补采不足的 case 会保留在前项失败清单中，但不会阻止后项
扫描；若前项异常退出且没有状态文件，watcher 不启动后项。

正式输出均位于 `results/proposal_oracle_ablation_20260912/`：A1/B1/C1 使用
`a1_c1_m16_v4/`、`b1_c4_m8_v4/`、`c1_c16_m4_v4/`，A2/B2 使用
`a2_c1_m64_v4/`、`b2_c4_m16_v4/`。日志为同级 `logs/*_v4.log`。无失败时整体墙钟
约10.4小时；实际时间随补采数增加。只有每个独立任务的 `batch_status.json` 均为
`complete=true` 且94例齐全时，才能进入正式汇总。旧 v3 目录使用固定-slot 重试语义，
已停止且不得与 v4 混合汇总。

## 完成结果与 Hybrid 后续

v4 对94例完成扫描，但 frozen DiffDock 的非有限输出使 A1/A2 只有59例生成成功，
B1/B2/C1只有57例成功。随后以256个确定性 `sdpi_v1` proposals 补齐缺失 case，并另行
完成全94例 SDPI-256 直接对照。完整 exact oracle 如下；`fallback` 只由 binary expert
是否成功触发，不读取 native。

| 方法 | mean | median | DockQ≥0.23 | DockQ≥0.49 |
|---|---:|---:|---:|---:|
| SDPI-256 | 0.25722 | 0.20901 | 39/94 | 10/94 |
| A1 + fallback | 0.29622 | 0.25258 | 56/94 | 12/94 |
| B1 + fallback | 0.31310 | 0.30885 | 61/94 | 12/94 |
| C1/C2 + fallback | 0.31488 | 0.30885 | 64/94 | 12/94 |
| A2 + fallback | 0.30156 | 0.26388 | 59/94 | 10/94 |
| B2 + fallback | 0.31230 | 0.28991 | 63/94 | 14/94 |

在 DiffDock 成功的57例上，B1 与同例 SDPI mean 分别为 `0.26969/0.17753`，但 SDPI
仍赢16例；C1相对B1仅增加 `0.00294`，不抵偿两倍 binary 轨迹。因此下一轮不做 LoRA，
采用 B1 作为结构化分支，并测试固定256预算的在线混合：`224/32`、`192/64`、
`128/128`（BiTSeed/SDPI）。待审计默认值为 `192/64`，完整生成、fallback、聚类和
walker 配额见 [`doc/BiTSeed-Hybrid在线初始化规格.md`](doc/BiTSeed-Hybrid在线初始化规格.md)。
