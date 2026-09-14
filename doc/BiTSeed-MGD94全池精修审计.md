# BiTSeed MGD94 全池精修审计

更新时间：2026-09-14。本文汇总同一 `192 BiTSeed + 64 SDPI` 候选池中全部 256 个
候选经过当前 TriCoopDiff 前后的 exact DockQ。MGD94 是 development benchmark；所有
best/retention 均为 native-reading 诊断，不是部署 top-1。

## 完整性与运行契约

MGD94 共 94/94 例完成，每例 256/256 条 trajectory，总计 24,064 条，全部 `valid=True`，
失败为零。来源严格为 18,048 条 BiTSeed 和 6,016 条 SDPI。六个分片的 checkpoint、候选
协议、seed、20-step refinement 和 full-pool contract 一致。累计 case runtime 为 30.69
小时，双 GPU、每卡两个 worker 的墙钟约 7 小时 50 分。

## 三张榜单

表中为每例最佳 DockQ 的 MGD94 平均值；`Union` 是逐候选保留 initial/refined 较优者的
oracle 上界。

| 选择方法 | Initial | Final | Union | Final 改善/下降 case |
|---|---:|---:|---:|---:|
| SDPI-64 | 0.2077 | 0.2515 | 0.2602 | 70 / 24 |
| BiTSeed-only-64 | 0.2380 | 0.2367 | 0.2636 | 52 / 42 |
| Geometry-64 | 0.2376 | 0.2665 | 0.2862 | 68 / 26 |
| Diversity-64 | 0.2414 | **0.2668** | **0.2887** | 67 / 27 |
| Hybrid-48/16 | **0.2445** | 0.2624 | 0.2867 | 65 / 29 |
| Hybrid-56/8 | 0.2406 | 0.2472 | 0.2740 | 58 / 36 |
| Full-pool-256 | 0.3044 | 0.3097 | 0.3412 | 52 / 42 |

Geometry-64 实际累计选择 3,815 BiTSeed + 2,201 SDPI，平均约 40.6/23.4；Diversity-64
为 3,704 + 2,312，平均约 39.4/24.6。当前 checkpoint 更适配 SDPI，因此它们比固定
48/16 保留更多 SDPI，并在 Final 榜略优。该比例不能直接外推到联合训练后的新模型。

## TriCoopDiff 与 selector 结论

- SDPI 每例最佳从 0.2077 升至 0.2515，候选级 5,214/6,016（86.7%）改善，证明当前
  TriCoopDiff 确实具有 SDPI refinement 能力。
- BiTSeed 候选级平均从 0.03979 升至 0.04479，13,578/18,048（75.2%）改善；但每例
  BiTSeed 最佳从 0.2726 降至 0.2631。模型改善多数弱候选，却不能稳定保护强初态。
- 初始化 oracle 与 Final oracle 只有 25/94 是同一 candidate。Diversity-64 的 Final
  oracle retention 为 83.5%，只精确保留 31/94 个全池 Final 赢家；selector 不能只用
  “初始化看起来最好”代替“refinement 后最好”。
- 在当前 checkpoint 下淘汰 56/8；48/16 可作固定配额基线，但 Geometry/Diversity 是
  更强的无固定配额对照。

## `acceptable` gate 审计

当前 gate 同时要求无蛋白 CA<3 Å、两侧 CA–ligand 最近距离均为 1.5–8 Å、蛋白最近
CA 距离为 3–15 Å。存档标签已逐候选重新计算并 100% 一致；仅 2,306/24,064（9.58%）
通过，BiTSeed 为 10.64%，SDPI 为 6.42%。非互斥拒绝原因如下：

| 原因 | 候选数 | 占全池 |
|---|---:|---:|
| 蛋白–蛋白 CA<3 Å | 14,018 | 58.25% |
| POI–ligand >8 Å | 7,823 | 32.51% |
| 蛋白间距 >15 Å | 4,543 | 18.88% |
| POI–ligand <1.5 Å | 2,733 | 11.36% |
| E3–ligand <1.5 Å | 2,063 | 8.57% |
| E3–ligand >8 Å | 1,864 | 7.75% |

对 native MGD94 运行同一 gate，92/94 通过；全部 native 都通过碰撞条件，只有 `3UJL`
和 `5MXO` 因 POI–ligand 分别为 8.09 Å、11.51 Å 而失败。因此 `<3 Å` 蛋白碰撞与
`<1.5 Å` ligand 穿插可继续作为 hard safety gate；`>8 Å` 接触不足应视为软兼容性
证据，不能作为普适 hard reject。只保留三条安全下界时，候选保留 9,008/24,064
（37.43%），native 为 94/94；此处仅为离线诊断，生产 selector 尚未改变。

Geometry-64 在所有 94 例中均包含 acceptable 候选，并保留 full-pool 的全部
acceptable-constrained Initial/Final/Union 最佳值（均值 0.1698/0.1976/0.2062）。这说明
64 个预算足以覆盖当前 gate 下的可行候选，但不证明该 gate 或最终 top-1 已经最优。

## 当前决策

1. 暂不改变正式阈值；先在 internal validation 固定 hard/soft 拆分，禁止用 MGD94
   native DockQ 调参。
2. 默认候选池继续使用 192/64；当前模型比较时保留 Geometry-64、Diversity-64 和
   Hybrid-48/16 三个主要 arm。
3. 联合初始化训练在独立 `exp/dual-init` 分支进行；新模型完成后必须重新比较来源配额，
   不能沿用当前模型偏好 SDPI 的结论。
4. 后续最终几何审计必须重新计算 refined structure 的碰撞/接触；当前 JSON 中的
   `acceptable` 只描述初始化，不能代表 Final 结构物理有效。

## 复现

```bash
PYTHONPATH=. /home/wl/micromamba/envs/tricoopdiff/bin/python \
  scripts/summarize_full_pool_refinement.py \
  --results-root results/mgd_full_pool_refinement_20260913 \
  --index results/reproduction_20260911/index_current/mgd94_full_v1.json \
  --out results/mgd_full_pool_refinement_20260913/summary_v1.json
```

机器可读结果为 `results/mgd_full_pool_refinement_20260913/summary_v1.json`。
