# DeepTernary 基线与 DockQ 对齐

本文合并论文分析、权重调查和本地审计结论，是 TriCoopDiff 与 DeepTernary 比较的唯一口径说明。

## 可复现基线

最终审计使用 `epoch_1000.pth`、`lig_kpt_mode_val='mask_predict'`、`FIX_TWO_ENDS=False`，运行 22 个 PROTAC case、每例 40 个 seed。

| 指标 | 本地审计 | 官方结果 |
|---|---:|---:|
| mean best DockQ | **0.6588** | 0.7062 |
| mean prediction DockQ | 0.3952 | 0.4287 |
| Acceptable+（DockQ ≥ 0.23） | 62.05% | — |

按 linker 可旋转键分层的 mean best DockQ 为：short（≤8）0.6800、mid（9–15）0.7173、long（>15）0.5463。剩余约 0.047 的差异与代码及数据版本演进有关；模型比较统一采用本地可复现值 0.6588。

## 审计配置

```bash
sbatch scripts/audit_sota.slurm
python scripts/audit_compute_metrics.py \
  --pred-dir results/audit_sota \
  --output results/audit_sota_metrics.json
```

推理不得使用 `mask_pocket`：它按真实 pocket 选择 ligand keypoint，会造成信息泄漏。原始公开 checkpoint、`FIX_TWO_ENDS=True` 或不同 PDB/SDF 版本的结果不能与上述基线混用。

## DockQ 唯一主口径

横向比较必须重建包含 protein1、protein2 和 PROTAC 的完整原子预测 PDB，然后调用：

```python
from DockQ.dockq_util import cal_dockq
fnat, irms, lrms, dockq = cal_dockq(pred_complex, native_complex)
```

链语义需与 DeepTernary benchmark 一致：protein1 与 ligand 属于 receptor 一侧，protein2 是 mobile chain。DockQ 使用原生接触、iRMS 和 Lrms；不能用 Cα-only、COM 距离、单界面 Fnat 或 `simple_dockq` 近似值替代主指标。近似指标应命名为 `approx_dockq`、`e3_dockq` 或 `poi_dockq`。

## TriCoopDiff 验收

1. 用 native pose 重建预测复合物，完整原子 DockQ 应 ≥0.999。
2. 同一预测 PDB 分别经 TriCoopDiff bridge 和 DeepTernary 工具计算，四个返回值应在浮点容差内一致。
3. ensemble 按完整 DockQ 选 best walker，并报告 `dockq_best/mean/median`、Fnat、iRMS、Lrms。
4. 结果记录 checkpoint、case 列表、seed 数、初始化模式和 DockQ 实现路径。

## 来源

- DeepTernary paper: [arXiv:2502.18875](https://arxiv.org/abs/2502.18875)
- Archived weights: [Zenodo 12727661](https://doi.org/10.5281/zenodo.12727661)
