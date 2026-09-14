# 独立 DiffDock 小分子评估（历史对照）

这是对已有两案例预测的历史评分，不是当前BiTSeed/MGD94全池结果，也不是新跑的实验。关闭Fuse、FiLM、LoRA，采用vendored DiffDock独立SDE采样；与官方完整推理入口的逐项一致性尚未核实。

## 样本与结果

PROTAC `5T35_D_A_759`、MGD `3JRQ_B_A_A8S`，每侧seeds42–45，共16条，不以native挑中间步。历史DockQ 2.1.3评分：

| 案例 | 界面 | 可评分/总数 | 最佳完整ligand LRMSD (Å) |
| --- | --- | ---: | ---: |
| 5T35 | E3–ligand | 3/4 | 12.2926 |
| 5T35 | POI–ligand | 1/4 | 13.5932 |
| 3JRQ | 第一蛋白–ligand | 4/4 | 6.2367 |
| 3JRQ | 第二蛋白–ligand | 4/4 | 6.9340 |

12条可评分样本均无LRMSD<2 Å；最佳仅从可评分样本中取得，非置信度top-1。完整PROTAC LRMSD不等于结合头RMSD，小样本也不能证明整个二元先验无效。

## 指标与失败解释

历史调用 `DockQ.DockQ.run_on_all_native_interfaces`，启用HETATM、固定链映射A=受体/B=完整配体。通过native定义的10 Å口袋骨架对齐及对称图匹配，计算完整重原子LRMSD。

当时2.1.3实现的小分子派生量为 `(1+1/(1+(LRMSD/8.5)^2))/3`，不是蛋白–蛋白三项DockQ；**不得套用0.23/0.49/0.80质量阈值**，也不得把匹配失败返回的内部默认值当有效得分。本说明仅限当时核对的版本。

PROTAC POI seed45第4步非有限。另三条PROTAC因距离推断成键图额外出现1或8条边而图匹配失败；这反映自接触/距离成键的限制，不意味着化学图真的新增共价键。失败必须计入16条分母。

## 历史验收与来源

四个native-vs-native对照LRMSD<1e-6 Å；PDB Cα到缓存frame拟合最大误差<1.6e-5 Å；历史官方CLI核对MGD第一侧seed42为6.237 Å。PROTAC预测用unbound受体，bound模板只参与评分；MGD用bound内部结构。

当时使用官方源码的 `operations_nocy` CPU后备实现。报告曾位于 `/tmp/pymp-f2c2db1a/results/binary_interfaces_dockq213_audited/`，其中 `summary.json/results.json` 记录源码哈希、命令、映射和失败。该临时路径**不保证仍可用**，不能当作当前复跑环境。

旧完整命令可用 `git show 3070922:doc/独立DiffDock小分子评估.md` 查看。需要重现时先恢复原预测、索引、版本与哈希，再使用现有 `scripts/score_binary_dockq.py`；不要直接照抄已失效的临时PYTHONPATH或覆盖原始结果。当前主评估口径见[数据与指标协议](Phase%201%20数据协议.md)。
