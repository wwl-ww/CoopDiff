# 独立 DiffDock 小分子评估

## 范围与结论

评估已有的 PROTAC `5T35_D_A_759`、MGD `3JRQ_B_A_A8S` 独立双界面预测，
每侧 seeds 42–45，共 16 条。没有重新采样或用真实结构选择中间步。
关闭 Fuse、FiLM、LoRA；预测来自仓库 vendored DiffDock 和独立 SDE 脚本。
**与官方完整 DiffDock 推理入口的逐项一致性仍未完成，因此不能当作官方 benchmark。**

DockQ 2.1.3 官方小分子评分结果如下；最佳值仅在成功评分样本中选择，不是置信度 top-1。

| 案例 | 界面 | 成功评分/总数 | 最佳 LRMSD (Å) | 对应 DockQ-small_molecules |
|---|---|---:|---:|---:|
| PROTAC 5T35 | E3–配体 | 3/4 | 12.2926 | 0.441158 |
| PROTAC 5T35 | POI–配体 | 1/4 | 13.5932 | 0.427034 |
| MGD 3JRQ | 第一蛋白–配体 | 4/4 | 6.2367 | 0.550014 |
| MGD 3JRQ | 第二蛋白–配体 | 4/4 | 6.9340 | 0.533478 |

12 条成功评分样本均未达到 LRMSD < 2 Å。4 条不可评分不能解释为成功，亦不能
从成功评分子集中推断整个数据集的性能。完整 PROTAC LRMSD 也不等于结合头 RMSD。

## 指标含义

直接调用官方 `DockQ.DockQ.run_on_all_native_interfaces`，开启 HETATM 解析，
固定链映射 A=受体、B=完整配体。小分子路径执行受体口袋骨架对齐（native 原子
距离定义的 10 Å 界面）及坐标成键图的对称性匹配，报告完整重原子 LRMSD。

CLI 单侧主要输出 LRMSD，但还会显示 `Total DockQ-small_molecules`。
2.1.3 源码对小分子调用 `dockq_formula(0, 0, LRMSD)`，即
`(1 + 1/(1+(LRMSD/8.5)^2))/3`。它在 LRMSD=0 时为 2/3，
不是传统蛋白–蛋白三项 DockQ，**不能套用 0.23/0.49/0.80 质量阈值**。
逐条 JSON 保留此派生字段，不能把匹配失败时返回的 1/3 当作有效评分。

PROTAC 的预测受体来自 unbound 输入，参考受体来自同目录的 bound 晶体模板；
MGD 两者均使用对应的晶体受体。bound 模板只参与评分，不进入预测。
所有坐标通过已验证的缓存参考系映射，配体保留完全相同的原子编号。

## 验证与失败分析

- 四个 native-vs-native 对照 LRMSD 均 < 1e-6 Å。
- PDB Cα 到缓存坐标的刚体拟合最大误差 < 1.6e-5 Å。
- 官方 CLI 独立复核 MGD 第一侧 seed 42：LRMSD=6.237 Å，派生分数=0.550。
- PROTAC POI seed 45：原始预测第 4 步出现非有限输出，无法评分。
- 另三条 PROTAC 坐标有限，但 DockQ 的坐标成键图不再与参考图同构：
  E3 seed 45 多 8 条推断边；POI seed 42 多 1 条；POI seed 44 多 8 条。
  三者参考图均为 75 条边，无缺失边。额外近距离原子对保存在 JSON 中。
  这揭示配体自接触及距离成键推断的限制，不表示化学图真的增加了共价键。

## 复现与产物

官方 PyPI `dockq-2.1.3.tar.gz` 位于 `/tmp/pymp-f2c2db1a/dockq-v2-source/`。
本机编译器路径异常，运行未修改的官方 `operations_nocy` 后备实现。
依赖安装在 `/tmp/pymp-f2c2db1a/dockq-v2-env`，原 conda 环境未修改。

```bash
export PYTHONPATH=/tmp/pymp-f2c2db1a/dockq-v2-source/dockq-2.1.3/src:/tmp/pymp-f2c2db1a/dockq-v2-source/dockq-2.1.3/src/DockQ
export TRICOOPDIFF_CACHE_DIR=/tmp/pymp-f2c2db1a/results/runtime_cache/diffdock
/tmp/pymp-f2c2db1a/dockq-v2-env/bin/python scripts/score_binary_dockq.py \
  --results /tmp/pymp-f2c2db1a/results/binary_interfaces_v2_run2/results.json \
  --index /tmp/pymp-f2c2db1a/epoch20_evaluation/protac22_unbound.json \
  --index /tmp/pymp-f2c2db1a/epoch20_evaluation/mgd94.json \
  --out /tmp/pymp-f2c2db1a/results/binary_interfaces_dockq213_reproduction
```

正式本轮产物：`/tmp/pymp-f2c2db1a/results/binary_interfaces_dockq213_audited/`。
其中 `summary.json` 保存版本、源码哈希、命令和对照；`results.json` 保存每条
LRMSD、原子匹配、失败原因和预测 PDB 路径。评分仅用 CPU。

官方说明：[DockQ README](https://github.com/wallnerlab/DockQ#scoring-small-molecule-docking-poses)。
