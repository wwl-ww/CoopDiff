# 脚本索引

从仓库根目录执行命令。训练、模型诊断及评估需要项目环境、已有缓存和对应 checkpoint；GPU 参数使用本地物理卡号。先用 `--help` 查看参数。

## 常用入口

```bash
python scripts/run_cpu_tests.py
python scripts/make_deepternary_index.py --help
python scripts/train_entry.py --help
python scripts/evaluate_entry.py --help
python scripts/evaluate_index.py --help
```

`train_entry.py` 是标准训练入口；`train_matcha_single.py` 是 Matcha 专用训练；`train_distributed.py` 的多卡实现与验收边界见[吞吐说明](../doc/TRAINING_THROUGHPUT.md)。评估入口的 backend、权重和初始化支持范围不同，使用前核对参数及[项目交接](../doc/PROJECT_HANDOFF.md)。

## 顶层工具

| 用途 | 文件 |
| --- | --- |
| 索引与面板 | `make_deepternary_index.py`、`make_index_subset.py`、`prepare_validation_index.py`、`prepare_local_mgd_panel.py` |
| 化学与条件准备 | `restore_ligand_chemistry.py`、`prepare_protac_binary_chemistry.py`、`make_protac_pocket_manifests.py` |
| 特征与输入检查 | `generate_esm_embeddings.py`、`audit_ternary_inputs.py`、`audit_ternary_featurization.py` |
| 评估准备与权重选择 | `prepare_checkpoint_evaluation.py`、`select_best_checkpoint.py` |
| 结构评估 | `evaluate_entry.py`、`evaluate_index.py`、`evaluate_selected_refinement.py`、`evaluate_binary_interfaces.py`、`score_binary_dockq.py` |
| 报告汇总 | `summarize_evaluation_reports.py`、`summarize_official_benchmark.py` |
| 运行辅助 | `gpu_status.sh`、`launch_dual_init_factorial_local.sh`、`run_cpu_tests.py`、`benchmark_training_throughput.py` |
| 小样本实验 | `run_small_sample.py`、`run_small_sample_fixed.py`、`run_coupled_small_sample.py` |

三个小样本脚本是不同实验协议，暂保留独立入口，不视为重复实现。`featurize_common.py`、`evaluation_common.py` 和 `evaluation_runner.py` 是共享模块，保留在顶层供现有入口导入。

## 诊断工具：`diagnostics/`

以下文件从顶层迁入，文件名与参数保持不变；旧路径不再保留包装脚本。

| 文件 | 用途 |
| --- | --- |
| `validate_initialization.py` | 真实案例初始化检查 |
| `validate_sdpi_flow.py` | SDPI 几何检查及可选 GPU smoke |
| `validate_assembly_model.py` | 装配模型与采样检查 |
| `audit_initialization_consistency.py` | 初始化一致性审计 |
| `audit_endpoint_support.py` | 推理与训练配体终点差异 |
| `audit_protac_conditions.py` | PROTAC 条件输入审计 |
| `train_loss_diagnostic.py` | 固定子集 local/trajectory loss 实验 |

例如：`python scripts/diagnostics/validate_sdpi_flow.py --help`。诊断通过不能替代正式数据集评估。

## 结果分析：`analysis/`

以下文件同样从顶层迁入：

- `analyze_ensemble.py`：分析已保存的构象集合。
- `analyze_cooperativity.py`：协同性分析。
- `compare_exact_runs.py`：按相同案例和初态配对比较完整 DockQ 报告。

例如：`python scripts/analysis/compare_exact_runs.py --help`。

## 维护约定

新增脚本先确认是否能扩展现有入口；专用诊断与离线分析分别放入对应子目录，并更新此索引。不要仅凭名称或缺少文档引用删除实验协议；删除前检查导入、调用方与复现需求。数据、结果及 checkpoint 不属于脚本清理范围。
