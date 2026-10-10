# TriCoopDiff

蛋白–配体–蛋白三元装配模型。两个冻结的二元专家通过 FiLM 接收另一侧界面的条件，提供逐节点证据；三元网络读取三个组分的完整节点与几何，独立预测合法的刚体及配体 torsion 运动。不是把两个专家的刚体建议直接平均。

## 当前主线

主模型在 `tricoopdiff/models/assembly.py`，冻结专家接口在 `models/expert.py`。当前主目录支持 DiffDock 和 Matcha；FABind 集成位于独立工作树，不应把其入口参数直接用于本目录。

- **训练**：从真实初始化执行20步，监督实际终点；不以旧速度 MSE 作为当前训练目标。
- **运动自由度**：E3 固定，POI 整体刚体运动，配体刚体＋合法可旋转键运动。不调整蛋白内部结构。
- **条件专家**：专家权重冻结，FiLM 与三元网络可训练；主训练保留专家输入坐标的梯度。
- **监督**：配体及 POI 坐标、E3–L／POI–L 双界面几何、蛋白–蛋白界面与接触、粗粒度排斥。配体目标投影到起点的可达空间。
- **辅助训练**：近目标零／小扰动入口，权重0.25；日志分开记录主终点和辅助 loss。
- **batch size**：当前为串行微批梯度累积，不是真正的多图并行。
- **验证**：按 case 固定初始化，计算蛋白–蛋白完整原子 DockQ；默认 EMA。Raw 对照必须使用相同检查点和起点。
- **续跑**：保存模型、EMA、优化器、调度器、计数与 RNG；必须满足检查点和运行契约，不能静默替换架构。

当前代码契约为 `ternary_endpoint_v14`，主训练目标为 `real_initialization_dual_interface_endpoint_v3`。具体检查点版本以代码与保存的 manifest 为准。固定起点拟合、邻域迁移、未见起点迁移与跨复合物验证是不同能力，不能混报。

## 初始化

| 初始化 | 用途与边界 |
| --- | --- |
| SDPI | 共享配体、按受体尺度生成双侧随机姿态；当前 Matcha 标准训练使用此入口 |
| BiTSeed | 两个冻结 DiffDock 的刚体二元预测组成三元候选；不是三元精修或排序器 |
| 耦合锚点邻域 | 从预测装配中心选取可分离局部锚点，改变合法 torsion，恢复 E3 侧锚点并耦合移动 POI；研究接口，尚非标准训练 CLI |

耦合邻域实现为 `initialization/candidates.py::coupled_anchor_perturbation`。接触区域仅根据预测中心确定，不读取 native。没有两侧可分离、非共线锚点时明确标记 `rigid_only_no_separable_anchors`；不能把它统计成双锚点 torsion。小幅刚体误差扰动有意破坏精确结合关系。邻域测试好不等于宽初始化泛化。

现有 BiTSeed 生成器依赖 DiffDock；Matcha 的标准训练入口仅支持 A1／SDPI。研究中可将已生成的 BiTSeed 坐标交给 Matcha 控制器，但这不等于已经实现 Matcha 二元 BiTSeed 采样。

## 目录

- `tricoopdiff/models/`：三元模型与冻结二元专家。
- `tricoopdiff/train.py`、`sdpi_flow.py`、`sampler.py`：训练、共享几何更新及采样。
- `tricoopdiff/initialization/`、`geometry/`、`inputs/`、`metrics/`：初始化、合法几何、输入与评分。
- `tricoopdiff/diffdock/`：vendored DiffDock；`legacy/` 保留历史模型与 BiTSeed 二元扩散实现，生产模型不导入旧模型。
- `tricoopdiff/ranking/`、`experiments/`、`conditioning/`：保留已有研究库及兼容接口；退役的 ranking 训练与一次性消融脚本已移除。当前二元专家的 FiLM 互条件化仍保留。
- `scripts/`：数据准备、当前训练、评估及通用分析入口。
- `tests/`：现行模型、训练契约、几何、数据及评估的回归检查。
- `doc/`：历史研究与协议文档，从 [索引](doc/README.md) 进入；旧架构、版本和结果不代表当前实现。
- `data/`、`checkpoints/`、`results/`、`logs/`：本地资产与生成产物，不随脚本清理删除。

## 环境与检查

从仓库根目录运行。项目未作为 Python 包安装；[environment.yml](environment.yml) 是历史环境导出，不保证在不同 CUDA／驱动环境直接复建。需准备匹配的 PyTorch、图网络及化学依赖、专家权重和缓存；检查不应自动下载模型。

```bash
python tests/test_model_build.py
python tests/test_training_contract.py
python tests/test_initialization.py
python scripts/run_cpu_tests.py
```

`run_cpu_tests.py` 在隔离的离线 CPU 子进程运行保留的测试，需要已存在的 DiffDock 数值缓存。模型入口应先导入 `tricoopdiff.diffdock_compat`。

## 训练与续跑

以下示例假定本地数据、Matcha 源码／权重和特征缓存已经准备好。GPU编号及输出目录应按实际资源选择。

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_entry.py \
  --train_data data/ternary_train.json --val_data data/ternary_val.json \
  --expert-backend matcha --training-arm A1 --cached-only \
  --batch_size 4 --epochs 20 --lr 0.0001 --seed 73 \
  --save-steps 1 --save_interval 1 --log_interval 1 \
  --output_dir results/matcha_run
```

续跑使用同样配置，并增加 `--resume results/matcha_run/progress_model.pt`。后台可使用本地 tmux，不使用 SLURM；有些环境需通过 `env -u LD_LIBRARY_PATH tmux` 启动。检查点可恢复不等于已有自动重启守护。

`--cached-only` 只保留已有特征缓存的条目；每轮按簇抽样，不是遍历每个条目。必须报告实际条目、簇和初始化覆盖，不能仅用“全数据”代替协议说明。详细参数见 `python scripts/train_entry.py --help`。

## 评估与报告

数据索引由 `scripts/make_deepternary_index.py` 准备；训练入口进行确定性逐例验证，单例／批量及候选池评估使用 `evaluate_entry.py`、`evaluate_index.py` 等入口。调用前核对入口支持的 backend、检查点格式、初始化与 walker 配置，不能假定所有入口均支持所有专家。

- DockQ 衡量蛋白–蛋白界面，不包含配体质量；同时报告配体误差并注明相对于晶体还是可达目标。
- cached-only 训练验证不是 MGD94 完整基准。正式基准需完整病例、walker 数、初始化、评分和选取协议。
- 区分初始化不动、模型训练前、训练后，以及 raw／EMA。
- 区分 native-free 选择与 oracle；不能通过目标或 DockQ 选训练初始化。
- 训练进度与实验事实以 `metrics.jsonl`、检查点及运行 manifest 为准，不把运行中的结果写成最终结论。

贡献规范见 [AGENTS.md](AGENTS.md)。清理仅删除明确退役、无现行用途的源码入口；不清除正在运行任务的依赖、用户实验产物或必要质量门。
