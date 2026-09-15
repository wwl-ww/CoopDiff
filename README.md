# TriCoopDiff / BiTSeed

TriCoopDiff 研究蛋白–配体–蛋白三元装配：两个冻结 DiffDock 专家提供二元建议，残差共识模型协调双界面运动。BiTSeed 是在线初始化模块，不是精修模型或排序器。

## 当前研究状态

候选池工作配置为 **192 BiTSeed + 64 SDPI**，从中选择 64 个初态交给 TriCoopDiff。该配置用于后续对照，不代表全局最优；selector 尚未冻结。

MGD94 全池精修审计已完成。当前 checkpoint 下，Geometry-64 / Diversity-64 的最终 oracle 均值略高于 SDPI-64，但不足以证明唯一最佳选择器。初始化优势不保证精修后仍保留；这些结果不是 model-ranked top-1。详见[初始化与实验协议](doc/BiTSeed-Hybrid在线初始化规格.md)。

SDPI/BiTSeed 单模型联合训练已合入：B/C 各完成 100 epochs，最终选择带 source conditioning 的 C epoch 81 EMA，内部 validation loss 为 `3.1384998601`。LoRA 消融未超过无 LoRA 父模型，可训练 adapter 实现已删除。详见[联合训练交接](doc/SDPI-BiTSeed联合训练交接.md)。

## 目录

- `tricoopdiff/`：模型、坐标、初始化、训练与采样；`diffdock/` 为 vendored 专家。
- `scripts/`：特征化、实验、审计入口和历史运行 wrapper。
- `tests/`：断言式回归检查。
- `doc/`：主题文档；从[文档索引](doc/README.md)进入。
- `data/`、`checkpoints/`、`results/`、`logs/`：本地资产/产物。

## 环境与检查

从仓库根目录运行；代码未作为 Python 包安装。仓库有 [environment.yml](environment.yml)，但它是环境导出，不保证不同 CUDA/驱动机器可直接复建。

```bash
python tests/test_model_build.py
python tests/test_training_contract.py
python tests/test_dual_init.py
bash scripts/gpu_status.sh
```

本项目在本地 GPU 环境直接运行，不使用 SLURM。模型任务需要冻结专家权重、ESM 缓存、数据与匹配的 checkpoint。

## 接手与贡献

先读 [todo.md](todo.md) 确认当前边界，再读 [Q.md](Q.md) 的已知限制。贡献约定见 [AGENTS.md](AGENTS.md)；模型版本、数据和运行参数以当前代码及对应实验 manifest 为准。

报告必须区分初始化 oracle、精修 oracle、初末态并集 oracle 和 native-free top-1；不要以测试标签选择候选或以小样本结果宣称泛化。
