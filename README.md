# TriCoopDiff / BiTSeed

TriCoopDiff 研究蛋白–配体–蛋白三元装配：两个冻结 DiffDock 专家提供二元建议，残差共识模型协调双界面运动。BiTSeed 是在线初始化模块，不是精修模型或排序器。

## 当前研究状态

本分支 `feat/deterministic-online-anchors` 固定工作候选池为 **192 BiTSeed + 64 SDPI**，从中选择 64 个初态交给 TriCoopDiff。候选池配置已确定用于后续对照，但不代表全局最优；selector 尚未冻结。

MGD94 全池精修审计已完成。当前 checkpoint 下，Geometry-64 / Diversity-64 的最终 oracle 均值略高于 SDPI-64，但不足以证明唯一最佳选择器。初始化优势不保证精修后仍保留；这些结果不是 model-ranked top-1。配置、三张榜单与限制统一见[初始化与实验协议](doc/BiTSeed-Hybrid在线初始化规格.md)。

兼容 SDPI / BiTSeed 的**单模型联合训练**在独立分支 `exp/dual-init` 推进，不属于本分支已完成能力。

## 目录

- `tricoopdiff/`：模型、坐标、初始化、训练与采样；`diffdock/` 为 vendored 专家。
- `scripts/`：特征化、实验、审计入口和站点相关 SLURM wrapper。
- `tests/`：断言式回归检查。
- `doc/`：主题文档；从[文档索引](doc/README.md)进入。
- `data/`、`checkpoints/`、`results/`、`logs/`：本地资产/产物，迁移时须另行传输。

## 环境与检查

从仓库根目录运行；代码未作为 Python 包安装。仓库有 [environment.yml](environment.yml)，但它是环境导出，不保证不同 CUDA/驱动机器可直接复建。不要因导出文件与当前解释器不同就原地升级运行环境。

```bash
python tests/test_model_build.py
python tests/test_initialization.py
python tests/test_sdpi_flow.py
bash scripts/gpu_status.sh
```

GPU 脚本一次查看所有卡及进程；占用快照不代表使用权限。模型任务需要冻结专家权重、ESM 缓存、数据与匹配的 checkpoint。详细入口和单例 smoke 见[运行指南](doc/训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md)。

## 接手与贡献

先读 [todo.md](todo.md) 确认本分支边界，再读 [Q.md](Q.md) 的已知限制。贡献约定见 [AGENTS.md](AGENTS.md)；模型版本、数据和运行参数以当前代码及对应实验 manifest 为准，不沿用旧训练文档的 v13/LoRA 配置。

每个主题只保留一个主文档。报告必须区分初始化 oracle、精修 oracle、初末态并集 oracle 和 native-free top-1；不要以测试标签选择候选或以小样本结果宣称泛化。
