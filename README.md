# TriCoopDiff

最新交接请先读 [项目交接：当前模型、实验与运行状态](doc/PROJECT_HANDOFF.md)（2026-10-10，`model-v4`）。包含运行任务、结果边界、环境资产与续跑说明。

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

## 部署范围与代码版本

以下部署说明针对 `model-v4` 的 **Matcha 条件二元专家＋三元装配模型**。FABind 在 `feat/fabind-expert` 分支，需使用该分支环境与模型；不能混用检查点。项目目前是研究代码，不是已发布的独立推理服务。

```bash
git clone --branch model-v4 https://github.com/wwl-ww/CoopDiff.git
cd CoopDiff
```

**代码仓库不包含数据、外部专家权重、我们训练的三元权重和大部分缓存。** 从头训练需要以下公开资源；部署已有模型还必须向项目维护者获取匹配的三元 checkpoint、训练 `config.json`、专家 provenance 和必要运行资产。目前没有发布我们三元权重的公共下载地址。外部 Matcha 权重不能代替我们训练出的三元权重。

## 环境安装

当前实际训练使用 Linux、Python **3.10.20**、PyTorch **2.13.0+cu130**（CUDA runtime13.0）和 NVIDIA GPU。驱动须支持所选 PyTorch CUDA runtime；`nvidia-smi` 中的 CUDA 字段不是 Python 环境实际加载的 runtime。32 GB 显存卡是本机运行配置，不是已验证的最低显存要求；20步全链训练的峰值随蛋白长度变化。CPU可跑轻量检查，不适合完整长训。

| 依赖 | 本机已用版本／作用 |
| --- | --- |
| numpy / scipy | 1.26.4 / 1.15.3；几何与投影 |
| torch-geometric / e3nn | 2.8.0.post1 / 0.6.0；图与等变专家 |
| transformers / safetensors | 4.46.3 / 0.8.0；ESM和专家权重 |
| RDKit / ProDy | 2022.09.5 / 2.6.1；分子图、torsion和蛋白几何 |
| biopython / networkx / PyYAML | 1.88 / 3.4.2 / 6.0.3；结构、图、配置 |
| huggingface-hub | 0.36.2；预先下载模型 |
| timm / torchvision | 1.0.22 / 0.28.0+cu130；Matcha导入所需，与上述PyTorch运行组合对应 |

`tricoopdiff.diffdock_compat` 提供部分 `torch_scatter`／`torch_cluster` 的兼容实现；入口须先导入它。不要盲装与 torch/CUDA 不匹配的编译扩展。

**优先复建现有训练环境，而非套用外部模型的完整环境。** 本机入口是 `results/matcha_qualification_env/bin/python`，它依赖 `/home/wl/micromamba/envs/tricoopdiff` 的基础解释器及包，不能只复制该目录就认为可迁移。部署交付时应同时提供基础环境导出、该解释器的 `python -m pip freeze`、CUDA／驱动信息，并在新机重新验证。

若新建环境，可先创建 Python3.10 环境，再按 [PyTorch安装说明](https://pytorch.org/get-started/locally/) 安装与目标驱动兼容的 torch 和 torchvision；上述版本表是当前运行参照，不是所有组合都已测试的兼容矩阵。其余主要依赖可安装：

```bash
python -m pip install numpy==1.26.4 scipy==1.15.3 \
  torch-geometric==2.8.0.post1 e3nn==0.6.0 \
  transformers==4.46.3 safetensors==0.8.0 \
  biopython==1.88 prody==2.6.1 networkx==3.4.2 \
  PyYAML==6.0.3 huggingface-hub==0.36.2 timm==1.0.22 tqdm pandas
```

RDKit应单独安装与Python3.10兼容的版本；例如历史PyPI包 `rdkit-pypi==2022.9.5`，或使用conda-forge对应版本。移交时以实际环境导出为准，不要同时混装多个RDKit发行包。`timm`还会导入torchvision；出现算子注册错误时先检查torch／torchvision版本，而非修改模型。

[environment.yml](environment.yml) 是历史环境导出，含其他模型的依赖，**不是当前最小安装清单**。当前集成只导入 Matcha 源码；不要直接 `pip install -e results/matcha_qualification_source` 或 `uv sync`：该上游包声明Python3.12及另一组numpy／transformers版本，与本项目已运行环境不同。我们不调用其完整GNINA／PoseBusters CLI，不需要为了三元训练安装GNINA、OpenMM或下载其二元训练数据。

## 额外模型与下载位置

| 资源 | 官方来源 | 本项目放置位置与用途 |
| --- | --- | --- |
| Matcha源码 | [LigandPro/Matcha](https://github.com/LigandPro/Matcha) | `results/matcha_qualification_source`；冻结二元模型实现 |
| Matcha三个阶段权重 | [LigandPro/Matcha模型库](https://huggingface.co/LigandPro/Matcha/tree/main) | `results/matcha_qualification_weights/matcha_pipeline/stage{1,2,3}/model.safetensors` |
| ESM2-35M | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) | `results/matcha_qualification_weights/esm2_t12_35M`；Matcha480维序列特征 |
| ESM2-650M | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | Hugging Face默认cache；公共featurizer1280维特征 |
| DiffDock-L配置／权重 | [官方模型包v1.1](https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip) | `checkpoints/score_model`；公共featurizer配置，DiffDock／BiTSeed还需score权重 |
| 我们训练的三元checkpoint | 项目维护者移交，未公开发布 | 例如 `results/matcha_full_v55_seed73/{best_model.pt,best_loss_model.pt,progress_model.pt}` |

即使使用Matcha，`train_entry.py`仍读取DiffDock的 `model_parameters.yml`，不能把这个配置删除。仅Matcha A1特征流程不需要运行DiffDock score网络；使用DiffDock专家或BiTSeed时才需要其score权重。这里不需要DiffDock confidence模型。

从项目根目录准备外部源码：

```bash
mkdir -p results checkpoints
git clone https://github.com/LigandPro/Matcha.git results/matcha_qualification_source
git -C results/matcha_qualification_source checkout ada87e6906c4505e80e543f0b55dcdcf239d2fba
```

上述提交是本机实际源码版本。权重也必须与它匹配；**恢复已有训练必须使用checkpoint所记录的源码／权重hash，不能重新下载latest后绕过provenance检查。** 下列命令下载公共模型的当前版本，适合首次准备；已有训练恢复优先移交原资源，或将 `revision` 固定为该训练实际使用的Hugging Face提交。

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('LigandPro/Matcha', allow_patterns=['matcha_pipeline/**'],
                  local_dir='results/matcha_qualification_weights')
snapshot_download('facebook/esm2_t12_35M_UR50D',
                  local_dir='results/matcha_qualification_weights/esm2_t12_35M')
# 不设local_dir：代码按模型ID且local_files_only=True读取默认HF cache。
snapshot_download('facebook/esm2_t33_650M_UR50D')
PY
```

两个ESM目录必须包含模型、config和tokenizer文件，不能只下载权重。设置 `HF_HOME` 时须在下载和运行时保持一致；650M若下载到任意自定义目录，现有模型ID读取不一定能找到。35M与650M不能互换。

下载DiffDock官方zip后先查看内容，再把 **score_model目录里的文件** 放进 `checkpoints/score_model/`，至少包括 `model_parameters.yml`；使用其专家／BiTSeed还需要 `best_ema_inference_epoch_model.pt`。不能把压缩包直接放入目录，或误放confidence模型。已有运行应复制原配置及权重，不以另一版本配置重新生成原实验。

```bash
mkdir -p data/downloads checkpoints/diffdock_download
curl -L --fail -o data/downloads/diffdock_models.zip \
  https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip
unzip -l data/downloads/diffdock_models.zip
unzip -n data/downloads/diffdock_models.zip -d checkpoints/diffdock_download
```

从解压后的层级定位score目录，再放入目标位置；`-n`避免覆盖已有文件。本项目并不调用官方脚本的自动权重下载。

## 数据下载与索引生成

本项目当前训练数据来自 DeepTernary 的 TernaryDB。无需重新训练Matcha，因此不必下载Matcha论文中的PDBBind、Astex、DockGen等二元训练／评测数据。

| 数据／代码 | 下载来源 | 用途 |
| --- | --- | --- |
| DeepTernary仓库 | [官方仓库](https://github.com/youqingxiaozhua/DeepTernary) | 官方cluster、PROTAC22与MGD测试名单 |
| TernaryDB完整结构 | [TernaryDB.tar.gz v1.0.1](https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.1/TernaryDB.tar.gz) | `pdbs/<case>/gt_complex.pdb`；训练结构和native |
| DeepTernary output资源 | [output.zip v1.0.0](https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.0/output.zip) | 仅官方基线复现／相关unbound输入需用，不是Matcha训练权重 |

```bash
git clone https://github.com/youqingxiaozhua/DeepTernary.git DeepTernary
git -C DeepTernary checkout 827821dccca31a5918bd0355e2d6bf70c072b6dd
mkdir -p data/downloads data/TernaryDB_release
curl -L --fail -o data/downloads/TernaryDB.tar.gz \
  https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.1/TernaryDB.tar.gz
tar -tzf data/downloads/TernaryDB.tar.gz | head
```

解压到专门的数据目录后，确认实际根目录。以下示例要求 `data/TernaryDB_release/pdbs` 存在；若压缩包多了一层 `TernaryDB/`，将 `--release-root` 改为该层，不要盲目覆盖已有数据。

```bash
tar --keep-old-files -xzf data/downloads/TernaryDB.tar.gz -C data/TernaryDB_release
# 确认pdbs实际层级后再执行；本例假设没有额外顶层目录。
python scripts/make_deepternary_index.py \
  --release-root data/TernaryDB_release --output-root data/ternary_deepternary \
  --index-dir data --split-seed 0 --validation-fraction 0.1
```

此入口拆出两个protein PDB和单个ligand PDB，生成 `data/ternary_train.json`、`data/ternary_val.json`。训练池取官方PROTAC／MGD cluster成员交集，排除官方测试PDB，再做cluster级划分；不应自行把测试94例混回训练。脚本严格检查发布数据规模和不支持的配体图数量，失败时应核对数据／版本，不能随意改expected数量绕过。

标准入口默认native根目录是 `DeepTernary/data/TernaryDB/pdbs`。若native实际位于刚解压的目录，训练时显式加 `--native-root data/TernaryDB_release/pdbs`；不要误把拆出的单蛋白当作native三元复合物。部署自有输入需提供两个蛋白结构及一个可解析、单连通配体图；不是仅输入蛋白序列就能直接三元预测。

## 特征缓存与首次启动检查

公共特征缓存需要自行生成，不能从Git下载。可以在下载好ESM2-650M和DiffDock配置后预处理索引：

```bash
python scripts/audit_ternary_featurization.py --index data/ternary_train.json \
  --model-dir checkpoints/score_model --cache-dir results/featurize_cache \
  --device cuda --out results/featurization_train.json
python scripts/audit_ternary_featurization.py --index data/ternary_val.json \
  --model-dir checkpoints/score_model --cache-dir results/featurize_cache \
  --device cuda --out results/featurization_val.json
```

该入口会生成公共特征并报告逐例失败；Matcha的额外序列／化学metadata由其准备接口在训练载入时生成。首次预处理会耗时，先用 `--limit 1` 检查离线模型加载、结构解析和GPU环境，再准备完整数据。已有cache签名包含路径等信息，跨机器／目录迁移后可能需要重建，不能保证复制后零计算。

`--cached-only`不会帮你下载或生成缺失公共缓存：它会缩小参与训练的数据总体。新机不要拿空cache启动cached-only再称全数据训练。也可去掉该flag让标准入口按需生成，但首次训练吞吐会受预处理影响。

SO(3)／torus数值表在首次导入时计算并保存，或从原机移交。统一设置路径，避免多个进程同时首次生成：

```bash
export TRICOOPDIFF_CACHE_DIR="$PWD/results/runtime_cache/diffdock"
python -c 'import tricoopdiff.diffdock_compat; from tricoopdiff.diffdock.utils import so3, torus'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python scripts/train_entry.py --help
```

从仓库根目录运行，项目未作为Python包安装。检查不应自动下载模型。模型／结构资产准备完后运行：

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
  --matcha-source-dir results/matcha_qualification_source \
  --matcha-weights-dir results/matcha_qualification_weights \
  --expert_model_dir checkpoints/score_model \
  --cache_dir results/featurize_cache --native-root data/TernaryDB_release/pdbs \
  --batch_size 4 --epochs 20 --lr 0.0001 --seed 73 \
  --save-steps 1 --save_interval 1 --log_interval 1 \
  --output_dir results/matcha_run
```

续跑使用同样配置，并增加 `--resume results/matcha_run/progress_model.pt`。后台可使用本地 tmux，不使用 SLURM；有些环境需通过 `env -u LD_LIBRARY_PATH tmux` 启动。检查点可恢复不等于已有自动重启守护。

`--cached-only` 只保留已有特征缓存的条目；每轮按簇抽样，不是遍历每个条目。必须报告实际条目、簇和初始化覆盖，不能仅用“全数据”代替协议说明。详细参数见 `python scripts/train_entry.py --help`。

## 评估与报告

数据索引由 `scripts/make_deepternary_index.py` 准备；训练入口进行确定性逐例验证，单例／批量及候选池评估使用 `evaluate_entry.py`、`evaluate_index.py` 等入口。调用前核对入口支持的 backend、检查点格式、初始化与 walker 配置，不能假定所有入口均支持所有专家。

**当前通用候选评估runner仍按DiffDock构建，不是可直接加载Matcha v55的完整MGD94部署CLI。** Matcha当前可运行的是标准训练及其逐例验证；已有三元模型推理还需使用正确的backend构建／权重加载接口。不要把上游 `matcha` 二元CLI当作本项目三元推理入口，也不要用一个DiffDock评估命令宣称验证了Matcha。部署交付时必须连同对应模型调用代码、checkpoint和初始化协议移交。

- DockQ 衡量蛋白–蛋白界面，不包含配体质量；同时报告配体误差并注明相对于晶体还是可达目标。
- cached-only 训练验证不是 MGD94 完整基准。正式基准需完整病例、walker 数、初始化、评分和选取协议。
- 区分初始化不动、模型训练前、训练后，以及 raw／EMA。
- 区分 native-free 选择与 oracle；不能通过目标或 DockQ 选训练初始化。
- 训练进度与实验事实以 `metrics.jsonl`、检查点及运行 manifest 为准，不把运行中的结果写成最终结论。

贡献规范见 [AGENTS.md](AGENTS.md)。清理仅删除明确退役、无现行用途的源码入口；不清除正在运行任务的依赖、用户实验产物或必要质量门。
