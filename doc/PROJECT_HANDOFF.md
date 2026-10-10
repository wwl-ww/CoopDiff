# TriCoopDiff 项目交接：当前模型、实验与运行状态

更新：2026-10-10。本文对应 `model-v4` 分支、提交 `9de282d`（`feat: integrate Matcha assembly and coupled initialization`）。运行数据是当天约 10:03 UTC 的快照，不是实时监控；接手后先检查进程与结果文件。本文优先于旧诊断、组会和历史架构说明；代码、训练 config、checkpoint 契约和逐例结果仍是最终依据。

## 1. 现在到底在做什么

项目目标是预测蛋白–配体–蛋白三元装配：固定一个蛋白参考系，调整另一个蛋白的相对姿态及配体的姿态／可旋转键，使两个蛋白–配体界面与蛋白–蛋白界面同时正确。

当前不是 Dreamer／语义 token／Actor–Critic 路线，也不是仅一步结构修正。主线已经回到真实几何上的20步控制：两个冻结二元专家互为条件，提供节点证据；三元网络独立输出运动；从实际初始化执行20步，监督真正执行出来的终点。

最重要的未解决问题是：**模型能拟合固定起点，但不能稳定把不同的新起点送到正确装配。** 训练 loss 下降并不自动带来验证 DockQ 的充分提升。当前工作围绕三个层次区分原因：

1. 固定起点能否学会：已出现高质量结果，不支持“任何输入都学不会”的判断。
2. 同一复合物的新起点能否学会：全局重采优于只记固定起点，但仍明显不足；预测中心附近的小扰动有较强正向信号。
3. 新复合物能否预测：全数据训练的固定初始化验证目前仍弱，远未达到历史结果的竞争水平。

现在并行运行 Matcha 和 FABind 全数据训练；同时用8个训练复合物比较固定起点、持续重采、BiTSeed 和新的耦合邻域初始化。**新初始化尚未接入标准全数据训练 CLI，也没有完成其宽起点泛化验证。** 接手者不应把局部邻域实验当作已经解决整个问题。

## 2. 代码从哪里接

| 分支 | 已推送提交 | 用途 |
| --- | --- | --- |
| `model-v4` | `9de282d` | 当前主目录；Matcha 条件专家＋三元装配、耦合初始化及脚本／测试清理 |
| `feat/fabind-expert` | `671c59f` | 独立 FABind 集成与训练，不能混用 v4 的专家入口和 checkpoint |
| `model-4.5` | `d2acac8` | 另一套局部兼容性架构研究，不是当前 Matcha 主训练 |
| `model-v3` | `cf771e6` | 历史版本 |
| `main` | `49d162a` | 不是当前工作版本；不要直接在 main 修改 |

远端：`https://github.com/wwl-ww/CoopDiff.git`。

本机工作树：

- `/mnt/disk/wl/MoRE-tri/CoopDiff`：`model-v4`。
- `/mnt/disk/wl/MoRE-tri/CoopDiff-fabind`：`feat/fabind-expert`。
- `/mnt/disk/wl/MoRE-tri/model-4.5`：`model-4.5`。
- `CoopDiff-protac`、`CoopDiff-bitseed-mixed-path-repro` 和 detached 历史运行目录不是本轮主线。

先运行 `git branch --show-current`、`git status --short`、`git worktree list` 确认位置。不要根据目录时间或名字猜最新版本。文档交接本身位于上述提交之后的工作区。

## 3. 当前架构和信息流

```text
实际 E3 / ligand / POI 节点、化学特征、当前几何、时间与 dt
        │
        ├─ E3–L 冻结 Matcha ← FiLM：另一侧 POI 与 ligand 条件
        ├─ POI–L 冻结 Matcha ← FiLM：另一侧 E3 与 ligand 条件
        │              ↓ 逐节点标量与向量证据
        └──────── 三元完整节点几何交互网络
                       ↓
            ligand 刚体＋torsion、POI 独立刚体运动
                       ↓
             已知合法几何更新器 → 下一实际状态
                       ↓ 重复20步
                  真实终点几何监督
```

核心实现是 `tricoopdiff/models/assembly.py`，不是旧 `consensus.py`。当前契约：

| 字段 | 当前值 |
| --- | --- |
| assembly checkpoint version | `55` |
| model profile | `ternary_endpoint_v14` |
| information contract | `conditional_node_evidence_without_rigid_advice_v2` |
| conditioning contract | `stage_specific_chemical_interface_v5` |
| rigid decoder contract | `independent_horizontal_rotor_coupling_v4` |
| 真实终点目标 | `real_initialization_dual_interface_endpoint_v3` |

具体设计与边界：

- 两个专家权重冻结，FiLM 和三元网络可训练。当前主线不是 LoRA 专家微调。
- Matcha 有三个预训练阶段；全局逆时间通过 `matcha_stage_clock` 映射到阶段与阶段内进度。三个阶段使用各自的零初始化 FiLM，而非一个条件头硬套所有阶段。
- 对侧条件同时包含蛋白、配体化学与当前几何。条件先做节点交互，再汇聚；保留界面距离信息，不仅传一个“语义 token”。
- 专家提供320维节点标量和5通道向量证据；三元主干节点宽度128，读取完整组分节点、几何与化学／rotor 信息。
- **不直接平均两个专家的刚体速度，也不把原始专家刚体建议交给最终解码器决定运动。** 三元网络独立预测 ligand 与 POI 的刚体动作；专家 torsion 相关证据仍可使用。
- 专家输入坐标梯度保留。冻结权重不等于切断坐标导数；当前主训练反传真实20步执行链。
- E3 固定；POI 仅整体刚体运动；ligand 允许整体刚体＋合法 torsion。不能调整蛋白内部、侧链、环，不能修改配体固定键角。E3／POI 是参考／移动组分角色，不表示所有 MGD 都是真实 E3 体系。
- 完整节点访问不等于已经证明没有信息瓶颈。有限起点拟合成功，也不能证明宽状态分布上的表示能力足够。
- `dt` 是网络条件之一。20步训练改为40／80步会改变网络输入，不能默认仅仅提升数值积分精度。

`legacy/` 中的旧模型不是生产三元主干；其中部分二元扩散实现仍供 BiTSeed 使用，不能整目录删除。`ranking/`、`conditioning/`、`experiments/` 保留历史接口，不代表正在训练候选排序器。

## 4. 当前 loss：优化的到底是什么

在1 Å 坐标尺度下，主终点误差为：

\[
E=0.1\big[ L_{L,\mathrm{coord}}+L_{POI,\mathrm{coord}}
+0.5(L_{E3-L,\mathrm{vector}}+L_{E3-L,\mathrm{contact}}
+L_{POI-L,\mathrm{vector}}+L_{POI-L,\mathrm{contact}})
+L_{PP,\mathrm{interface}}+L_{PP,\mathrm{contact}}
+0.05L_{\mathrm{repulsion}}\big].
\]

实际训练还加入近目标入口：

\[
L_{\mathrm{train}}=E_{\mathrm{main}}+0.25E_{\mathrm{near}}.
\]

| 项 | 意义 |
| --- | --- |
| ligand 坐标 MSE | 监督最终配体位置／构象，对应起点可达的配体目标 |
| POI 坐标 MSE | 监督移动蛋白在固定 E3 系中的姿态 |
| E3–L、POI–L 相对向量与接触距离 | 两侧界面分别直接监督，不能靠整体配体误差代替 |
| PP 界面坐标与接触距离 | 蛋白–蛋白界面几何，与 DockQ 更相关，但不是 DockQ 本身 |
| 粗粒度排斥 | 三个跨组分配对的近距离排斥，不是完整物理能量 |
| near 辅助入口 | 已经接近目标时也应修正／保持，而非破坏 |

蛋白–配体接触集合使用 CA–ligand 8 Å 阈值，蛋白–蛋白界面使用 CA 10 Å 阈值；各项的精确归一化以 `train.py` 的终点损失函数为准。坐标 MSE 对笛卡尔分量取平均，不能直接把其平方根当作常规定义的原子 RMSD。

near 入口使用零扰动或小扰动，平移标准差0.5 Å、旋转0.05 rad、torsion 0.05 rad；起始时间为1或0.2，分别执行20或4步。验证不把该辅助入口加入主验证 loss。

当前没有旧 FM 速度 MSE，没有直接可微 DockQ loss；专家一致性／残差等旧辅助权重为0。DockQ 是另行测量的蛋白–蛋白完整原子指标，不衡量小分子位置。

### 可达投影与物理限制

配体 native 内部几何可能不在起点的“刚体＋torsion”可达集合中，因此先联合优化刚体旋转和 torsion，构造可达目标。当前 solver 契约是 `explicit_rotation_torsion_least_squares_v2`，投影契约是 `torsion_rigid_endpoint_projection_v1`；投影失败应报错，不能静默跳过。

**低可达目标 loss／RMSD 不等于恢复晶体配体。** 报告配体指标时必须写明相对 crystal 还是 projected target，并另查投影残差。预生成 coupling 已带目标的路径可能不重新记录投影求解统计。

现有排斥是 coarse、native-aware 的跨组分项，阈值上限约2 Å；没有元素范德华半径，也没有完整配体内部非键合排斥。数值有限不等于物理有效，torsion 保持局部键几何也不能防止远端自碰撞。

## 5. 三种初始化及新方法的准确定位

| 方法 | 当前用途 | 必须注意 |
| --- | --- | --- |
| SDPI | Matcha 标准 A1 训练；在线重采起点 | 宽分布更难；验证固定 case seed |
| BiTSeed | 冻结 DiffDock 双二元生成预测装配中心 | 不是 Matcha 二元生成器，不是 oracle 或排序器 |
| 耦合锚点邻域 | 围绕预测中心生成合法、结合关系耦合的局部扰动 | 研究接口；未接标准全量训练；不是全局泛化证明 |

新方法实现：`initialization/candidates.py::coupled_anchor_perturbation`，版本 `coupled_predicted_anchor_v1`。

只根据预测中心、化学图和蛋白 CA 接触确定锚点，不读取 native。选择能分开两侧接触片段的合法可旋转键；每侧需要至少3个非共线锚点。改变该键角后，把 E3 侧片段拟合回中心，按另一侧片段的运动耦合移动 POI／完整蛋白，再增加刻意的 POI 小刚体误差。

没有可分离锚点时明确退化为 `rigid_only_no_separable_anchors`，不能统计成双锚点 torsion。当前一次主要扰动一条选择出的 rotor，不是全 linker 多角度覆盖。锚点是几何接触启发式，不是人工化学 warhead 标注；错误预测中心可能给出错误接触归属。没有额外 native 筛选，也没有完备碰撞拒绝机制。

当前32个耦合实验状态中，仅14个是 `two_anchor_coupled_torsion`，其余18个是刚体 fallback。训练／测试共享每例两个预测中心，仅扰动随机数不同。因此它测的是**中心附近邻域迁移**，不是独立新中心迁移。

## 6. 全量训练的数据与执行协议

“全数据”目前准确含义是索引中的 cached-only 可用部分，而不是索引每条记录都参与每轮训练：

| 项 | 当前配置 |
| --- | --- |
| 训练索引 | `data/ternary_train.json`，可用5101／6834记录，覆盖587／587 clusters |
| 验证索引 | `data/ternary_val.json`，可用65／253记录，覆盖65／65 clusters |
| 每轮训练 | 每 cluster 抽一例，共587个样本；80%随机成员、20%代表例 |
| 蛋白角色 | 训练可随机翻转；验证固定 |
| 初态 | 训练在线 SDPI 重采；验证 `case_seed(case, 0)` 固定 |
| seed | 单模型随机种子73；初始化测试 seed 不是额外模型训练 seed |
| batch | 4，串行微批梯度累积，非多图并行；每轮147次 optimizer update |
| 计划 | 20 epochs，约2940次 update；学习率0.0001 |
| 执行 | 每个主入口真实20步；终点反传；全局梯度裁剪1 |
| 保存 | 每 update 写 `progress_model.pt`；每 epoch 保存独立 checkpoint |
| 验证 | 主终点 loss、完整原子 PP DockQ，默认 EMA |

梯度日志中的 `grad_norm` 是裁剪前值。曾用 float64 汇总范数避免有限 float32 梯度的范数计算溢出；这不意味着专家导数已经天然很小，也不证明大梯度不存在算法问题。全20步保留专家坐标导数仍昂贵。

## 7. 当前结果：哪些成立，哪些不成立

### 全数据训练

| 模型 | 快照进度 | 最近完成验证 | val loss | 平均 DockQ |
| --- | --- | --- | --- | --- |
| Matcha v55 | epoch7，update1008 | epoch6／update882 | 90.4560 | 0.05995 |
| FABind v51 | epoch3，update360 | epoch2 | 102.3554 | 0.05251 |

Matcha epoch1→6：loss 121.6699→90.4560，DockQ 0.04678→0.05995；最高已完成轮均值为 epoch5 的0.06076。**有小幅改善，但不足以宣称三元预测已做好。** 两个训练速度不同，当前轮数不同，不宜只按这个快照作最终排名。

epoch1同检查点、同65例和起点的 raw／EMA 对照：raw 平均 DockQ 0.04791、loss118.1657；EMA 0.04678、loss121.6699；两者均2／65超过0.23。这个对照不支持“弱表现主要由 EMA 造成”，但只覆盖该 epoch。

### 8个训练复合物上的起点对照

病例：`1AFQ_B_C_0FG`、`1ALZ_A_B_EOH`、`1BHX_B_F_R56`、`1HKD_A_B_GYP`、`5DRN_H_L_5CT`、`1CJT_A_B_DAD`、`1JLL_B_D_COA`、`1D8D_B_P_FII`。它们是缓存可用训练 cluster 的前8个代表例，不是按结果挑的；也不都是典型 PROTAC。

共同设置：从 Matcha raw epoch1／update147 权重开始，优化器重置，400次 update、batch4、真实20步及相同 near 辅助训练；初始化 seed73／74用于训练、75／76用于 held-out。同8个复合物的未训练起点不等于未见复合物验证。下表 ligand RMSD 是**可达目标 RMSD**。

| 组 | 最后可用评估 update | 已训练／参考起点 DockQ | held-out DockQ | held-out loss | held-out ligand RMSD |
| --- | --- | --- | --- | --- | --- |
| A：固定 SDPI | 400，已完成 | 0.67479 | 0.09118 | 39.4477 | 8.4959 Å |
| B：在线重采 SDPI | 400，已完成 | 0.19018 | 0.18665 | 17.9707 | 6.7466 Å |
| C：名义 fresh | 320附近，进行中 | 不作独立对照 | 无效对照 | — | — |
| D：独立 BiTSeed 起点 | 360，进行中 | 0.69185 | 0.10351 | 37.4551 | 9.4215 Å |
| E：耦合局部邻域 | 360，进行中 | 0.68286 | 0.65172 | 2.0137 | 3.4839 Å |

A 的 held-out 0／16超过0.23；B 为5／16。B 日志里的 `train` 是固定参考起点评估，不是在线训练起点本身，不能把它解释为训练集记忆分数。

E 的 held-out 中，真 torsion 耦合7例均值0.62491，其余刚体 fallback9例均值0.67258。高分说明当前中心附近存在可学习邻域；**不能把混合均值叫作全部双锚点 torsion 成功，更不能声称覆盖宽起点分布。** 当前小扰动尺度：torsion0.15 rad、POI平移0.5 Å、旋转0.05 rad。

**C 组无效原因已核对实际启动命令：仍执行 epoch1 `load_state_dict`，并非随机权重；自动字符串替换没有达到目的。其轨迹与 A 重复。** 不要相信 config 中的 fresh 标签，不要把它纳入权重迁移消融结论。

### 历史单样本证据

`4G8A_B_C_LP5` 的旧 Matcha 单起点实验150次更新后：loss0.39111、DockQ0.96529、可达 ligand RMSD1.92821 Å，见 `results/matcha_dual_interface_fit_lr1e4`。证明该设定能拟合，不是 v55 全数据泛化成绩。

FABind 同类单样本最低 loss 约1.05704，对应 DockQ0.91996、ligand RMSD3.22563 Å，见 FABind 工作树 `results/fabind_single_joint_lr1e4`。没有同时满足 DockQ≥0.90 与 ligand RMSD≤2 Å。

## 8. 活跃任务与产物位置

所有任务是本地后台 tmux，不是 SLURM；没有自动故障重启守护。PID 仅为快照。

| GPU | tmux / PID | 输出目录，相对所在工作树 |
| --- | --- | --- |
| 4 | `matcha_full_v55_gpu4` / 3813044 | `results/matcha_full_v55_seed73` |
| 3 | `fabind_full_v51_gpu3` / 3813052 | `results/fabind_full_v51_seed73`（FABind 树） |
| 7 | `matcha_small8_queue_gpu7` / 802693 | `results/matcha_v55_small8_fixed2_fresh_seed73`，无效 C 对照 |
| 5 | `matcha_small8_bitseed_gpu5` / 3586660 | `results/matcha_v55_small8_bitseed_seed73` |
| 0 | `matcha_small8_coupled_gpu0` / 3632097 | `results/matcha_v55_small8_coupled_seed73` |

日志分别是：

- `logs/matcha_full_v55_gpu4.log`。
- FABind 树 `results/fabind_full_v51_gpu3.log`。
- `logs/matcha_v55_small8_fixed2_gpu7.log`、`logs/matcha_v55_small8_resample_gpu7.log`、`logs/matcha_v55_small8_fixed2_fresh_gpu7.log`。
- `logs/matcha_v55_small8_bitseed_gpu5.log`、`logs/matcha_v55_small8_coupled_gpu0.log`。

小实验看逐次 loss／eval JSONL、`latest_eval.json` 和最后 `complete` 事件，不能仅凭进程消失认定完成。GPU7 队列 A→B→C，A／B已完成，C结束后无下一组。

GPU0／5与别人的 LoRA 共享；本实验的 PyTorch allocator fraction0.35不是 GPU 算力限额，不保证共享公平。GPU1／2／6有其他使用者，尤其不能杀 GPU6 的他人进程。授权使用的物理卡是0、3、4、5、7；接手后重新确认资源授权。

## 9. 启动、检查与恢复

Matcha 运行环境：`results/matcha_qualification_env/bin/python`。从 root 工作目录运行：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
TRICOOPDIFF_CACHE_DIR=/mnt/disk/wl/MoRE-tri/CoopDiff/results/runtime_cache/diffdock \
results/matcha_qualification_env/bin/python -u scripts/train_entry.py \
  --train_data data/ternary_train.json --val_data data/ternary_val.json \
  --output_dir results/matcha_full_v55_seed73 --expert-backend matcha \
  --cached-only --training-arm A1 --epochs 20 --lr 0.0001 \
  --batch_size 4 --seed 73 --save-steps 1 --save_interval 1 --log_interval 1
```

这是**重建命令，不是让接手人现在再启动一个同目录任务**。恢复时使用相同配置并增加：

```bash
--resume results/matcha_full_v55_seed73/progress_model.pt
```

确认原进程已结束后才恢复，输出目录只允许一个 writer。恢复包含模型、EMA、Adam参数状态、调度器、计数、RNG和契约检查；不能改 checkpoint 字段绕过不兼容检查。`best_model.pt` 按 DockQ，`best_loss_model.pt` 按主验证 loss；与最新 `progress_model.pt` 不同。

FABind 必须在 FABind 工作树，用 `results/fabind_py310_env/bin/python` 和该树训练入口；其专家参数为 `--expert-backend fabind --expert-source-dir results/fabind_source --expert_model_dir results/fabind_weights`，featurizer 指向主树 `checkpoints/score_model`。其他数据、缓存和 native 路径也要按实际 config 保留；不能用 Matcha v55 环境／模型恢复 FABind v51。

小实验通过内联 `python -c` 启动，**没有独立、版本化训练脚本**。已导出实际启动 argv 到 `results/project_handoff_runtime_snapshot/launchers.json`；该文件在 ignored results 中，必须随资产交付。恢复需阅读该源码，保留 couplings、配置和权重来源，再在其循环中接入 checkpoint 恢复；不能认为标准 `train_entry.py --resume` 会复现研究循环。研究计数从0开始，与原 epoch1 权重的147次更新分开。

常用只读检查：

```bash
env -u LD_LIBRARY_PATH tmux list-sessions
nvidia-smi
tail -n 40 logs/matcha_full_v55_gpu4.log
git status --short
```

本机 tmux 可能受继承的 `LD_LIBRARY_PATH` 影响，使用上面的前缀。不要用系统默认 Python 猜模型环境。

## 10. 移交机器／账号时必须一起交付的东西

**Git clone 不包含运行所需资产。** `data/`、`checkpoints/`、`results/`、`logs/` 大量内容被忽略；只交代码接手人无法复跑。

| 资产 | 作用 |
| --- | --- |
| `data/ternary_train.json`、`ternary_val.json` 与实际结构输入 | 样本划分、cluster与路径；索引路径需可解析 |
| `DeepTernary/data/TernaryDB/pdbs` 及相应官方数据 | native结构与完整原子评分 |
| `results/matcha_qualification_source` | 实际使用的外部 Matcha 源码 |
| `results/matcha_qualification_weights` | 三阶段 safetensors 与本地 ESM2-35M 资源 |
| `checkpoints/score_model` | DiffDock 配置／权重，featurizer及 BiTSeed 依赖 |
| `results/featurize_cache`、其中 Matcha metadata | cached-only训练的可用输入基础 |
| `results/runtime_cache/diffdock` | SO(3)／torus等数值缓存 |
| 两个工作树的运行输出、日志及 configs | 权重、续跑、逐例证据、初始化 couplings |
| FABind 树 `results/fabind_source`、`fabind_weights` | FABind 外部代码／资源 |
| `results/project_handoff_runtime_snapshot/launchers.json` | 内联实验真正的运行命令 |

Matcha 内部 ESM2-35M 与原 featurizer 的 ESM2-650M 不是同一资源／维度，不能互相替代。专家来源、权重路径与契约需要核对 adapter provenance，勿静默切换预训练版本。外部模型／数据许可证仍要按原项目要求处理。

当前 Matcha 环境 Python3.10.20；已核对版本：torch2.13.0、numpy1.26.4、scipy1.15.3、e3nn0.6.0、torch-geometric2.8.0.post1、transformers4.46.3、safetensors0.8.0、biopython1.88、RDKit2022.09.5。实际解释器关联本地 `/home/wl/micromamba/envs/tricoopdiff`；仅复制虚拟环境目录可能因绝对路径／链接失效。`environment.yml` 是历史导出，不保证直接复建当前 CUDA 环境。应在原机导出实际环境，并在目标机验证依赖、离线模型加载和缓存。

根目录 `loss_curve.png` 是本地生成图、非 Git 交付；如需要应单独复制。分支迁移应急备份在 `/tmp/coopdiff-branch-recovery-3g2Jyv/`，不是可靠长期归档。

## 11. 评估协议：最容易接错的地方

当前65例训练验证不是 MGD94 正式完整验证。过去用户要求的协议包含94例、各例 walker 数、多种初始化和候选选择；这些都要记录，不能用单起点65例均值替代。

`scripts/evaluation_runner.py` 目前仍按 DiffDock 构建模型；**不能直接拿它加载 Matcha v55 就宣布完成标准 MGD 验证。** 新 backend 的完整协议桥接是待做工作。训练验证的 DockQ 是完整原子 PP 指标，但样本／初态／候选池协议不同。

`build_trainer` 本身不会替调用者加载 checkpoint 权重。只做 raw 验证时，必须明确加载 raw state；使用 `validate(..., weights='raw')` 或 `'ema'` 并保持同 case seed／角色。不能只构造模型然后误评随机权重。

历史二元专家系统曾有用户所述 MGD 0.25+、256候选池接近0.3的结果；正式对比前必须找到对应 manifest、walker、初始化与 selector，不能把这个数字与当前65例单起点结果直接作同协议排名。

### 已完成的 DeepTernary 对照复现

产物：`results/deepternary_mgd94_reproduction_827821d_v1`，包含 `summary.json`、`per_case.json`、`manifest.json`、`verification.json`、`environment_freeze.txt`、`last_invocation.json`、`reproduce.py` 和预测结构。

94／94执行完成；一个公开 checkpoint、CPU推理、显式种子，**不是论文多个独立训练模型的完整复现**。结果：平均 DockQ0.22909，中位数0.09685，DockQ>0.23 为29／94；配体整体 pose RMSD平均12.33474 Å、中位数9.70879 Å，整体 RMSD<2 Å为0／94。

配体 RMSD 在固定蛋白框架、固定原子对应下计算，不是单独把配体对齐后计算。逐原子误差<2 Å占比与“整个配体 RMSD<2 Å的病例通过率”不同，不能混用。论文及补充材料并非只报告蛋白指标；具体对照与原始资料看该目录，不要重复传播“没有小分子指标”的旧判断。

## 12. 问题清单和建议接手顺序

| 优先级 | 未解决事项 | 现有证据／处理方向 |
| --- | --- | --- |
| P0 | 交付可运行性 | Git不含资产；先移交环境、权重、cache、couplings、启动源码，确认无重复 writer |
| P0 | C 对照标签错误 | 已证实不是 fresh；标为无效重复，不能纳入结论；是否停止需由负责人决定 |
| P1 | 宽起点收敛不足 | A固定拟合强、held-out弱；B重采改善但仍低，不能只继续刷固定点到0.99 |
| P1 | 跨复合物验证弱 | 主训练均值约0.06；需完整曲线和逐例结果，loss下降不是充分成果 |
| P1 | 耦合邻域结论范围窄 | train/test共享中心，且半数以上是rigid fallback；下一步必须独立中心和更宽扰动 |
| P1 | loss／评价与目标投影错位 | PP DockQ不含ligand；同时报告双界面、crystal及reachable RMSD、投影残差 |
| P1 | Matcha完整MGD协议缺口 | generic evaluator还是DiffDock；接backend并严格核验checkpoint和候选协议 |
| P1 | 全链计算与专家梯度代价 | 保留坐标导数的20步仍贵；batch4是累积，不能据此估多图吞吐 |
| P2 | 信息表达是否充分 | 全节点并不证明充分；对侧条件、几何向量、长linker与阶段适配仍需代码层面审查 |
| P2 | 物理合法性盲区 | coarse排斥不是完整碰撞检测；蛋白内部运动不可达，配体自碰撞未完整约束 |
| P2 | 研究实验可重放性 | 内联driver未版本化；复用现有入口整理必要接口，而非继续堆一次性脚本 |

推荐先完成现有小实验的400步收尾并记录最终 raw 指标，不再启动重复队列；保留主训练作为不动的基线。A／B在同 held-out 集上可直接比较；D独立中心与E同中心邻域必须分开解释。

若继续新初始化，最有依据的下一步是：预测中心附近持续重采合法扰动、扩大扰动范围、保留真实宽起点比例，并用**独立新预测中心**验证。成功邻域扩张只是待验证训练组织方案，不是已经实现的全数据课程，也不是替换终点监督的理由。

最终报告至少包含：起点 DockQ、模型0更新时结果、最终 DockQ均值／中位数／逐例变化／通过数、ligand crystal与reachable RMSD、初始化mode与中心独立性、实际GPU成本。不能靠几个高分seed、critic分数或total loss单独宣布成功。

## 13. 代码导航、检查和协作约束

| 位置 | 负责内容 |
| --- | --- |
| `models/assembly.py` | 当前三元节点交互、条件化与运动解码 |
| `models/expert.py` | 冻结专家adapter、阶段映射、证据／梯度接口 |
| `train.py` | 实际终点rollout、loss、验证、checkpoint／resume |
| `sdpi_flow.py`、`sampler.py` | 合法几何更新、时间步与采样 |
| `initialization/candidates.py` | BiTSeed及耦合锚点扰动 |
| `initialization/endpoints.py` | 可达目标投影 |
| `inputs/`、`metrics/` | 输入契约、结构映射和评分 |
| `scripts/train_entry.py` | 标准训练配置／cached-only入口 |

从仓库根目录、正确环境执行：

```bash
python tests/test_model_build.py
python tests/test_training_contract.py
python tests/test_initialization.py
python scripts/run_cpu_tests.py
```

提交前本地完整 CPU 检查曾50文件／418项通过；随后删除5个未跟踪旧分支测试，不能把418当作干净clone固定测试数量。CPU通过只证明相应契约，不证明GPU训练或模型性能。不要让测试静默下载模型。

用户明确偏好：只训练单模型seed；看loss也看DockQ；不再平均投入大量新范式；不要把一次性试验变成大量文件；错误实现应根本修正，而不是叠兼容补丁。最近清理授权针对无用脚本／测试，**不是授权为了文件少而删核心模型**。遵循 `AGENTS.md`，不使用SLURM，不修改main，不杀他人任务，不删除数据／checkpoint／结果，不把历史负结果或局部成功包装成已解决问题。

项目交付的诚实边界：当前已有完整条件二元＋三元控制架构、实际终点训练和可恢复后台任务；固定点及小邻域可高分，但宽初态与跨复合物预测仍未达到目标。接手者最该延续的是这个问题与证据链，而不是此前不停重构的节奏。
