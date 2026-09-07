# Dual consensus v1：当前实验分支实现

本分支基于 `origin/master` 的 `52d5e5b`，实现根目录
[`sug_gpt6.md`](../sug_gpt6.md) 第 24 节的基础版。
这是待验证的新建模假设，不能把数值稳定、loss 下降或双假设一致称为 DockQ 改善。

## 状态与坐标契约

`state = {a, b, poi, body}`。两份配体共享原子顺序、分子图、旋转键定义，
各自拥有完整的刚体和 torsion 状态；只有 E3 固定。
`body` 将 POI 原始局部坐标旋转到当前 E3 全局坐标。
Expert A 只读取 E3 和 `a`，Expert B 只读取 POI 和 `b`。

Expert 的 tr/rot/tor 经预训练单位预条件化后，作为可训练速度参数化；
它不是未经训练即可使用的 score-to-flow 恒等式。
Fuse 读取两份几何上下文、expert proposal 和逐层标量信息，预测三套独立刚体速度：
A 配体、B 配体相对 POI、POI；另有两套逐键 torsion 残差。
刚体头以旋转不变量预测系数，组合全局几何向量和 expert 向量，保持 SE(3) 等变。
跨专家逐层桥接读取 detached 标量证据，专家本身的 LoRA 梯度通路仍保留。

反向积分 `dt > 0`：各自将配体更新 `-dt*v`，再使用 POI 独立运动搬运 B 配体。
A 配体不跟随 POI；B expert 不直接移动 POI；不再执行每步 B→A 对齐或共享 torsion 合并。
重建仅使用刚体及旋转键旋转，训练辅助损失也使用可微重建。

## 初始化与可达性

每个化学图使用固定 seed 0 的 ETKDGv3 / `useRandomCoords=True` 基构象。
训练和推理共享该几何生成策略，walker 通过两套独立均匀 torsion、Haar 旋转和高斯平移产生多样性。
不再每个 walker 换一套键长/环几何，以免训练目标流形和推理流形不同。

两侧局部配体平移尺度仍来自各自受体 RMS radius 的 SDPI 公式。
POI 现在有独立 Haar 姿态和以 E3 质心为中心、标准差为两侧 SDPI 尺度之和的平移先验。
这个新增 POI 先验是本分支的明确选择，需要消融；它不是旧 SDPI 逆装配先验。

监督端先将相同 ETKDG 基构象通过 torsion + proper Kabsch 最小二乘投影到晶体，
得到 reachable target，记录 `projection_rmsd`。优化最多 100 次函数评估；
这是局部数值投影，不声称找到全局最优构象，也不声称消除了投影偏差。
native 坐标只用于训练目标；推理不读取 native 配体几何。
无 torsion 模式下只做刚体拟合。

时间 `t=0` 为可达监督端，`t=1` 为独立先验端。
训练路径对 POI 本体运动及两侧配体相对运动分别给出解析切向量，
包含 torsion 刚体 gauge spin；有限差分测试独立核查三套坐标更新。

## 目标函数与消融

主损失监督两套局部 tr/rot/tor 和独立 POI tr/rot；额外保留可配置的 raw expert torsion 辅助项。
使用单步速度外推的 clean estimate 计算辅助损失。这是近似估计，
不能当作一般时变速度场的精确完整轨迹端点。

令 `s=1-t`：internal / 周期 torsion 共识权重为 `0.1*s`；
endpoint 权重为 `max((s-0.3)/0.7,0)^2`；
全局 pose 共识权重为 `max((s-0.65)/0.35,0)^2`。
配体 Fuse 残差另乘 `s`，早期不通过这一路施加强制几何统一。
endpoint 使用分子图直径端点，支持 PROTAC 和 MGD，不能把它称为已知 warhead 注释。
未获得可信 warhead assignment 时不伪造 atom-wise responsibility mask；
本版采用第 24 节的完整二元 proposal。直接逐原子缩放刚体速度会破坏刚体/化学约束。

独立 global loss 包含：POI 本体速度、clean estimate 的 POI/配体坐标、
E3–POI / POI–配体 / E3–配体的接触概率回归。
接触项均匀确定性抽样最多 64 个受体残基，属于 surrogate，绝非完整 DockQ。
训练同时记录 `pol_rot_cos`（局部 B 配体）与 `poi_rot_cos`（独立 POI），防止混淆。

`--dual-stage`：

| 阶段 | 跨侧 FiLM/bridge | 几何残差 | 共识 | global loss |
|---|---|---|---|---|
| independent | 关闭 | 关闭 | 无 | 无 |
| exchange | 开启 | 关闭 | 无 | 无 |
| soft | 开启 | 开启 | internal/torsion/endpoint | 无 |
| annealed（默认） | 开启 | 开启 | 加入 late pose | 有 |

这些模式用于独立训练的消融；resume 必须保持阶段一致。

## 最终输出与边界

v1 简单 collapse 输出 `E3–a–POI`，保留 A 的物理构象，不做全原子坐标平均。
额外返回 `hypothesis_b_coords`、`consensus_rmsd` 和逐步 trace。
这保证只导出一份化学完整配体，但不能保证其 B 界面正确。
`valid` 只表示数值积分完成，不表示共识达标或 docking 成功。
learned collapse、动态 anchor、confidence gating、adaptive scheduler 均未启用。

## 运行与验收

训练入口仍为 `scripts/train_entry.py`；评估入口仍为 `scripts/evaluate_entry.py`。
必须从头训练 adapter/Fuse；checkpoint v18 / `dual_consensus_v1` /
`dual_consensus_flow_v1` 与旧 v17 不兼容，加载时显式拒绝。
checkpoint 同时校验构象、先验、投影、共识和 collapse 契约以及消融阶段。
请使用新的输出目录，例如 `results/dual_consensus_phase1`。

```bash
python tests/test_model_build.py
python tests/test_dual_consensus.py
python -m pytest tests/test_evaluation_common.py
python scripts/check_dual_consensus.py \
  --cache-dir results/featurize_cache_protac22_verify \
  --cases 5T35_D_A_759 --steps 20 --sampling-steps 20 \
  --out results/dual_consensus_check
```

最后一个命令只使用显式 v5 缓存做离线数值诊断，不能作为训练集/测试集性能。
`--cases` 接收多个 case，可运行 N=32/64 固定路径 overfit；需要足够多的有效缓存。
在集群分配的 GPU 上使用 `scripts/train_dual_consensus_smoke.slurm`；本地工作站可直接运行。
正式训练前还需完成真实训练集 N=1、N=32/64 overfit，以及两侧局部质量和完整 ternary DockQ 对照。

### 专家保持性检查的数值控制

`scripts/check_dual_consensus.py` 必须在 CUDA 初始化前启用确定性运算，
并预热原生/FiLM 两条路径，再分别重复前向。固定随机种子不能固定 CUDA reduction 顺序；
首次 TorchScript 执行与预热后的舍入差异也不应算作路径差异。
报告在断言前写入 `parity.json`，区分 `native_repeat`、`film_repeat` 和 `native_vs_film`，
保留原来的 `atol=rtol=2e-4`。`--trace-parity-layers` 可追踪受体/配体 embedding 和各层卷积。

MGD `1KKQ_A_E_471` 的负对照在 `rec_emb_layers.0` 就出现输入指纹完全相同、
输出不同的现象（约 `2.9e-6`）；原生路径自身重复的最终误差可达 `0.109`。
启用确定性计算并预热后，同一机器上两种路径的重复检查、两路径比较均逐位相同（最大误差 0）。
证据分别位于 `results/mgd_parity_warm_negative/parity.json` 和
`results/mgd_parity_fixed_smoke/parity.json`。这次失败是验证数值控制缺失，
不是双状态专家输入接错；它也暴露了该样本对早期微小舍入差异的放大效应。

`--parity-only` 只做前向诊断；`--nondeterministic-audit` 必须与它一起使用，
其结果明确标记为负对照，不作为通过验收的依据。
