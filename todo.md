集群：ssh.cn-zhongwei-1.paracloud.com

在自己的电脑上执行：

  scp scwb792@<集群登录地址>:/data/home/scwb792/run/mwj/CoopDiff/results/
  tricoopdiff_phase1_joint.tar.gz .

  再下载校验文件：

  scp scwb792@<集群登录地址>:/data/home/scwb792/run/mwj/CoopDiff/results/
  tricoopdiff_phase1_joint.tar.gz.sha256 .

  进入下载目录后校验：

  sha256sum -c tricoopdiff_phase1_joint.tar.gz.sha256

下载执行回复（已完成）：SSH 端口 22 超时后，改用你提供的
[Hugging Face 仓库](https://huggingface.co/chiww6/CoopDiff-checkpoints/tree/main/tricoopdiff_phase1_joint)，
固定 revision `ee45e6740443e56ab7511ac320f26cc1783dbff5`。
`best_model.pt`、epoch 5/10/15/20 权重和 `config.json` 已下载至
`/tmp/pymp-f2c2db1a/checkpoints/phase1_joint/`，5 个权重均通过仓库 LFS SHA256 校验。
`best_model.pt` 与 `checkpoint_epoch20.pt` 均记录 epoch=20、global_step=11740，
checkpoint_version=17、architecture=trifuse_v13、generative_process=sdpi_flow_v1。
结论：下载阻塞已解除；这是训练计划 100 epoch 中的 20 epoch 阶段性权重，不是最终模型。
下一轮诊断以 `checkpoint_epoch20.pt` 为基线，保留原文件不覆盖。

# FiLM＋LoRA 条件流排查清单与逐项结论

## 本轮诊断范围

分支：`diagnose/film-lora-flow`。诊断代码提交：`d1c085a`、`80aed04`；运行指南：`3c33cc7`。
全部 GPU 工作只使用物理 GPU 3。本轮从冻结 DiffDock 预训练专家与新初始化的
TriFuse/LoRA 开始，以下首轮结果未使用你之前训练得到的 checkpoint，不能直接归因于旧模型。
现已取得 epoch 20 checkpoint，第二轮模型对照已执行；新增结果在文末逐项单列，
首轮记录保留用于追溯，当前状态以第二轮结果为准。

真实训练样本为 `3EXE_B_C_TPP`、`6JLL_A_b_LMT`、`5Z86_A_M_PGV`，
分别包含 8、16、42 个 torsion；均取自官方 cluster 训练划分，并排除了 benchmark 同 PDB。
“保留路径”指这些训练结构上未用于优化的随机路径，不是独立结构验证集。

总体结论：固定目标能拟合，模块更新正常；随机路径效果不一致，条件收益尚未证明。
已推进至小样本条件消融，未进入全量训练。下文保留原检查要求，并逐项补充回复、结论与未完成工作。
详细原始结果见 [诊断报告](results/flow_diagnostic_report.md)；`results/` 为本地产物，不随 Git 提交分发。

## 原检查要求与执行回复

针对我们当前的 FiLM＋LoRA 条件流模型，我会按下面的优先级查。这里 P0 指“不排除
  它，继续大训练可能没有意义”。

  先明确：随机采样时间和路径时，单步 loss 波动不等于训练失败；条件流匹配的 MSE 也
  不保证能降到零。 必须先用固定诊断样本判断。

  ### P0：目标、计算或梯度有问题

#### 1. 输入与速度目标不对应

  重点检查：

  - 两侧参考系、旋转方向及反向积分符号。
  - POI 分支目标是否正确映射到全局坐标。
  - torsion bond 顺序、旋转侧 mask、自旋修正是否对应。
  - 训练 target 的归一化与推理速度还原是否一致。

  现有测试覆盖了这些关键路径，降低了风险，但不能替代对实际训练 batch 的检查。

  最强诊断：固定一个结构、一个时间、一次随机路径采样，反复训练同一个输入—目标对。
  如果这样都不能明显降损失，先别讨论数据不足。

执行回复（已完成单样本检查）：对 `3EXE` 固定结构、`t=0.5` 和一次 coupling，
关闭 dropout，使用生产损失训练 200 步。loss 从 `7.5513077` 降至 `0.0000514633`。
实际样本在 `t=0.1/0.5/0.9` 的反向积分坐标检查均通过，步长减半后误差约缩小至 1/4，
最终最大坐标误差约 `1e-5 Å`。归一化/速度还原与固定 coupling 损失一致性测试通过。

结论：已检查样本未发现参考系、积分符号或目标对应错误；固定目标可拟合。
这不能排除其他结构或历史 checkpoint 的问题。

后续：在之前训练的 checkpoint 上复查，并优先检查第 8 项的退化样本。
证据：[固定拟合](results/flow_diagnostic_fixed_3EXE_run4/summary.json)、
[实际坐标检查](results/flow_diagnostic_fixed_3EXE_run4/tangent_audit.json)。

#### 2. 参数没有获得有效更新

  检查的不只是 requires_grad=True，而是：

  - FiLM、bridge、torsion coordinator、两侧 LoRA 的梯度和实际参数增量。
  - 是否存在意外 detach、错误的 optimizer 参数组。
  - LoRA warmup 前后是否真的切换。
  - activation clamp 和梯度裁剪是否长期限制更新。

  尤其当前前 100 steps 默认不训练 LoRA，不能把 warmup 阶段直接当成全部适配能力已经
  开启。

执行回复（模块级检查通过，裁剪影响待查）：记录了 FiLM、bridge、torsion coordinator
和两侧 LoRA 的梯度、绝对及相对参数增量。前三类模块均有实际更新；两侧 LoRA 在
步骤 0–99 增量为零，第 100 步开启后增量范数分别约为 `0.00574/0.00579`。
逐层 activation clamp 比例也已记录，但未做关闭/改变 clamp 阈值的对照。

结论：当前运行未发现模块级梯度断链或 warmup 不切换；非零模块增量不等于每个参数
都获得有效梯度，也不能单凭 `requires_grad` 或 optimizer 成员检查排除全部问题。
持续梯度裁剪是否限制学习仍未解决，见第 4 项。

后续：复查旧 checkpoint 的 optimizer 状态及实际恢复配置，再做 clamp/裁剪对照。
证据：[逐步梯度、更新与 activation probe](results/flow_diagnostic_fixed_3EXE_run4/steps.jsonl)。

#### 3. 某个分量尺度异常，压住其余任务

  我会优先看 torsion 残差经过归一化后的幅度。原始 score 单位下的残差再除以较小的归
  一化因子，可能形成过大的预测和梯度。

  要分别记录两侧 translation、rotation、joint torsion、auxiliary torsion 的：

  - loss；
  - 预测／目标 RMS；
  - 梯度贡献；
  - 不同时间段的分布。

  总 loss 一个数不够定位问题。

执行回复（已完成小样本分项统计）：日志包含两侧平移/旋转、joint/auxiliary torsion
loss、预测/目标 RMS，并在指定步骤记录各分项未加权梯度贡献；auxiliary torsion 在
总目标中的权重为每侧 `0.125`。随机路径按 0.2 宽度时间段统计。
`3EXE` 初始 `t=0.1/0.9` 的 joint torsion 预测 RMS 为 `0.2003/0.5623`，目标均为 `0.6513`。
第一组随机路径平均两侧平移 loss 为 `1.9386/1.1537`，joint torsion 为 `0.3319`。

结论：这些探针未显示 torsion 普遍压住其他任务；较大损失主要来自刚体分量，不能据此
排除全数据或训练后局部状态的 torsion 异常。分项梯度范数不可直接相加当作总梯度范数。

后续：对退化路径逐分量复查，并扩大时间/结构覆盖。
证据：[时间 0.1](results/flow_probe_3EXE_t01/steps.jsonl)、
[时间 0.9](results/flow_probe_3EXE_t09/steps.jsonl)、
[随机路径分组](results/flow_diagnostic_random_3cases/breakdown.json)。

  ### P1：目标能拟合，但优化或条件化没学起来

#### 4. 学习率和裁剪不合适

  TriFuse 与 LoRA 的学习率不同，新速度任务也不同于原来的 score 任务。可能是更新过
  小，也可能是震荡。

  先看梯度裁剪前范数、实际参数更新比例，再调整；不能一见平台就加大学习率。

执行回复（已观测，尚未做超参数对照）：使用 TriFuse LR `1e-4`、早层 LoRA LR `5e-6`、
晚层 LoRA LR `1e-5`、全局梯度裁剪阈值 `1.0`，与 Phase 1 wrapper 一致。
两组随机路径实验的全部更新步骤均触发裁剪，裁剪前范数中位数分别为 `23.04/35.36`；
固定目标在接近零损失后也出现小幅震荡。

结论：持续强裁剪值得优先对照，但尚不能认定它或学习率就是根因。
AdamW 的实际更新不能由梯度裁剪比例直接推算；暂未调整生产学习率或裁剪阈值。

后续：固定相同初始化、训练路径、保留路径和步数，单独改变学习率或裁剪阈值。
证据：[第一组统计](results/flow_diagnostic_random_3cases/breakdown.json)、
[第二组统计](results/flow_diagnostic_condition_3cases/breakdown.json)。

#### 5. 从去噪专家迁移到速度预测器比较困难

  FiLM 能改变方向和幅度，这不是问题本身。但“有能力表达”不等于“从当前预训练状态容易
  优化到”。

  可以分阶段诊断：先只拟合平移，再加旋转，最后加 torsion。这是定位哪一项困难，不是
  把最终模型永久拆开训练。

执行回复（部分完成）：联合目标的单样本固定路径可以拟合；已做两组 3 结构随机路径训练。
seed=42、200 步的保留路径平均 loss 为 `5.30510 → 3.42927`；seed=43、150 步为
`4.78759 → 5.90245`。两组 seed、保留路径与步数不同，不是严格控制变量的 seed 对照。
**尚未执行“仅平移 → 加旋转 → 加 torsion”的分阶段拟合。**

结论：不能认定从 score 专家迁移到速度预测器在表达上不可行；随机路径的学习收益
尚不稳定，也不足以定位究竟是哪一任务造成优化困难。

后续：在同一失败路径上按原建议做分阶段对照；本轮未永久拆分生产目标。
证据：[200 步结果](results/flow_diagnostic_random_3cases/summary.json)、
[150 步结果](results/flow_diagnostic_condition_3cases/summary.json)。

#### 6. 另一个界面的条件没有被有效利用

  FiLM 的设计目的明确，但模型仍可能忽略条件，或接收到不易利用的信息。

  固定本侧输入，改变另一侧状态，检查：

  - FiLM 参数是否变化；
  - 输出速度是否变化；
  - 正确条件是否比打乱条件带来更低验证损失。

  仅仅“输出会变”还不足以证明条件有用。

执行回复（已完成一个方向的条件消融）：第二组训练后，固定 expert 输入和原目标，
只替换 TriFuse 看到的 POI 状态，保持共享 ligand 和 E3-L 输入不变。
在 3 结构 × 3 时间点上，E3-L FiLM RMS 变化约 `0.000077–0.000221`，输出速度也变化。
正确/打乱条件的平均 E3-L 平移加旋转 loss 为 `1.90937/1.90977`，仅 1/9 探针正确条件更好。

结论：条件通路有响应，但本次短训练未证明稳定收益；不能将此结论推广成“整个模型
忽略条件”。本实验仅覆盖 E3-L 的 TriFuse 通路，未覆盖反方向或 bridge 的全部条件通路。

后续：在旧 checkpoint 上重复，增加独立路径、独立验证结构与反方向消融。
证据：[条件干预、FiLM/速度变化与 loss](results/flow_diagnostic_condition_3cases/summary.json)。

  ### P2：训练目标本身有噪声，或数据覆盖有问题

#### 7. 条件流匹配存在不可约的目标方差

  模型看不到生成当前状态时使用的完整随机路径信息。不同路径可能在相近状态给出不同速
  度，MSE 最优解学的是条件平均：

  [
  v^*(x,t,c)=\mathbb E[u\mid x,t,c]
  ]

  因此，随机路径训练 loss 不归零可以是正常现象。要结合固定路径诊断、验证 loss 和生
  成结构质量判断，不能追求零 loss。

执行回复（尚未验证噪声下限）：已有固定目标近零 loss 与随机路径非零 loss 的对照，
但尚未估计相近 `(x,t,c)` 下的条件目标方差，也未用这些诊断权重评估生成结构质量。

结论：随机路径 loss 不归零本身不等于训练失败；目前同样没有证据把剩余 loss
归因于不可约噪声。优化不足、条件利用不足及路径覆盖不足仍是候选解释。

后续：先完成第 4–6 项的受控实验，再增加路径统计及独立结构的生成质量评估。
现有证据只限于第 1、5 项，不存在已完成的噪声下限或 benchmark 结论。

#### 8. 数据混杂或少数异常样本主导

  例如原子映射错误、缺失结构、不同数据来源，以及 PROTAC／MGD、配体长度和 torsion
  数量带来的任务差异。

  按 case、任务类型、时间段和可旋转键数量拆开统计，查看是不是少数样本贡献了大部分
  损失。

执行回复（已完成 3 case 分层，未完成全数据审计）：已有按 case、时间段、torsion
数量及 warmup 状态的统计；未做完整 PROTAC/MGD 类型、配体长度或全池异常归因。
第二组 `5Z86_A_M_PGV, t=0.1` 保留路径 loss 为 `4.1825 → 15.9726`，
最终 POI 平移/旋转 loss 为 `7.0453/6.8613`；该 case 训练梯度范数中位数约 `371.12`。
数据流水线 10 项检查通过，本地压缩包完整性检查通过；这不等于全量结构通过检查。

结论：已定位一个值得优先复现的退化 case/时间点，但不能仅因 loss 高就判定原子映射
错误、数据损坏或直接剔除该样本。当前样本量也不足以证明少数异常样本主导全数据。

后续：固定该保留路径检查 POI 目标、几何、梯度与裁剪，再扩展训练内验证结构；
全量训练前仍须通过正式数据门禁。
证据：[逐探针前后对照](results/flow_diagnostic_condition_3cases/summary.json)、
[case/时间/torsion 分组](results/flow_diagnostic_condition_3cases/breakdown.json)。

  ———

  我的实际排查顺序：固定单个输入—目标对 → 检查各模块真实更新 → 拆分各项尺度 → 固定
  少量结构、随机路径训练 → 条件消融 → 扩到全数据。

  另外，推理只用 20 步通常不是训练 loss 下不来的直接原因。那主要影响采样精度，不应
  先拿增加推理步数来解决训练优化问题。

## 实际执行顺序：每步回复与当前状态

| 步骤 | 状态 | 回复与结论 |
| --- | --- | --- |
| 固定单个输入—目标对 | 单样本通过 | loss 明显下降，坐标切向检查通过；见第 1 项。 |
| 检查各模块真实更新 | 模块级通过 | 更新和 warmup 正常；裁剪影响未排除，见第 2、4 项。 |
| 拆分各项尺度 | 小样本已完成 | 目前以刚体损失为主，未见 torsion 普遍主导；见第 3 项。 |
| 固定少量结构、随机路径训练 | 已执行，收益未稳定 | 两组保留路径 loss 一降一升，不能据此宣布泛化门禁通过；见第 5 项。 |
| 条件消融 | 单方向已执行，收益未证明 | 输出有变化，损失收益缺乏一致性；见第 6 项。 |
| 扩到全数据 | 未启动 | 小样本随机路径与条件收益问题尚未解决，先做受控复查及正式数据门禁。 |

关于推理步数的回复：本轮没有增加推理步数或用 benchmark 调参；也未执行 20 步与
更多步数的采样精度对照。当前没有证据将训练优化问题归因于推理步数。

## 第二轮：原始 epoch 20 checkpoint 的逐项回复与结论

本轮恢复原权重、optimizer 和 global_step=11740。每个对照单独恢复同一 checkpoint，
没有覆盖原模型，也没有改变生产学习率或裁剪参数。GPU 仅为物理 3。
运行命令须加 `env -u GCC_ROOT`，避免继承的无效 GCC 路径阻止 Triton backward 编译。
以下内部验证来自当前仓库官方 val clusters 的 `5OL2_C_F_FAD`、`6U9D_U_V_60G`、
`1XK4_F_J_FLC`，每结构 3 条路径 × 3 个时间点，共 27 探针；不是 benchmark。
原训练索引清单未随 checkpoint 提供，因此尚不能独立核验原训练使用的划分完全相同。

### 1. 输入与速度目标对应：已执行，所测路径通过

回复：历史失败路径回放最大 loss 差 `4.63e-5`，目标 RMS 一致，符合 CUDA 浮点容差。
原模型在 `5Z86, t=0.1` 的 loss 为 `4.02010`，不是首轮短训练后的 `15.9726`。
该结构 `t=0.1/0.5/0.9` 实际坐标切向检查均通过，半步误差最大约 `3.18e-5 Å`。
结论：没有复现目标映射/反向符号错误；不能把首轮退化归因于用户的训练模型。
证据：`results/epoch20_diagnostic_run3/{summary.json,tangent_audit.json}`。

### 2. 模块更新与 warmup：已执行，模块级通过

回复：恢复后 LoRA 已启用，没有重启 100-step warmup。FiLM、bridge、torsion coordinator
及两侧 LoRA 均有非零梯度和实际参数增量。失败路径初始总梯度范数 `16.0556`，
FiLM 为 `15.3711`；两侧最大逐层 clamp 比例约 `3.14%/7.98%`。
结论：未发现模块级 detach 或 optimizer 漏参；存在裁剪，但裁剪存在本身不是错误。
证据：`results/epoch20_diagnostic_run3/all.jsonl`，clamp 对照见第 4 项。

### 3. 分项尺度：已执行，所测退化以刚体为主

回复：原模型该路径 POI 平移/旋转 loss 为 `1.28741/1.51304`，joint torsion 为 `0.39821`。
未加权分项梯度范数：E3 平移/旋转 `3.3195/1.0019`，POI 平移/旋转 `13.7254/6.0312`，
joint torsion `0.6185`，两侧 auxiliary torsion `0.5345/1.1417`（实际各乘 0.125）。
结论：该路径不支持“torsion 梯度压住刚体”的假设；不同分量梯度不能直接相加。
证据：`results/epoch20_gradient_probe/summary.json`；27 条训练结构探针分项见第二轮 summary。

### 4. 学习率、裁剪、clamp：受控实验已执行，保留原配置

回复：同一固定路径各训练 40 步，基线联合 loss `4.02010 → 0.72713`；
学习率减半终值 `0.96152`；clip=10 终值 `0.25947`；clamp=200 为 `4.87630 → 0.91357`。
改变 clamp 会改变初始函数，不能只按下降幅度比较。
进一步固定 100 条随机训练路径、dropout seeds、原权重和 optimizer：
clip=1 的独立结构验证 `3.14563 → 3.36449`；clip=10 为 `3.14553 → 3.72697`。
结论：放宽裁剪能改善局部拟合，但没有改善这组验证；不据此修改生产配置。
此为 3 个训练结构上的短实验，不是全数据最优超参数搜索。
证据：`results/epoch20_diagnostic_run3/summary.json`、`results/epoch20_random_clipping/summary.json`。

### 5. 平移／刚体／全目标拟合：已执行，表达能力门禁通过

回复：分别从同一原 checkpoint 开始，关闭 dropout，各训练 40 步：
仅平移 `1.69080 → 0.00753`；加旋转 `3.52145 → 0.30474`；全目标 `4.02010 → 0.72713`。
全目标末尾主要剩余 E3 rotation `0.26059`、joint torsion `0.34870`。
同一 27 条独立验证路径，epoch 5/10/15/20 loss 分别为
`3.82851/3.31656/3.34178/3.14560`；最后一轮较 epoch 5 约降低 18%。
结论：训练确实学到了东西，不支持“速度任务完全学不动”；仍不能保证生成质量。
证据：`results/epoch20_diagnostic_run3/summary.json`、
`results/epoch20_validation_conditions/epoch_comparison.json`。

### 6. 双向条件消融：已执行，有响应但收益未证明

回复：固定本侧 expert 输入、共享 ligand 和监督目标，仅替换 TriFuse 的对侧坐标。
反方向干预先对齐公共 ligand frame，并断言本侧几何不变。
独立验证 27 探针的正确/打乱条件刚体 loss：E3-L `1.518705/1.512688`，
POI-L `1.247008/1.247022`；正确条件更好分别为 `12/27`、`11/27`。
两方向 FiLM 均有约 `4e-5–2.3e-4` RMS 响应，输出速度也改变。
结论：没有稳定的正确条件优势；极小差值不能排除数值噪声。此干预只针对 TriFuse，
并未删除 bridge 等全部条件通路，不能据此宣称整个模型无条件化。
证据：`results/epoch20_validation_conditions/summary.json`。

### 7. 条件目标方差：端点检查已执行，整体噪声下限未确定

回复：每个训练结构抽取 128 个 coupling，在 `t=0` 断言模型状态输入完全相同；
归一化且按生产权重加权的经验目标方差为 `2.96421/2.95336/3.70654`。
结论：同一干净端点确实对应多种速度目标。但连续时间采样命中精确端点的概率为零，
该数值不能作为整体训练 loss 下限，也不能解释所有剩余误差。
不把“已做噪声诊断”写成“已证明训练到达不可约下限”；生成质量另用 benchmark 检验。
证据：`results/epoch20_diagnostic_run3/endpoint_target_variance.json`。

### 8. 异常样本与全池数据：失败路径复查与全池拆分门禁完成

回复：`5Z86` 目标/坐标检查通过，原模型 loss 正常，且固定拟合可显著降低。
不能将首轮短训练的高梯度当作损坏标签而剔除该 case。
从已通过 gzip 完整性检查的压缩包提取官方 7,198-case 联合池，无缺失。
现行 index builder 完整运行通过：训练 6,834 accepted / 110 rejected，
验证 253 accepted / 1 rejected；111 个拒绝项全部为协议预期的不连通配体，
无其他 materialization error，合计 7,087 usable cases。
索引：`/tmp/pymp-f2c2db1a/epoch20_evaluation/indexes/ternary_{train,val}.json`。
验证 253/253、训练 6,834/6,834 均通过有限坐标与双界面接触审计，无失败。
证据：`results/epoch20_input_audit_{train,val}.json`。
审计显式使用 `--min-ligand-heavy-atoms 1`，不套用旧的 20-heavy-atom 过滤器。
联合池是 PROTAC/MGD 两训练来源的交集，不能把来源目录直接当作互斥任务标签。
本轮 loss 统计只覆盖 6 个结构，不声称已完成全池神经网络 loss 归因。

## 后续生成质量评估（用户追加）

上述 8 项均已给出执行回复与结论；未证实的假设保留限制，不标作通过。
数据门禁完成，现进入生成质量评估，使用未经诊断更新的原始 `checkpoint_epoch20.pt`，
按 `doc/Phase 1 数据协议.md` 评估 PROTAC22 unbound × 40 seeds、MGD94 bound × 1 seed，
SDPI 初始化、20 步、seed=0 起始，仅物理 GPU 3；先做单例完整原子 DockQ 门禁。
权重、索引哈希和逐 case 输出须保留。PROTAC 的 best 只称 oracle-best，
当前不存在经监督的模型排名 top-1；失败 case、无效 walker 与样本分母必须公开。
本轮不继续训练剩余 80 epochs，也不使用 benchmark 选择参数或 checkpoint。

评测接口修正：PROTAC unbound 文件的链名/残基编号与 native 不一致。
严格复用 `DeepTernary/predict_cpu.py::predict_one_unbound` 的重建方式：
先从 unbound 输入与预测坐标求刚体运动，再将其作用到同目录预对齐的 bound 全原子模板，
调用 `cal_dockq`。模板仅用于打分，不进入模型特征或初始化；
通过 `evaluate_index.py --metric-template-root /tmp/coopdiff_protac/output/protac22` 显式开启，
模板内容纳入 input_signature，run_config 记录 reconstruction 方式。
PROTAC 不注入 DeepTernary 的双端点二元先验，因此历史 DeepTernary 分数不是同先验对照。
每条 seed 的指标、计算失败原因和完整原子 DockQ 成功数均保留。

代码提交：`14a0c02`（真实 checkpoint 受控诊断）、`a5ce49d`（官方输入准备与逐 seed 记录）。
回归：model-build 全部通过；数据/evaluation helper 共 21 项通过；逐 seed 与
unbound-template 刚体重建测试通过。`.gitignore` 用户原有改动未包含在提交中。

### 推理门禁与当前进度

- PROTAC 首次门禁发现 Cα alternate-location 数量不一致；改用与 DiffDock 相同的
  ProDy/get_coords 选择后解决。失败产物保留在 `results/epoch20_protac_gate{,_v2}/`。
- 共享盘 `DockQ/fnat` 无法执行；改用已有 `/tmp/pymp-f2c2db1a/DeepTernary` 副本，
  `dockq_util.py` 与 `fnat` SHA256 均与仓库一致，分别为
  `22389a7e0743119b969f8c3788203dbc2c860274198c7cf3ac9adb7795b3b7fc`、
  `6ce16a2241b4dd1d44486343a126ecdb010d09b978f6511282d4420f091218fa`。
  通过 `--deepternary-root` 显式指定，报告记录两项哈希。
- 完整原子门禁通过：`5T35_D_A_759` seed 0 DockQ 约 `0.0129`；
  `3JRQ_B_A_A8S` seed 0 DockQ 约 `0.0186`，各 1/1 有效且完成精确打分。
  证据：`results/epoch20_protac_gate_v3/`、`results/epoch20_mgd_gate/`。
- 已启动 PROTAC 官方 22×40、MGD 官方 94×1，20 步，物理 GPU 3。
  PROTAC 分为 8/7/7-case 三片，输出 `results/epoch20_benchmarks/protac/shard_{0,1,2}/`；
  MGD 输出 `results/epoch20_benchmarks/mgd/`。日志为
  `results/epoch20_protac_shard{0,1,2}.log` 与 `results/epoch20_mgd.log`。
  所有 shard 同时只占用 GPU 3，不使用其他 GPU。
- 修正批处理状态：完整原子 DockQ 缺失不再写成成功；逐 seed 报告先保存再标记失败。
  新增官方集合汇总器，缺失 case 和失败 seed 计入分母，不能只汇总幸存样本。
  最终汇总尚未完成，当前单例数值不能当作 benchmark 结果。
- 后续 SDF 预检发现 `6HR2_F_E_FWZ` 含 58 个显式 H、`7KHH_C_D_WEP` 含 4 个。
  打分 SDF 模板改为去 H 且保留重原子顺序，与 DiffDock 输出一致；新增重原子顺序测试通过。
  受影响的 shard 1 已恢复运行，跳过已完成的首例；只丢弃当时未完成的第二例轨迹，
  重用同一 seed 与原权重，不修改输入文件或采样算法。其他分片继续，已完成指标不受影响。
- 用户查看时的阶段快照：PROTAC 完成 3/22，120/120 seeds 精确打分，
  mean oracle-best DockQ `0.15659`，ensemble-mean DockQ `0.02760`；
  MGD 完成 29/94，29/29 seeds 精确打分，mean DockQ `0.02514`。
  已完成 case 中 DockQ≥0.23 均为 0；未完成 case 不能据此判为失败。
  快照：`results/epoch20_partial_{protac,mgd}.json`，`complete_all_cases_and_seeds=false`。
  全部评测仍在后台运行，最终指标待全量汇总。

## 用户叫停全量评测：转入失败轨迹分析

用户明确要求停止后，已向四个已核实的评测进程发送 SIGTERM，并确认全部退出；
没有删除文件或终止其他用户的进程。停止时保留 PROTAC 6/22 cases（240 seeds）、
MGD 64/94 cases（64 seeds）；已完成部分 mean oracle-best PROTAC DockQ `0.15032`、
MGD mean DockQ `0.02163`，均无 DockQ≥0.23 的 case。
这是主动中止的部分评测，不是完整 benchmark；未完成 case 不算模型失败。

接下来仅物理 GPU 3 上运行 4 条失败轨迹复现：`5T35_D_A_759` seeds 0、18
（18 为已完成 40 次中的 oracle-best），`3JRQ_B_A_A8S` seed 0、
`6Q0R_B_D_O6M` seed 0（停止时已完成 MGD 中最好者）。
不改变模型权重、初始化或 20-step sampler；逐状态保存完整原子 DockQ、
配体放置/内部构象 RMSD、POI 刚体位姿、两界面 Cα 接触与模型速度统计。
初始化状态作为“同 seed、不积分”的对照，不混称冻结 DiffDock 对照。
原始数组保存在 `.npz`，每步指标与 trace 保存在 `.json`，
输出目录 `results/epoch20_failure_trajectories/`，日志 `results/epoch20_failure_trajectories.log`。
首轮严格复现门禁因 seed 0 的 DockQ 差 `0.001166` 停止，未将其当作成功复现。
因此增加同初始化、关闭 trace 的两次重跑以及确定性算法对照，结果如下。

### 失败轨迹分析结果（已完成，未修改生产模型）

| Case / seed | 初始 DockQ → 最终 | 配体质心误差 Å | POI 质心误差 Å | POI 朝向误差 ° |
| --- | --- | --- | --- | --- |
| 5T35 / 0 | 0.01103 → 0.01384 | 39.08 → 31.40 | 43.85 → 40.06 | 149.83 → 171.92 |
| 5T35 / 18（oracle-best seed） | 0.15218 → 0.17500 | 17.55 → 5.34 | 34.55 → 34.31 | 172.84 → 154.55 |
| 3JRQ / 0 | 0.01670 → 0.01864 | 46.12 → 34.04 | 65.95 → 63.11 | 137.16 → 142.16 |
| 6Q0R / 0 | 0.03889 → 0.08702 | 38.73 → 22.25 | 23.93 → 5.84 | 137.16 → 169.73 |

误差定义：配体质心相对于原参考配体；POI 位姿相对于原输入模板，unbound 是预对齐输入，
不能把它的角度误差冒充基于完整 native 原子映射或对称校正的朝向指标。
DockQ 始终使用完整原子标准接口和先前相同的模板重建协议。

1. **不是完全没有运动，但正确三元装配没有形成。** 5T35/18 的配体位置明显改善，
   POI 仍偏移约 34 Å、旋转偏差约 155°；6Q0R 的 POI 质心接近了，朝向却更差。
   配体独立对齐 RMSD 分别 `5.01→4.87`、`5.07→4.30`、`2.24→1.61`、`2.70→2.33 Å`，
   所以问题不只是内部构象，刚体摆放和两界面协调同样明显失败。
2. **碰撞与错误界面仍然存在。** 5T35/0 末态 ligand–E3 最小 Cα 距离约 18.51 Å，
   ligand–POI 最小 Cα 距离仅 0.90 Å；5T35/18 ligand–E3 最小距离 0.87 Å，
   蛋白 Cα 距离小于 3 Å 的原子对从 34 增至 50。Cα 检查仅是几何诊断，
   不是替代全原子 clash score，也不是可直接用于筛选的新排名函数。
3. **后半程会退化，但改成少步数不足以解决。** 6Q0R 第 10 步 DockQ 达 0.16145，
   第 20 步退回 0.08702；5T35/18 第 18 步为 0.19760，终值为 0.17500。
   四条轨迹即使按真实结构挑最佳中间步，也均未到 0.23；不据此用 benchmark 调步数。
4. **确定了部分轨迹的数值复现问题。** 独立控制实验中，5T35/0 三次初始化坐标完全一致，
   默认计算的最终 ligand RMSD 相对 traced run 差 3.35–5.21 Å，POI 差 4.94–6.62 Å。
   使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `torch.use_deterministic_algorithms(True)` 后，
   traced + 两次 untraced 的末态坐标完全一致，DockQ 均为 `0.01293295`。
   其他三条轨迹的重复末态误差约 `1e-5–3.4e-4 Å`，说明并非所有 seed 都同样敏感。
   这支持非确定性数值计算在部分轨迹中被放大，但尚未定位具体算子；
   不归咎于单一 scatter、模型层或 integration step。确定性只恢复复现性，未恢复质量。
5. **远距离回收的线索。** seed 0 的 E3–ligand cross edges 随时间从 9,798 减到个位数，
   末端 POI–ligand 仍约 6,300 条。半径本来就随 schedule 缩小，所以这里只能说：
   配体尚未靠近正确界面时，E3 的局部接触信息已经很稀疏；因果关系需固定状态对照。

证据：`results/epoch20_failure_trajectories_v2/summary.json` 与同目录每条 `.json/.npz`；
`results/epoch20_replay_control/`、`results/epoch20_replay_deterministic/`。
四条分析轨迹末态 DockQ 与历史结果误差分别为 `5.85e-4`、`5.55e-7`、`6.71e-7`、`4.23e-6`；
seed 0 必须同时参照上述坐标差异，不能仅凭 DockQ 容差称其精确复现。
新脚本 `scripts/analyze_failed_trajectories.py` 支持 `--limit 1 --deterministic`，
`tests/test_failure_geometry.py` 验证内部形状与刚体位姿误差的分离。

当前结论：停止追加大规模评测/训练是合理的；下一步优先在确定性诊断设置下，
检查固定状态的速度方向和 POI 刚体目标，再做同起点的条件/专家对照。
本轮没有修改生产 sampler、checkpoint、学习率或启用碰撞过滤，也未证明架构必然不可行。
