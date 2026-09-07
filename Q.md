# 问题与修复追踪

本文件持续记录失败分析、代码缺陷、修复和验证结论；历史实验步骤详见 [todo.md](todo.md)。
新增发现使用递增编号，更新已有条目时保留原始证据与适用范围。
**当前不能收工：数值复现问题已修复，但模型装配质量仍未解决。**

## 工作规则与收工标准

- 状态：待验证、已确认、修复待验证、已修复、本案例不支持、已知限制。
- 观察到低指标不等于找到代码根因；假设必须经过固定输入对照才升级为已确认。
- 每个修复记录测试、结果路径和 commit；性能结论注明 case、seed、协议和样本范围。
- 仅使用物理 GPU 3；不覆盖原 checkpoint，不恢复已叫停的全量评测或长训练。
- 保留原始结果；新旧数值协议分目录保存，不用真实结构挑选中间步作为生产策略。
- 收工需满足：已确认代码缺陷有回归验证；主要质量问题有可复现解释；未解决项及
  所需训练/数据/架构变更明确列出。不能把“测试通过”或“没有崩溃”当成质量达标。

## 当前问题索引

| 编号 | 优先级 | 问题 | 状态 |
| --- | --- | --- | --- |
| Q-001 | P0 | 同 seed 跨进程推理漂移 | 已修复 |
| Q-002 | P0 | 原训练 torsion 归一化表没有保存 | 已知限制；新 checkpoint 已防护 |
| Q-003 | P0 | 刚体位置/朝向及双界面装配失败 | 已确认：学习速度场不足；继续定位训练原因 |
| Q-004 | P1 | 20-step 积分误差与后半程退化 | 已确认两种误差；数值改进待设计 |
| Q-005 | P1 | 配体远离 E3 时局部 cross edges 变稀疏 | 已关闭：放大机制，非已证根因 |
| Q-006 | P1 | 专家输出方向接近零引发归一化爆炸 | 已关闭：本案例假设被否定 |
| Q-007 | P1 | 新旧数值协议的报告被错误复用 | 已修复防护 |
| Q-008 | P0 | 坐标变换、速度符号或双界面组合错误 | 已关闭：真实路径对照不支持该假设 |
| Q-009 | P1 | 官方 train/val index 缺失显式 split 标签 | 已修复生成器；旧产物不回写 |
| Q-010 | P0 | Trainer 的 activation clamp 依赖入口脚本设置隐藏环境变量 | 已修复并复核 |
| Q-011 | P0 | Adapter checkpoint 未绑定冻结 DiffDock 基座身份 | 已修复；旧 checkpoint 明示不确定 |
| Q-012 | P0 | 推理 ETKDG scaffold 不在训练 native-scaffold 端点支持上 | 已确认；训练/状态设计待修 |

## Q-001：同 seed 跨进程推理漂移

- **根因：** torus 模块导入时用未固定 RNG 的 `10000 × 5001` 次 Monte Carlo
  采样生成 score normalization 表；每个进程的归一化常数不同。
- **修复：** 改为分块确定性积分，并默认启用严格确定性推理。
- **验证：** 5T35 seed 0、20 steps、epoch20 权重，两次独立进程完整 case JSON
  相等，DockQ 均为 `0.013508286730742941`。traced 与两次 untraced 末态坐标差均为 0。
  模型构建测试及 26 项定向回归通过。
- **证据：** `results/deterministic_production_gate_{1,2,3,4,5,6}/`；
  `results/epoch20_fixed_normalization_probe/`。修复提交：`7741b87`。
- **结论：** 恢复复现性，不代表恢复预测质量；只证明已测试的运行环境与样例。

## Q-002：原训练的随机表无法恢复

- **证据：** 原 epoch20 checkpoint 未记录 torsion normalization 表或版本。
- **处理：** 旧 checkpoint 加载时明确警告；新 checkpoint 保存版本，已知版本不匹配
  时拒绝加载。当前版本为 `wrapped_normal_quadrature_v1`，提交 `7741b87`。
- **结论：** 无法宣称现在是原训练数值环境的逐位重放。原权重没有改写，旧评测保留。

## Q-003：刚体摆放与双界面协调失败

- **证据：** 确定性修复后，5T35 seed 0 终态 ligand–E3 最近 Cα 距离约 `17.06 Å`，
  POI 朝向相对输入模板误差约 `178.13°`，DockQ 仅 `0.01351`。
  修复前四条选定失败轨迹也显示位置改善但朝向/另一界面仍失败。
- **路径：** `results/epoch20_fixed_normalization_probe/`；
  `results/epoch20_failure_trajectories_v2/`。
- **下一步：** 固定状态分别检查预测与监督速度的方向、尺度；先验证 Q-008，再比较
  完整模型与条件/专家对照。不能直接通过反转符号或修改损失权重“试出”测试集效果。
- **方向审计：** 三个显式 validation cases、两条独立路径、`t=0.1/0.5/0.9` 共 18
  个固定状态中，E3 translation/rotation 与目标同向比例均为 `66.7%`，POI translation
  为 `88.9%`，但 POI rotation 仅 `55.6%`，torsion 为 `61.1%`。POI rotation 平均
  cosine 仅 `0.0261`；torsion 仅 `0.1114`。实际局部一步相对零速度的 ligand 改善率
  `50%`，POI 平均改善为 `-0.00084 Å`。这是学习场质量不足的直接证据。
- **证据：** `results/epoch20_velocity_direction_audit/summary.json`；固定验证路径，不是
  benchmark。下一步用完全相同 probes 比较 epoch5 与 epoch20，判断继续训练是在改善还是退化。
- **epoch 曲线：** 相同 18 probes 的 POI rotation mean cosine 在 epoch
  `5/10/15/20` 分别为 `0.3317/0.2985/0.1567/0.0261`，持续退化；与此同时 POI
  translation 为 `0.4788/0.5554/0.5483/0.6055`，明显改善。POI 实际局部一步平均
  改善为 `-0.0494/-0.0287/-0.0150/-0.0008 Å`，主要由平移进步抵消旋转退化。
  因此不能用“继续训练会一起变好”关闭本项，需检查多任务梯度/选择指标。
- **曲线证据：** `results/epoch{5,10,15,20}_velocity_direction_audit/summary.json`；所有
  checkpoint 使用相同 cases、path seeds、times 和确定性数值协议。
- **限制：** Cα 接触是几何诊断；unbound POI 朝向误差基于预对齐输入，不是全 native
  残基映射或对称校正指标。上述为选择后的失败分析，不是无偏 benchmark。

## Q-004：后半程积分退化

- **证据：** 修复前 6Q0R seed 0 的 DockQ 从第 10 步 `0.16145` 降至末态 `0.08702`；
  5T35 seed 18 从第 18 步 `0.19760` 降至 `0.17500`。
- **下一步：** 在新数值协议下检查固定状态的局部更新与累计误差，区分速度场错误和
  积分离散误差；旧轨迹数值不能直接当成新协议结果。
- **结论：** 原四条轨迹即使按真实结构选择最佳中间步也未达 DockQ 0.23，单纯少走几步
  尚无解决质量问题的证据。
- **oracle 积分审计：** 对真实 `5Z86` 固定路径使用精确 target velocity，Euler
  `10/20/40/80` 步终点 ligand RMS 为 `0.311/0.143/0.071/0.036 Å`，POI RMS 为
  `2.382/1.150/0.560/0.276 Å`，呈一阶收敛。证据：
  `results/oracle_integration_step_audit.json`。因此 20 steps 确有可测离散误差。
- **预测场对照：** 5T35 seed 0 将模型从 20 改为 40 steps 后 DockQ 从 `0.01351`
  降至 `0.01277`、ligand RMSD 从 `5.05` 升至 `5.55 Å`。证据：
  `results/epoch20_steps40_probe/`。当前权重的学习场误差大于积分误差，不能把加步数
  作为质量修复；且训练只见过 `dt=0.05`，直接改步数会改变模型条件输入。

## Q-005：远距离回收时 E3 接触信息不足

- **证据：** 修复前 5T35 seed 0 的 E3–ligand cross edges 从 9,798 降至个位数，
  POI–ligand 末端仍约 6,300 条；采样的邻域半径本来就随时间缩小。
- **下一步：** 固定坐标与时间检查 crop、cross-edge 和专家响应，判断是否是失败的原因
  或结果。不得直接扩大半径、改变预训练图契约后混称同模型评测。
- **时序复核：** 5T35 seed 0 在 `t=1` 时已有 9,798 条 E3 cross edges，模型仍未沿
  精确方向充分回收；随着 schedule cutoff 缩小、状态没有及时上轨，edges 逐步降至
  28 条。末端并非零边，且训练和推理都使用同一 native dynamic cutoff。
- **结论：** 关闭“缺边是起始根因”的假设；它是预测偏离训练路径后的放大机制。
  在没有重训对照前不擅自扩大 cutoff 或改冻结 expert 的图分布。

## Q-006：输出头近零方向归一化假设

- **证据：** `results/epoch20_head_direction_probe/5T35_D_A_759_seed0.json` 中，
  POI rotation 最小原始范数约 `0.0630`、最大归一化增益约 `2.518`，未见近零分母。
- **结论：** 该案例不支持此假设，暂不添加任意截断；不能推广为所有样例均无此问题。

## Q-007：数值协议变更后的续跑污染

- **修复：** 推理报告/续跑匹配纳入确定性后端设置和 torus 归一化版本；旧报告缺失
  版本或设置不符时不能复用。训练 checkpoint 也记录版本。
- **验证：** `tests/test_evaluation_common.py::test_resume_rejects_changed_numerical_policy`
  及 `tests/test_torus_score_norm.py`；提交 `7741b87`。

## Q-008：速度符号、参考系与组合契约审计

- **已有检查：** `tests/test_sdpi_flow.py` 包含双界面 Cartesian tangent、步长收敛、
  oracle round trip、全局刚体等变与归一化测试；此前通过，不支持直接认定符号错了。
- **当前阅读结论：** `SDPIFlow.path` 定义 clean→prior 速度，`step` 使用负时间增量；
  POI 使用两个同源刚体 proposal 的精确组合，不应改成简单旋转向量相减。
- **真实案例验证：** 上述 18 个固定 validation 状态中，精确 target 经生产
  `SDPIFlow.step` 后相对解析 `t-dt` 状态的 Euler 坐标误差约 `1e-5–5e-3 Å`；证明
  path target、body-frame 映射和 reverse-step 符号在真实分子上相互一致。
- **结论：** 关闭“采样器符号/组合实现错误”假设。模型预测一步经常不改善，是预测场
  与已验证 target 不一致，不应通过整体反号或删除 `body.T` 修补。

## Q-009：官方 index 缺失显式 split 标签

- **问题：** 已生成的 `ternary_val.json` 虽由 val 分支产生，却没有
  `metadata.split`；严格诊断入口无法证明它是 validation，不能仅凭文件名放行。
- **修复：** `make_deepternary_index.py` 在每个输出中写入 `split=train|val`，新增
  `test_written_index_preserves_explicit_split_label`。历史 `/tmp` index 保留不回写；
  当前方向审计使用已有显式 `split=val` 的三案例归档子集。
- **结论：** 修复未来生成器的数据 provenance；不改变现有训练成员或切分结果。

## Q-010：Trainer 未落实自身记录的 activation clamp

- **问题：** `TriCoopDiffTrainer` 保存并校验 `activation_clamp=100`，但 expert 卷积实际
  从进程环境变量 `NAN_CLAMP` 读取；构造 Trainer 本身没有设置它。新方向审计作为库调用
  时没有复制训练入口的隐藏副作用，`1XK4` 固定验证路径出现非有限刚体速度。
- **修复：** Trainer 初始化在任何 forward 前将已校验的 `self.activation_clamp` 写入
  expert 当前使用的运行设置。评测 Runner 原本已执行同样配置。
- **下一步：** 重跑同一验证方向审计；若非有限值消失则关闭本项，否则继续定位具体层。
- **边界：** `torch.clamp` 不能修复已经产生的 NaN；本修复保证模型契约一致，不把非有限
  输出替换成伪造数值。
- **复核：** 完全相同的 18 个方向 probes 全部 finite；`1XK4` 第二条路径不再失败。
  `tests/test_training_contract.py::test_trainer_owns_recorded_activation_clamp_runtime` 通过。
- **结论：** 已关闭。所有 Trainer 调用现在都落实 checkpoint 所声明的 clamp；仍保留
  非有限检测，真正的 NaN 会继续硬失败。

## Q-011：Adapter checkpoint 未绑定冻结 expert 基座

- **问题：** checkpoint 只包含 TriFuse 与 LoRA adapter；原格式没有记录所依赖的冻结
  DiffDock 权重哈希。错误基座仍可能接受形状相同的 adapter，产生不可解释结果。
- **修复：** 新 checkpoint 保存稳定身份字段：DiffDock commit、冻结权重 SHA256、
  `no_torsion`；排除机器相关绝对路径。训练恢复与评测均拒绝已记录的身份不匹配。
  旧 checkpoint 明确标为 `legacy_unrecorded` 并警告，原 epoch20 文件不修改。
- **验证：** `tests/test_checkpoint_contract.py` 覆盖稳定字段、旧格式标记和错基座拒绝。
- **结论：** 关闭未来静默错配风险；无法追溯补齐旧 checkpoint 当时未保存的证据。

## Q-012：推理与训练的 ligand 内部端点支持

- **问题：** 训练 t=1 从 native ligand scaffold 施加随机 torsion；推理 t=1 则从新生成
  ETKDG scaffold 施加随机 torsion。刚体和 torsion DOF 一致不代表不可旋转的键角、环系
  与局部几何一致；这些差异无法由当前 rigid + torsion sampler 消除。
- **审计方法：** 对显式 validation cases 使用完全相同的 torsion 角和 canonical atom
  order，刚体 Kabsch 对齐后比较 inference ETKDG 与 training-native 端点的 RMSD、全原子
  pair-distance RMSE。这样不会把不同随机 torsion 或全局位姿误记为 scaffold mismatch。
- **首轮审计纠错：** 仅对两种 scaffold 施加相同 torsion *update* 后得到的 `2.62 Å`
  aligned RMSD 仍混入了不同初始绝对二面角，不能当作不可达残差。审计已改为逐键对齐
  绝对可旋转二面角，并同时报告 torsion 不可改变的 bond-length/bond-angle 误差。
- **修正结果：** 三个 validation cases × 5 seeds，在所有可旋转绝对二面角对齐到
  `3.1e-13°` 以内后，仍有 mean `2.013 Å` Kabsch RMSD、`1.389 Å` pair-distance
  RMSE；bond-length RMSE `0.109 Å`、bond-angle RMSE `10.81°`。证据：
  `results/endpoint_support_audit_v2.json`。这部分无法由当前 rigid + torsion 状态消除。
- **结论：** 已确认训练/推理 endpoint support mismatch，是 Q-003 的一个根因候选。
  修复必须让训练看到推理 scaffold 并使用可达 clean target，或扩展内部自由度；不能在
  benchmark 推理时读取 native ligand 作为捷径。旧 epoch20 权重无法通过 sampler 小改补齐。

## 更新日志

- 2026-09-07：建立问题账本，收录已确认的复现缺陷、修复证据、质量现象和待验证假设。
  当前没有宣布模型质量恢复，也没有重启大规模任务。
- 2026-09-07：发现官方 index 生成器遗漏显式 split 标签，登记 Q-009 并修复生成器。
- 2026-09-07：方向审计暴露 Trainer 的 activation clamp 隐式入口依赖，登记 Q-010 并修复。
- 2026-09-07：发现 adapter checkpoint 未绑定冻结 expert，登记 Q-011 并加入身份校验。
- 2026-09-07：完成 18 个真实 validation 路径方向对照，关闭 Q-008/Q-010，将 Q-003
  收窄为学习速度场不足，并开始 epoch5→20 同路径进展审计。
- 2026-09-07：补齐 epoch5/10/15/20 同路径曲线；确认 POI translation 学习改善但
  POI rotation 方向持续退化，下一步转向多任务梯度与 checkpoint 选择审计。
- 2026-09-07：登记 Q-012，开始定量检查推理 ETKDG 与训练 native-scaffold 端点失配。
