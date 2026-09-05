# 双 DiffDock 界面坐标架构

## 架构结论

TriCoopDiff 以 E3 为固定全局参考系，扩散状态不是“全局 Ligand 加独立 POI 位姿”，而是两个二体界面的相对状态：

- `T_E3L`：Ligand 相对 E3 的刚体位姿；
- `T_POL`：Ligand 相对 POI 的刚体位姿；
- `chi`：Ligand 内部 torsion。

同一个冻结 DiffDock 基座以两个 branch-specific LoRA/FiLM 分支运行。`sub1` 对 `T_E3L` 去噪，`sub2` 对 `T_POL` 去噪。两者不是待插值的 Ligand score，而是各自界面的完整 proposal。

## 一步更新

每个反向扩散步按以下顺序执行：

1. 两个 expert 在同一个旧 Ligand 状态上预测 score。
2. 从该旧状态分别生成 E3–Ligand proposal `L1` 和 POI–Ligand proposal `L2`。
3. Ligand 采用 `L1`；求 `L2 → L1` 的精确刚体变换，并把同一变换施加给旧 POI。
4. TriFuse 按可旋转键融合两侧 torsion score，并将唯一的共享 torsion 更新应用一次。
5. 同步重建 POI local frame、receptor crop、cross-edge 和两个 expert 图。

旋转必须使用 SO(3)/SE(3) 群组合；禁止把 rotation vector 简单取负来近似逆变换。架构中不存在 residual-gated merge，也不存在独立预测 POI 运动的 TriFuse SE(3) head。

## TriFuse 职责

TriFuse 输出三类信息：

- E3、POI、Ligand 类型感知 token，以及 E3–L、POI–L、E3–POI 距离/接触编码；
- 两个 expert 各层独立的 FiLM 参数；
- 基于两侧 per-bond score 和三体 context 的 torsion 权重与 residual；
- 不改变坐标动力学的 scalar auxiliary heads。

TriFuse 不合并 rigid score、不输出 Ligand/POI SE(3) residual，也不替代 DiffDock 完成全局刚体装配。

## 训练与推理对齐

训练分别对 `T_E3L` 和 `T_POL` 使用 native DiffDock forward noise。加噪后的两个相对刚体通过精确 SE(3) 组合重建唯一且一致的全局 Ligand/POI 坐标，然后构建两个二体图。每个 expert 接收自己界面的真实时间和噪声目标，禁止独立扰动全局 POI 后让 `sub2` 复用 `sub1` 的时间条件。

推理同样维护这两个界面状态。初始化策略只能决定两个界面 proposal 的起点，不能改变状态定义或绕过精确坐标重建。

## Checkpoint 边界

新架构 checkpoint 必须同时满足：

- `checkpoint_version == 7`；
- `architecture == "trifuse_v11"`；
- 包含与当前 branch-specific FiLM/LoRA 配置匹配的状态。

加载器必须拒绝 version 13 及更早 checkpoint；不提供静默迁移路径。旧 checkpoint 缺少 score control 或 LoRA warmup 契约，只能留作历史结果。

Expert 每层卷积采用固定 activation clamp 100，并将该值写入 checkpoint 的 `runtime_config`。这是冻结 DiffDock/LoRA 路径的显式数值策略；它不能替代界面坐标加噪，也不得在训练与评估之间改变。

## 验收门禁

- SE(3) 正反变换、组合和全局等变性测试通过；
- 两个 rigid proposal 必须从同一个旧 Ligand 生成；
- 每一步两个界面 proposal 与重建后的全局坐标一致；
- `L2 → L1` 变换前后 POI–Ligand 跨界面距离保持不变；
- 两分支输出有限，TriFuse 协调后的 torsion 只应用一次；
- forward-only、optimizer step 和 smoke training 全程 finite；
- v13 及更早 checkpoint 被明确拒绝，v14 架构标识不匹配时立即失败。
