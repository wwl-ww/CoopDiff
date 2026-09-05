# 双 DiffDock 界面坐标架构

## 架构结论

TriCoopDiff 以 E3 为固定全局参考系，扩散状态不是“全局 Ligand 加独立 POI 位姿”，而是两个二体界面的相对状态：

- `T_E3L`：Ligand 相对 E3 的刚体位姿；
- `T_POL`：Ligand 相对 POI 的刚体位姿；
- `chi`：Ligand 内部 torsion。

同一个冻结 DiffDock 基座以两个 branch-specific LoRA/FiLM 分支运行。`sub1` 对 `T_E3L` 去噪，`sub2` 对 `T_POL` 去噪。两者不是待插值的 Ligand score，而是各自界面的完整 proposal。

## 一步更新

每个反向扩散步按以下顺序执行：

1. `sub1` 预测 E3–Ligand proposal，并直接更新 Ligand 的全局刚体位姿；E3 始终不动。
2. 在更新后的 Ligand 上运行 `sub2`，得到 POI–Ligand 相对 proposal。
3. 保持 Ligand 不动，对 `sub2` proposal 应用精确逆刚体变换以更新 POI。
4. 两个 expert 的 torsion proposal 取均值并应用一次。
5. 同步重建 POI local frame、receptor crop、cross-edge 和两个 expert 图。

旋转必须使用 SO(3)/SE(3) 群组合；禁止把 rotation vector 简单取负来近似逆变换。架构中不存在 residual-gated merge，也不存在独立预测 POI 运动的 TriBrain SE(3) head。

## TriBrain 职责

TriBrain 只输出两类信息：

- 两个 expert 各层独立的 FiLM 参数；
- 不改变坐标动力学的 scalar auxiliary heads。

TriBrain 不合并 expert score、不输出 Ligand/POI residual score，也不替代 DiffDock 完成全局刚体装配。

## 训练与推理对齐

训练分别对 `T_E3L` 和 `T_POL` 使用 native DiffDock forward noise。加噪后的两个相对刚体通过精确 SE(3) 组合重建唯一且一致的全局 Ligand/POI 坐标，然后构建两个二体图。每个 expert 接收自己界面的真实时间和噪声目标，禁止独立扰动全局 POI 后让 `sub2` 复用 `sub1` 的时间条件。

推理同样维护这两个界面状态。初始化策略只能决定两个界面 proposal 的起点，不能改变状态定义或绕过精确坐标重建。

## Checkpoint 边界

新架构 checkpoint 必须同时满足：

- `checkpoint_version == 4`；
- `architecture == "interface_coordinate_v1"`；
- 包含与当前 branch-specific FiLM/LoRA 配置匹配的状态。

加载器必须拒绝 version 3 及更早 checkpoint；不提供 legacy merge 或静默迁移路径。旧 checkpoint 的参数语义与新坐标动力学不同，只能留作历史结果，不能用于恢复、微调或正式评估。

## 验收门禁

- SE(3) 正反变换、组合和全局等变性测试通过；
- `sub1` 更新只改变 Ligand，`sub2` 更新只改变 POI；
- 每一步两个界面 proposal 与重建后的全局坐标一致；
- 两分支 native/inline 输出有限，torsion 平均只应用一次；
- forward-only、optimizer step 和 smoke training 全程 finite；
- v3 checkpoint 被明确拒绝，v4 架构标识不匹配时立即失败。
