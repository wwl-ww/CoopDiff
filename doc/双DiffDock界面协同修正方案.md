# 双 DiffDock 界面协同修正方案

## 问题定义

当前 `TriCoopDiff` 将同一个 DiffDock 基座用于两个分支：`sub1` 评估 E3–Ligand，`sub2` 评估 POI–Ligand。两条分支拥有独立 LoRA 和 FiLM，但都输出 Ligand 的平移、旋转和扭转 score。随后模型用

```text
s_q = s_1 + g * (s_2 - s_1)
```

合并 Ligand score，而 POI 的刚体 score 由 TriBrain 独立预测。

这个实现适合 near-native POI 的局部精修，却没有真正完成双界面联合装配。MGD 单例已经显示：`coupled5a` 下 Ligand RMSD 约 1 Å，但 Fnat 为 0、DockQ 接近 0；即 DiffDock 正常工作，POI 全局位姿没有恢复。

## 之前的主要错误

1. **把两个相对界面 score 都解释为 Ligand 更新。** `sub2` 描述的是 Ligand–POI 相对位姿；它既可用于移动 Ligand，也应通过逆变换用于移动 POI。当前代码丢弃了后者。
2. **用插值代替联合约束。** 当 `g=1` 时 E3–Ligand score 被完全丢弃；这不是两个界面能量的联合 score，也不能保证两个界面同时成立。
3. **让 TriBrain 从零学习 POI docking。** TriBrain 在两个 expert 运行前生成 `s_se3`，看不到本步 raw expert score，无法利用 DiffDock 已学到的 POI–Ligand几何梯度。
4. **扩散变量定义不一致。** 训练分别扰动全局 Ligand 与 POI，`sub2` 实际看到两份噪声之差，却只按 Ligand 的时间和 sigma 条件化。即使 `t_se3=t`，相对方差仍是两项之和。
5. **初始化掩盖架构缺口。** `anchor_local` 保留 near-native POI 位姿，因此可得到约 0.6 DockQ；`coupled5a` 要求模型完成全局装配时，缺口立即暴露。

## 正确的状态与更新方式

扩散状态应以两个界面相对位姿为核心：

- `T_E3L`：Ligand 相对 E3 的刚体位姿，由 expert 1 去噪；
- `T_POL`：Ligand 相对 POI 的刚体位姿，由 expert 2 去噪；
- `chi`：Ligand 内部 torsion；
- E3 固定。

每一步分别得到两个界面的更新 `Delta T_E3L` 和 `Delta T_POL`。先由 `Delta T_E3L` 更新 Ligand 全局位姿，再对 POI 应用 `Delta T_POL` 的严格 SE(3) 逆变换，使 POI–Ligand 相对位姿满足 expert 2。旋转必须通过刚体变换组合实现，不能简单将 rotation score 乘以负一。

在能量视角下，平移 score 应满足：

```text
s_L = s_E3L + w * s_POL + residual_L
s_POI = -w * s_POL + residual_POI
```

其中反向 SDE 的实际位移仍分别使用 Ligand 和 POI 自己的扩散系数。`w` 是非负、受限的协调权重；同一个 `w` 同时用于作用与反作用，避免两个独立 gate 破坏能量一致性。torsion 只更新 Ligand。

## 实施方案

### 阶段一：暴露接口并建立可验证的最小闭环

1. `tricoopdiff/model.py` 返回映射到 E3 frame 后的 raw `scores_sub1`、`scores_sub2`，不再只返回 merge 结果。
2. 新增 `joint_interface` 模式，保留 `legacy_merge` 仅用于旧 checkpoint 回归。
3. Ligand 使用 gated additive score：`s1 + w*s2`，不再在 `s1` 与 `s2` 之间插值。
4. 从 `sub2` 生成 POI reaction score，与 TriBrain 的 SE(3) residual 相加；第一版先验证 translation，rotation 必须在通过 SO(3)/SE(3) 有限差分测试后启用。
5. 将 SE3 residual head 的最后一层零初始化，使未训练网络不会覆盖 expert reaction。

### 阶段二：改为界面坐标扩散

1. 训练时直接对 `T_E3L`、`T_POL` 添加各自与 DiffDock 一致的 forward noise，再由两者重建全局 Ligand/POI 坐标。
2. 两个 expert 分别接收与其真实相对噪声一致的 `tr/rot/tor` 时间，不再让 `sub2` 混用全局 Ligand 时间。
3. TriBrain 改为两阶段协调器：先产生 FiLM；expert forward 后再读取 raw score、score norm 和界面图摘要，输出非负 coupling weight 与小幅 residual。
4. 初始化直接采样两个界面 proposal，再通过 SE(3) 组合生成完整三元结构；`anchor_local` 只作为局部精修基线。

### 阶段三：checkpoint 与兼容性

1. 新架构 checkpoint 升级为 version 4，并记录 `coordination_mode=joint_interface`。
2. version 3 checkpoint 只能以 `legacy_merge` 加载，禁止静默套用新公式。
3. 正式训练从新架构初始化；旧 checkpoint 仅用于 A/B，不作为恢复训练起点。

## 验证顺序与验收标准

1. **数学单元测试**：平移作用/反作用和 SE(3) 组合可逆；全局刚体变换后结果等变；zero coupling 精确恢复 sub1 parity。
2. **合成三体测试**：固定 E3 和 Ligand，仅给 POI 错误平移/旋转，确认 sub2 reaction 单调降低 POI–Ligand pose error。
3. **单例轨迹**：在 `5T35_D_A_759` 与 MGD `3QN1_A_B_A8S` 上使用 `coupled5a`，记录每步两个界面距离、raw score、reaction、POI 位移和首次非有限位置。
4. **训练门禁**：100 次随机 forward、全索引 forward-only、单 optimizer step 和 3-epoch smoke 全部 finite。
5. **质量门禁**：32 walkers 中绝大多数 finite；E3–POI 不单调发散；`coupled5a` 相比 legacy merge 明显提高 Fnat/DockQ；`anchor_local` 不低于现有基线。

在阶段一通过前，不启动正式 Phase 1；在 `coupled5a` 不能恢复 POI 全局位姿前，不以 `anchor_local` 的高 DockQ 宣称全局三元 docking 成功。
