# 问题与证据账本

更新：2026-09-14。这里记录限制与证据，不充当任务调度器。当前结果见[BiTSeed 协议](doc/BiTSeed-Hybrid在线初始化规格.md)，执行边界见 [todo.md](todo.md)。

## 尚未解决的核心问题

### Q-003：搜索覆盖与学习场质量

历史固定状态/路径实验显示 POI 朝向学习不足，位置和朝向需要联合搜索。MGD 五例×8 seeds 的 native-reading 干预中，无干预 mean DockQ 0.04058、仅平移 0.06021、仅旋转 0.06617、全刚体 1.0；最后一组还使用 native 构象，只能定位自由度，不能当模型成绩。

现有回归降低了已检查映射、坐标符号错误的可能性，**不证明所有实现缺陷已排除**，也未证明某个 coupling 假设是唯一根因。历史方向证据：`results/epoch20_velocity_direction_audit/summary.json` 与 `results/epoch{5,10,15,20}_velocity_direction_audit/`。这些旧 checkpoint 诊断不直接外推当前 factorial 权重。

### Q-004：精修不保证单调改善

精确 target 的 Euler 10/20/40/80 步在 5Z86 上呈一阶收敛，见 `results/oracle_integration_step_audit.json`。旧模型 5T35 seed0 从20加到40步反而略降，见 `results/epoch20_steps40_probe/`。数值积分误差和学习场误差都可能存在，不能直接认定缩短时间或增加步数就能修好。

最新 MGD94 中，SDPI-64 的 case-best 均值提升，而全 BiTSeed-192 的 case-best 均值下降；但多数 BiTSeed 单候选仍提高。必须区分候选平均收益、每例最佳保留与最终 top-1，不再笼统写“BiTSeed 不利好精修”或“精修通常一定变好”。

### Q-012：训练与推理的内部几何支持

训练 SDPI 端点由 native scaffold 施加 torsion 得到，推理使用 ETKDG scaffold；键长、不可旋转键角和环系差异不是现有刚体+torsion 能全部消除的。三个 validation cases×5 seeds 对齐绝对二面角后，历史审计仍有 mean aligned RMSD 2.013 Å、bond-length RMSE 0.109 Å、bond-angle RMSE 10.81°，见 `results/endpoint_support_audit_v2.json`。

这证明对应样本存在支持失配，不证明全部低 DockQ 都由它解释。联合训练必须检查实际端点、可达 target、gauge 和时间一致性；不能在推理读取 native ligand，也不能仅添加来源标志便宣布解决。

### Q-015：selector 的初态与末态目标不一致

MGD94 全池初态/末态 oracle 赢家 ID 仅25/94一致。Geometry-64 和 Diversity-64 的最终逐例平均 retention 约83.3%/83.5%，说明 native-free 256→64 仍会丢失最终好构象。二者90/94例 final 最大值相同，0.00029均值差不足以定型。新模型下须重测，不能用初始化 oracle 消融替代末态 retention。

### Q-016：粗几何标记不能充当最终物理判决

当前 `acceptable` 混合 Cα 穿插与接触距离，而且记录的是**初态**。只有2,306/24,064初态通过；native 对照92/94通过，3UJL和5MXO因 POI–ligand 最小距离超过8 Å失败。

因此该标记既不是全原子化学有效性，也不是经过验证的 hard safety。不能由初态不通过推断其精修后的高 DockQ 无效。改 hard filter 前需要独立验证好姿态召回，并分别计算初态/末态全原子几何。当前只作软排序。

### Q-013 / Q-014：二元先验与小分子指标

历史两案例16条独立 DiffDock 轨迹仅12条可评分，均无 LRMSD<2 Å；一条非有限，三条距离成键图匹配失败。范围太小，且官方完整推理一致性未核对，不能据此否定所有二元先验。DockQ-small_molecules 不可套用蛋白–蛋白 DockQ 阈值，详见[二元评估](doc/独立DiffDock小分子评估.md)。

## 已修复和关闭的历史项

以下为历史证据，不表示本次文档整理重新运行过测试。

| 编号 | 状态与边界 | 证据 |
| --- | --- | --- |
| Q-001 | 随机 torus 表导致跨进程漂移；改确定性积分。只承诺已验证环境 | `7741b87`；`results/deterministic_production_gate_{1,2,3,4,5,6}/` |
| Q-002 | 旧训练归一化表未保存，无法恢复；新 checkpoint 记录版本 | `wrapped_normal_quadrature_v1`；`tests/test_torus_score_norm.py` |
| Q-005 | 远离口袋后 cross-edge 变稀疏是该失败轨迹的放大机制，非已证起始根因 | `results/epoch20_failure_trajectories_v2/` |
| Q-006 | 该案例未发现输出头近零归一化爆炸；不推广为全库无风险 | `results/epoch20_head_direction_probe/5T35_D_A_759_seed0.json` |
| Q-007 | 续跑校验加入数值协议，避免新旧报告混用 | `tests/test_evaluation_common.py` |
| Q-008 | 解析速度/参考系/反向更新经18个真实验证状态互相核对；不支持整体反号修补 | `tests/test_sdpi_flow.py`；方向审计 |
| Q-009 | 生成器补显式 split 标签，旧索引不回写；PDB 交集仍须另查 | `scripts/make_deepternary_index.py` |
| Q-010 | Trainer 落实已记录 activation clamp；历史18 probes复核有限 | `tests/test_training_contract.py` |
| Q-011 | checkpoint 绑定 frozen expert 身份；旧格式未记录信息不可追溯补齐 | `tests/test_checkpoint_contract.py` |

## 历史研究结论的使用边界

小幅 loss 变体、增加 walkers 和旧 anchor 面板只在各自记录的样本/预算上成立，不能混入最新统一池榜单。“多模态 coupling 冲突”“来源条件可帮助混合训练”保留为有动机的研究假设，不写成已被因果证明的唯一解释。

历史结果可能位于未迁移的本地产物目录；引用路径不代表资产必然随 Git 存在。完整旧问题记录和低 DockQ 交接可从 `3070922` 查看。新增问题保留编号、可反驳假设、证据和适用范围；不再加入固定 GPU、禁止收工等过期工作命令。
