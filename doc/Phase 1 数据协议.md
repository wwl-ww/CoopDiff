# TriCoopDiff Phase 1 数据协议

## 训练池

正式 Phase 1 使用 DeepTernary 公开的两个训练 cluster 文件作为数据来源：

- `PROTAC/train_clusters_poc6.json`
- `MolecularGlue/train_clusters_poc3_exUNL.json`

仅保留两个成员集合的精确交集，并排除与 PROTAC22 或 Molecular Glue 94
benchmark 具有相同四位 PDB ID 的 case。最终联合池固定为 **7,198 cases、
653 个合并 cluster**。两个来源中的 cluster 关系通过并查集合并，防止关联
case 被拆到训练和验证两侧。

## 内部验证与采样

使用 `split_seed=0` 按 cluster 留出 10% 作为内部 validation：训练为 588
clusters，validation 为 65 clusters。训练时每个 epoch 从每个 cluster 采一个
case：80% 从全部成员选择，20% 优先选择官方 representative；无
representative 时退化为全部成员。Validation 固定使用每个 cluster 的首个
representative，保证 checkpoint 选择可复现。

与 DeepTernary 一致，训练期以 50% 概率交换两个 protein 的角色；交换时整个
三元状态一起平移到新的固定 protein 参考系。Validation 不做随机交换。

正式 benchmark 不参与 `t_anchor`、epoch 或 checkpoint 选择。

## 正式评估目标（与当前工程诊断区分）

- PROTAC：官方 22 cases，unbound 输入，每个 case 40 seeds。
- Molecular Glue：官方 94 cases，bound 输入，每个 case 1 seed。

PROTAC 的主结果应按经验证的预测置信度选择 top-1；当前代码尚无经过监督的
ranking head，评估报告的 best 是按真实结构指标挑选的 oracle-best，不能称作
model-ranked top-1。旧的 23-case bound、70-case 或 791-case representative
结果以及当前晶体拆分输入的初始化 smoke 仅作工程诊断。

## 可复现性门禁

`scripts/make_deepternary_index.py` 会验证联合池计数，并在任一结构无法拆分时
终止。正式入口记录 Git commit、索引与专家权重 SHA256，并在本地 GPU 上先完成
冻结专家的 smoke，再从头启动正式训练。当前交付模型不训练 DiffDock 或 LoRA。
