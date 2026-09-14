# 当前工作与交接

更新：2026-09-14。这里仅记录执行状态，不重复协议或历史流水账。

## 本分支已完成

- [x] 在线 BiTSeed、短补采、部分成功保留和 SDPI 缺额补位。
- [x] 确定性 256→32/64 选择：几何排序、固定 E3 坐标系去重、来源保护与回填。
- [x] PROTAC22 固定 256 候选的比例 oracle 消融。
- [x] MGD94 全池 24,064 条精修；同一轨迹用于各 selector 的反事实比较。
- [x] 分来源统计与 Initial / Final / Union 三榜；结论见[协议](doc/BiTSeed-Hybrid在线初始化规格.md)。
- [x] `scripts/gpu_status.sh` 一次查看全部 GPU。
- [x] 合并旧架构、公式、初始化及运行交接，删除过期重复文档。

## 当前决定

候选池沿用 192 BiTSeed + 64 SDPI；不做 DiffDock LoRA。保留 Geometry-64、Diversity-64、Hybrid-48/16 三个 selector 对照，暂不宣布唯一最佳。Hybrid-56/8 和纯 BiTSeed-64 在当前 checkpoint 下不作优先方案。

本分支目前以保留基线、等待联合训练对照为主。下面是后续工作清单，不构成自动启动 GPU 作业或重新训练的授权。

## 待办

- [ ] 等待 `exp/dual-init` 交付共享模型、训练配置、checkpoint 身份和双来源验证结果。
- [ ] 在内部验证集比较旧模型、无来源条件的混合训练、有来源条件的混合训练；使用相同初态和预算。
- [ ] 新权重下重新审计三种 selector 的最终 retention、来源收益与 SDPI 能力保持。
- [ ] 分别检查初态、末态的全原子几何；验证好构象召回后，才决定哪些规则可 hard reject。
- [ ] 在内部验证集冻结 selector；MGD94 仅作已使用过的开发评估，不冒充未触碰 holdout。
- [ ] 部署 top-1 仍需独立验证的 native-free 选择规则；Union oracle 不替代排序器。

## 联合训练分支边界

分支：`exp/dual-init`。工作树：`/mnt/disk/wl/MoRE-tri/CoopDiff-bitseed-mixed-path-repro`（目录保留旧名）。

目标是训练**同一个**兼容两种初始化的模型，不是只复现混合路径失败，也不是拆成两个模型。该分支的 `doc/SDPI-BiTSeed联合训练交接.md` 负责具体实现计划；本目录的存在不代表其成果已合并。不要从本分支清理或改写另一工作树。

用户提供的 `hh.md` 保留原文，属于设计意见，不是代码契约或实验结论。已采纳的主张是“共享模型 + 真实双来源训练路径 + 来源条件消融”，不是“只加一个 bit 就解决分布失配”。

## 产物与恢复

- 最新 MGD94：`results/mgd_full_pool_refinement_20260913/summary_v1.json`，逐例文件位置由其中 `integrity.case_files` 指定。
- 早期 proposal 消融：`results/proposal_oracle_ablation_20260912/`；只比较同协议目录，不混用 v4 前后的失败处理。
- 旧 SDPI 对照：`results/mgd_followup/mgd94_x64_seed31000/`、`mgd94_x64_seed41000/`；它们不是当前统一池的同一批初态。
- 精修 checkpoint 与复跑命令见[运行指南](doc/训练入口%20SLURM%20与%20Phase1%20smoke%20交接文档.md)。
- 已清理的历史文档/分析脚本可从提交 `3070922` / `a097801` 读取；恢复到独立目录，不覆盖当前源码与结果。

GPU 状态、PID 和 ETA 不写成长期规范。需要时运行 `bash scripts/gpu_status.sh`，再查具体日志与输出更新时间。
