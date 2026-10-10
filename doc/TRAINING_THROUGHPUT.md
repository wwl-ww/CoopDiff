# 训练吞吐与多卡准备

2026-10-10：当前只授权短测，不自动启动 400-update 训练。性能报告以实际生成的 manifest、更新计时和验证结果为准。

## 单卡

当前 `--batch_size` 是顺序 microbatch 的平均梯度累积；增加它不会让多个复合物同时进入模型。装配模块明确要求每次输入一个节点对齐的复合物，不能直接把坐标 padding 成 batch。真正的图批处理仍需另行实现并检查跨图边、扭转索引和逐图损失。

`--no-checkpoint-ternary` 保留更多三元主干激活，避免其反向重计算；冻结二元专家的输入梯度和专家重计算仍保持原实现。专家权重继续冻结。不要通过关闭输入梯度、减少积分步数或改变损失来声称同等训练提速。

首轮完整试跑：开启重计算更新 148.64 s、峰值 allocated 1.07 GiB；关闭为 101.08 s、7.74 GiB。这是不同共享 GPU 的首个更新，含编译影响，不能直接作为严格加速比。配对基准入口：

```bash
source results/deployment/activate.sh
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_training_throughput.py \
  --checkpoint-mode on --output-dir results/throughput_on
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_training_throughput.py \
  --checkpoint-mode off --output-dir results/throughput_off
```

两模式分别从 seed=73 新模型开始，固定 8 例、有效 batch=4、20 步展开，默认预热一次再计时两次；不运行 DockQ。以预热后的样本/秒、同步墙时和 allocated/reserved 峰值比较，整体 GPU 占用包含其他任务，不能视为本进程显存。

## 多卡

`scripts/train_distributed.py` 采用每 GPU 一个进程和显式梯度 all-reduce；不是包装标准 DDP 后直接并行整个装配模型。先对各卡局部平均梯度按样本数加权求全局平均，再做一次全局范数裁剪、优化器更新和 EMA 更新。

全局有效 batch 保持一致时，各卡分担样本；每卡复制冻结专家，不能指望模型显存自动分片。near-target 的样本序号需要全局偏移，`samples_seen` 按全局样本数推进。只有 rank 0 写公共 checkpoint，所有 rank 的 RNG 和运行拓扑须记录。入口是否支持分布式续跑必须明确声明，单卡 checkpoint 不能自动当作多卡可精确恢复的证据。

验收顺序：CPU/Gloo 梯度与计数契约、失败传播测试 → 双 GPU 单次真实更新 → 同全局 batch 单卡/双卡预热后吞吐比较。未完成实际验收前，不宣称线性加速或生产多卡训练已就绪。

示例（需已经预处理的完整索引和可用 GPU；全局 batch=4，两卡各2例；仅一次更新）：

```bash
source results/deployment/activate.sh
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --standalone \
  --nproc-per-node=2 scripts/train_distributed.py \
  --train_data results/small_sample_seed73/train.json \
  --val_data results/small_sample_seed73/same_complex_val.json \
  --output_dir results/distributed_short_check \
  --cache_dir results/featurize_cache --expert-backend matcha \
  --training-arm A1 --seed 73 --batch_size 4 --lr .0001 \
  --no-random-flip-proteins --no-checkpoint-ternary --max-updates 1
```

当前入口只训练，不运行验证；必须另行验证检查点，不能从 loss 或吞吐推断 DockQ。只支持 A1，拒绝缓存过滤和不完整全局尾批，不静默丢弃或复制样本。性能测量的更新墙时不包含特征装载和保存，需同时观察端到端耗时。

### 断点续训

使用相同训练参数，添加 `--resume <checkpoint_updateN.pt>`、新的输出目录和累计目标 `--max-updates M`。例如上例的 checkpoint_update1.pt 加 `--max-updates 2` 只再运行一次更新。

恢复模型、Adam 状态、scheduler、EMA、global_step、samples_seen、每个 rank 的随机状态以及下一批数据游标。当前要求 world size、全局 batch、seed、数据索引和采样配置匹配；换物理 GPU 可以，改变卡数需另做迁移协议。新 checkpoint 记录索引 SHA，标记 `distributed_training`，标准单卡入口拒绝误恢复。

最早的多卡 smoke artifact 虽标记 inference-only，仍保留完整优化器及每卡 RNG，可以由多卡入口结合旁边的 manifest 恢复；它没有保存数据索引历史 SHA，只能校验原索引路径，因此记录 `legacy_dataset_identity=path_only`。新保存的断点使用内容哈希校验。

真实双卡续跑已完成：从 update1 恢复到 update2，只新增一次更新，`samples_seen=8`，下一批游标为 epoch1/offset0；Adam step=2，模型及 EMA 更新，两卡参数 SHA 完全一致，新断点 `resume_supported=true`。该次更新 100.71 s；结果记录在 `results/small_sample_seed73/distributed_resume_summary.json`。针对性 CPU 检查也验证恢复后的下一次更新与不中断运行一致，以及错误拓扑/单卡入口拒绝恢复；未额外启动长训练。

已完成 CPU/Gloo 加权梯度、缺失梯度、失败传播和全局 near-target 调度检查；最终训练契约模块 73 项通过。真实 GPU 1+2 的最终一次更新耗时 50.29 s，全局样本数 4，显存峰值 6.47/7.75 GiB；两 rank 更新后参数 SHA 完全一致，检查点保存两份 RNG。结果：`results/small_sample_seed73/distributed_smoke_summary.json`。此轮使用生产默认 lr=.001，仅用于集成验收，不能与 lr=.0001 的研究实验比较指标，也不是预热后的吞吐基准。
