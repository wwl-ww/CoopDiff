# 冻结 DiffDock 专家：当前实现与配置

本文以仓库 vendored core、`expert_factory.py` 和实际 score checkpoint 配置为准，不把原论文的训练超参数或旧实验完成状态当作 TriCoopDiff 的当前配置。

## 来源与加载

源码来源记录在 [diffdock/__init__.py](../tricoopdiff/diffdock/__init__.py)：上游 `47eb91a`，另有仓库内导入适配与数值稳定修改。运行模型不依赖相邻的 DiffDock checkout。新增入口先导入 `tricoopdiff.diffdock_compat`。

[expert_factory.py](../tricoopdiff/expert_factory.py) 从 `model_parameters.yml` 构造模型，并严格加载 score 权重。输出报告记录来源说明、checkpoint 路径与 SHA256。两个 expert 共用冻结基座，分支 LoRA/FiLM 独立；不向冻结配体图注入片段或端点先验。

## 当前 score checkpoint

以下数值来自 [model_parameters.yml](../checkpoints/score_model/model_parameters.yml)，不是所有 DiffDock 版本的通用常量。更换权重时必须同时核对构建参数和扩散 schedule。

| 参数 | 当前值 |
|---|---|
| `ns / nv` | 48 / 10 |
| `num_conv_layers` | 11 |
| `sigma_embed_dim` | 64 |
| `embedding_type` | sinusoidal |
| `distance_embed_dim / cross_distance_embed_dim` | 64 / 64 |
| `receptor_radius / c_alpha_max_neighbors` | 15 Å / 24 |
| `dynamic_max_cross` | true |
| `all_atoms / no_torsion` | false / false |
| `sh_lmax / scale_by_sigma` | 2 / true |
| translation σ | 0.1 → 19 |
| rotation σ | 0.03 → 1.55 |
| torsion σ | 0.0314 → 3.14 |

受体输入为 Cα 残基图和 ESM2 特征，配体为完整原子与共价键图。`CGModel` 包含 embedding、受体/配体交叉卷积及 tr/rot/tor heads。具体 irreps 与头部切片必须查 [cg_model.py](../tricoopdiff/diffdock/models/cg_model.py) 和 [expert.py](../tricoopdiff/expert.py)，不能套用另一配置的六层结构。

## 扩散与初始化不可混用

[diffusion.py](../tricoopdiff/diffusion.py) 使用指数 σ schedule、IGSO(3) 旋转噪声和 torsion 查表归一化。反向采样默认 20 步，末步不加随机噪声；温度修正由 `InterfaceDiffusion` 的参数定义。

`tr_sigma_max=19` 是扩散 schedule 上限，不能用作当前初始化标准差。唯一初始化 `dual_native` 对两侧分别使用：

`sigma_init = receptor_rms_radius × 1.4601642460337794 / 1.73`

完整构象、共享 torsion 和精确逆装配见 [初始化方案_gpt6.md](初始化方案_gpt6.md)。训练时间采样与损失以 [train.py](../tricoopdiff/train.py) 为准；TriCoopDiff 的 Phase 1 不照搬 DiffDock 原始训练的 batch size、学习率或 epoch 数。

## 集成门禁

- crop、cross-edge 与同侧 ligand/receptor 坐标必须处于同一参考系；
- 两分支都读取同一完整 ligand 和同一 canonical torsion 顺序；
- inline 卷积与原专家采用相同 clamp 配置，零 FiLM 下核对逐层和最终 score；
- 非有限 score 使 walker 失效，不用零值替换；
- clamp 值必须从 checkpoint 的 `runtime_config` 恢复；默认 100 不是不可更改的模型常量；
- 缓存、权重、输入数据和运行参数齐全后才报告通过，不能把历史 checklist 当作本次验收结果。

SO(3)/torus 缓存按仓库运行环境解析；命令从仓库根目录执行。CPU 回归与离线模型验证命令见 [初始化方案_gpt6.md](初始化方案_gpt6.md)。
