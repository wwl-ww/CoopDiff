# DiffDock 实验交接文档

> 给服务器同学 | 2026-08-07

---

## 环境

```bash
conda create -n diffdock python=3.10
conda activate diffdock
pip install torch torch_geometric e3nn rdkit biopython prody scipy

cd CoopDiff/DiffDock
pip install -r requirements.txt

# 下载 DiffDock-L 权重
wget https://github.com/gcorso/DiffDock/releases/latest/download/diffdock_models.zip
unzip diffdock_models.zip
```

---

## 实验一：二体基线 🟡 1-2小时

**目的**：测纯二体 DiffDock 在 PROTAC benchmark 上的表现，作为 TriCoopDiff 的下界参照。

**步骤**：

对 benchmark 里每一个三元 case，拆成两对蛋白-配体分别跑：

```bash
# E3 + PROTAC（蛋白=E3，配体=PROTAC）
python inference.py \
    --protein_path <case_dir>/e3.pdb \
    --ligand_description <case_dir>/protac.sdf \
    --out_dir results/diffdock/<case_name>/e3l \
    --samples_per_complex 10

# POI + PROTAC（蛋白=POI，配体=PROTAC）
python inference.py \
    --protein_path <case_dir>/poi.pdb \
    --ligand_description <case_dir>/protac.sdf \
    --out_dir results/diffdock/<case_name>/poil \
    --samples_per_complex 10
```

**记录**（每个 case）：

| 指标 | 说明 |
|------|------|
| E3-L DockQ | 配体对 E3 的界面接触恢复率 |
| E3-L RMSD | Kabsch 对齐后配体 RMSD |
| POI-L DockQ | 配体对 POI 的界面接触恢复率 |
| POI-L RMSD | Kabsch 对齐后配体 RMSD |
| 冲突率 | 蛋白-配体原子间距 < 3.0Å 的比例 |

**产出**：`results/diffdock_baseline.json`

```json
{
  "model": "DiffDock-L v1.1",
  "date": "2026-08-__",
  "per_case": [
    {
      "name": "5t35",
      "e3l_dockq": 0.xx, "e3l_rmsd": 0.xx,
      "poil_dockq": 0.xx, "poil_rmsd": 0.xx,
      "e3l_clash_rate": 0.xx, "poil_clash_rate": 0.xx
    }
  ]
}
```

**DockQ 计算**：用 `DeepTernary/DockQ/DockQ.py`，predicted vs ground truth complex PDB。

---

## 实验二：模型参数确认 🟡 5分钟

**目的**：确定 DiffDock-L v1.1 的实际参数，否则我们的 FrozenDiffDock 初始化对不上。

**步骤**：

```python
import torch
ckpt = torch.load('diffdock_models/diffdock_models.pt', map_location='cpu')
# 或者 workdir/v1.1/score_model/best_ema_inference_epoch_model.pt

for k, v in ckpt.items():
    if 'ns' in k or 'conv_layers' in k:
        print(k, v)
# 重点看:
# - ns 的值 (16 还是 32)
# - num_conv_layers 的值 (2 还是 6)
# - sh_lmax 的值
```

**产出**：截图或口头告知 `ns`, `num_conv_layers`, `sh_lmax` 三个值。

---

## 实验三：双子图零 OOD 验证 🔴 必须做，10分钟

**目的**：验证我们的 Subgraph 构造与原生 DiffDock 输出数值一致。这是整个架构的基石——不一致的话后续全部作废。

**步骤**：

```python
import torch

# 1. 加载原生 DiffDock
from models.cg_model import CGModel
model = CGModel(...)  # 用实验二的参数
model.load_state_dict(ckpt)
model.eval()

# 2. 取一个 E3+PROTAC 测试样本，跑原生推理
native_graph = load_diffdock_graph(e3_pdb, protac_sdf)  # DiffDock 标准格式
native_out = model(native_graph)  # (tr, rot, tor, _)
print("Native:  tr=", native_out[0][:3], " rot=", native_out[1][:3])

# 3. 用我们的 build_subgraph_e3l() 构造同样的图
from tricoopdiff.graphs import build_subgraph_e3l
our_graph = build_subgraph_e3l(e3_coords, e3_feat, lig_coords, lig_feat, ...)

# 4. 跑我们的 FrozenDiffDock（无 FiLM）
from tricoopdiff.expert import FrozenDiffDock
frozen = FrozenDiffDock(model)
our_out = frozen(our_graph)  # (tr, rot, tor, _)
print("Ours:    tr=", our_out[0][:3], " rot=", our_out[1][:3])

# 5. 比较
assert torch.allclose(native_out[0], our_out[0], atol=1e-5), "TR MISMATCH"
assert torch.allclose(native_out[1], our_out[1], atol=1e-5), "ROT MISMATCH"
print("PASS: zero OOD verified")
```

**产出**：终端输出截图，PASS 或 FAIL。

---

## 实验结果交付

三个实验都完成后，把以下文件发给我：

```
results/
  diffdock_baseline.json       ← 实验一
  diffdock_params.txt          ← 实验二 (ns=?, num_conv_layers=?, sh_lmax=?)
  ood_verification.png         ← 实验三终端截图
```

有问题随时找我。
