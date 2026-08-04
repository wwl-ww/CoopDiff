# DeepTernary 审计交接文档

> 给服务器同学的 DeepTernary 审计任务说明 | 2026-08-04

---

## 一、审计目的

在 PROTAC benchmark（~50个结构）上对 DeepTernary 进行完整评估，建立 baseline 数据供 TriCoopDiff 对比。

---

## 二、环境搭建

```bash
# 1. Clone 项目（已在项目根目录）
cd CoopDiff/DeepTernary

# 2. 安装依赖
pip install torch==2.3.1  # 或最新版
pip install -r requirements.txt

# 3. 下载预训练权重 + PROTAC unbound 结构
wget https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.0/output.zip
unzip output.zip

# 4. 下载 TernaryDB（可选，如果已有 PROTAC benchmark 数据）
# wget https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.1/TernaryDB.tar.gz
```

---

## 三、运行预测

对测试集每个 PROTAC 跑 40-seed 预测：

```bash
# 批量跑（参照 predict_cpu.py 的逻辑）
# PROTAC test set 在 output/protac22/ 目录下

python predict_cpu.py output/checkpoints/PROTAC
```

或者单独跑某个 case：
```bash
python predict.py \
    --task PROTAC \
    --name <case_name> \
    --lig output/protac22/<case>/ligand.pdb \
    --p1 output/protac22/<case>/unbound_protein1.pdb \
    --p2 output/protac22/<case>/unbound_protein2.pdb \
    --unbound-lig1 output/protac22/<case>/unbound_lig1.pdb \
    --unbound-lig2 output/protac22/<case>/unbound_lig2.pdb \
    --lig1-mask output/protac22/<case>/unbound_lig1.pdb \
    --lig2-mask output/protac22/<case>/unbound_lig2.pdb \
    --outdir ./audit_results/<case_name> \
    --seeds 40
```

每个 case 输出到 `audit_results/<case_name>/`：
- `complex_pred_<name>_<seed>.pdb` — 40 个预测结构
- `summary_<name>.csv` — seed, pred_p2_rmsd, clash_ratio

---

## 四、审计指标（按执行计划 Phase 0.2）

### 4.1 结构质量指标

| 指标 | 定义 | 计算方法 |
|------|------|---------|
| **全局 DockQ** | CAPRI DockQ，衡量界面接触恢复率 | 用 `DeepTernary/DockQ/DockQ.py` 对每个 predicted complex vs ground truth complex 计算 |
| **按 linker 长度分层 DockQ** | 短（<8）/ 中（8-15）/ 长（>15）可旋转键 | PROTAC 分子先按可旋转键数分组，再计算各组平均 DockQ |
| **空间冲突率** | 蛋白-蛋白/蛋白-配体穿模比例 | 分子间 Cβ-Cβ 距离 < 3.0Å 的比例 |
| **Linker 应变能** | MMFF94 能量评估 linker 扭转应变 | RDKit `AllChem.MMFFGetMoleculeForceField().CalcEnergy()`，只算 linker 部分 |
| **配体 RMSD** | Kabsch 对齐后预测 vs 真实配体 RMSD | `scipy.spatial.transform.Rotation.align_vectors()` 后计算 |

### 4.2 系综质量指标

| 指标 | 定义 | 计算方法 |
|------|------|---------|
| **40-seed 多样性** | 聚类数和 pairwise RMSD | 对所有 seed 的配体坐标聚类（2Å cutoff），报告聚类数 |
| **Top-1 命中率** | 采样最多的构象是否接近真实 | 聚类后占比最大的 cluster 的 centroid ←→ ground truth RMSD |
| **收敛性** | pred_p2_rmsd 随 seed 的分布 | 统计 40 seed 中 pred_p2_rmsd 的 mean/std/min/max |
| **Seed 选择偏差** | 不同 seed 子集的结果稳定性 | 随机抽 5/10/20/40 seed 各 10 次，报告 DockQ 均值±std |

---

## 五、输出要求

请输出一个结构化 JSON 文件 `deepternary_audit.json`：

```json
{
  "audit_date": "2026-08-__",
  "model": "DeepTernary",
  "checkpoint": "output/checkpoints/PROTAC/",
  "n_cases": 50,
  "global_metrics": {
    "mean_dockq": 0.0,
    "std_dockq": 0.0,
    "median_dockq": 0.0,
    "mean_ligand_rmsd": 0.0,
    "mean_clash_rate": 0.0,
    "mean_linker_strain": 0.0
  },
  "stratified_dockq": {
    "short_linker_leq8": {"n": 0, "mean_dockq": 0.0, "std_dockq": 0.0},
    "mid_linker_8_15": {"n": 0, "mean_dockq": 0.0, "std_dockq": 0.0},
    "long_linker_gt15": {"n": 0, "mean_dockq": 0.0, "std_dockq": 0.0}
  },
  "ensemble_metrics": {
    "mean_n_clusters": 0.0,
    "mean_top1_rmsd": 0.0,
    "pred_p2_rmsd_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    "seed_stability": {
      "n5": {"dockq_mean": 0.0, "dockq_std": 0.0},
      "n10": {"dockq_mean": 0.0, "dockq_std": 0.0},
      "n20": {"dockq_mean": 0.0, "dockq_std": 0.0},
      "n40": {"dockq_mean": 0.0, "dockq_std": 0.0}
    }
  },
  "per_case": [
    {
      "name": "...",
      "linker_n_rotatable": 0,
      "linker_group": "short|mid|long",
      "dockq": 0.0,
      "ligand_rmsd": 0.0,
      "clash_rate": 0.0,
      "linker_strain": 0.0,
      "n_clusters_40seed": 0,
      "top1_rmsd": 0.0,
      "pred_p2_rmsd_mean": 0.0,
      "pred_p2_rmsd_best": 0.0
    }
  ]
}
```

---

## 六、注意事项

1. **DockQ 计算**：直接用 `DeepTernary/DockQ/DockQ.py`，输入为 predicted complex PDB 和 ground truth complex PDB
2. **Linker 可旋转键计数**：用 RDKit `Chem.rdMolDescriptors.CalcNumRotatableBonds()` 对 PROTAC SMILES 计算，去除 warhead（E3 binder + POI binder）内部的键，只保留 linker 区域
3. **冲突率**：只计算 **分子间**冲突（E3-POI, E3-PROTAC, POI-PROTAC），不算分子内部冲突
4. **40 个 seed 都要跑**：不要只跑 top-1，系综分析需要全量
5. **输出路径**：所有预测结果放在 `audit_results/` 下，JSON 放项目根目录

---

有问题随时找我。
