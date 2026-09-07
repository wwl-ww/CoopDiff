# TriCoopDiff 模型架构图

```mermaid
flowchart TB
    subgraph INPUT["三元复合物输入"]
        E3["E3 蛋白<br/>残基特征 + 坐标"]
        POI["POI 蛋白<br/>残基特征 + 坐标"]
        LIG["PROTAC<br/>原子特征 + 共价键 + 坐标"]
        TIME["流时间与预条件状态<br/>t · sigma · dt · progress"]
    end

    subgraph TRIFUSE["TriFuse 协调网络"]
        GEO["三元几何编码器<br/>距离 · 接触 · 方向 · 手性"]
        TOKENS["5 个协商 Token<br/>Global · E3-L · POI-L<br/>Linker · Diffusion"]
        TRANS["Token Transformer"]
        FILM1["E3-L 逐原子、逐层 FiLM"]
        FILM2["POI-L 逐原子、逐层 FiLM"]
        CONTROL["Score Control<br/>平移/旋转幅度校准"]
    end

    E3 --> GEO
    POI --> GEO
    LIG --> GEO
    TIME --> TOKENS
    GEO --> TOKENS --> TRANS
    TRANS --> FILM1
    TRANS --> FILM2
    TRANS --> CONTROL

    subgraph EXPERTS["双 DiffDock Expert：11 层同步运行"]
        SUB1["E3-L DiffDock<br/>冻结基础权重 + sub1 LoRA"]
        BRIDGE["Layerwise Expert Bridge<br/>仅交换 Ligand scalar irreps"]
        SUB2["POI-L DiffDock<br/>冻结基础权重 + sub2 LoRA"]
    end

    FILM1 --> SUB1
    FILM2 --> SUB2
    SUB1 <--> BRIDGE
    SUB2 <--> BRIDGE

    SUB1 --> E3S["E3-L scores<br/>translation · rotation · torsion"]
    SUB2 --> POIS["POI-L scores<br/>translation · rotation · torsion"]

    subgraph COORD["联合决策"]
        TELE["逐层激活 Telemetry"]
        TOR["逐键 Torsion Coordinator<br/>融合两位专家的 torsion 意见"]
        RIGID["双界面刚体组合"]
    end

    SUB1 --> TELE
    SUB2 --> TELE
    TELE --> TOR
    E3S --> TOR
    POIS --> TOR
    CONTROL --> RIGID
    E3S --> RIGID
    POIS --> RIGID

    subgraph UPDATE["采样坐标更新"]
        E3FIX["E3 固定"]
        LUPDATE["Ligand 执行 E3-L 刚体 proposal"]
        PUPDATE["POI 接收精确对齐变换<br/>保持 POI-L proposal"]
        TUPDATE["PROTAC 执行联合 torsion"]
    end

    RIGID --> LUPDATE
    RIGID --> PUPDATE
    TOR --> TUPDATE
    E3FIX --> LUPDATE
    LUPDATE --> TUPDATE
    TUPDATE --> OUTPUT["新的三元复合物状态<br/>进入下一流积分步"]
    PUPDATE --> OUTPUT
```

## 模块职责

- **TriFuse** 读取完整三元几何、Ligand 化学结构和流时间与预条件状态，产生两条专家路径各自的逐原子 FiLM。
- **sub1 DiffDock** 预测 E3–PROTAC 界面的平移、旋转和 torsion score。
- **sub2 DiffDock** 在 POI 局部参考系中预测 POI–PROTAC 界面，再将向量 score 映射回 E3 全局参考系。
- **Expert Bridge** 每层仅交换两路 Ligand 的标量表示，禁止跨参考系直接交换向量表示。
- **Torsion Coordinator** 按可旋转键融合两位专家的 torsion 意见，而不是固定求平均。
- **Score Control** 只校准专家 score 的幅度，不自行生成笛卡尔方向。

## 单步坐标更新

两个专家始终读取同一个更新前状态。Ligand 执行 sub1 的刚体 proposal；随后计算一个精确刚体变换，将 sub2 的 Ligand proposal 对齐到 sub1 proposal，并将同一变换施加到 POI。最后执行 TriFuse 协调后的 PROTAC torsion 更新。E3 在整个采样过程中保持固定。

该组合在 torsion 更新前精确保留两侧刚体 proposal 的 receptor–ligand 距离；随后共享 torsion 会同时改变两侧局部几何，不能把该保证延伸到完整一步之后。初始化及其精确装配见 [初始化方案_gpt6.md](初始化方案_gpt6.md)。
