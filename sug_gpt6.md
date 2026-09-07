# A–PROTAC–B 三元复合物模型中的“共享 PROTAC 争夺”问题与解决方案

## 1. 背景与当前模型设定

我们的目标是预测三元复合物：

\[
A-\mathrm{PROTAC}-B
\]

其中 PROTAC 可进一步写成：

\[
\mathrm{PROTAC}=W_A-L-W_B
\]

- \(W_A\)：面向蛋白 \(A\) 的结合头；
- \(W_B\)：面向蛋白 \(B\) 的结合头；
- \(L\)：连接两端结合头的 linker。

当前模型由三个主要模块组成：

1. **Binary Expert A**
   - 专门负责 \(A-\mathrm{PROTAC}\) 二体体系；
   - 继承/利用强二元结合模型的先验；
   - 擅长预测 \(A\) 与 \(W_A\) 一侧的局部结合几何。

2. **Binary Expert B**
   - 专门负责 \(\mathrm{PROTAC}-B\) 二体体系；
   - 同样利用强二元结合先验；
   - 擅长预测 \(W_B\) 与 \(B\) 一侧的局部结合几何。

3. **Fuse**
   - 在两个二元专家之间传递信息；
   - 建模真正的三元耦合关系；
   - 负责 PROTAC、linker、A/B 相对构象以及跨二体的信息交互。

原始设计思想是：

\[
E_A + E_B + F
\]

其中两个 binary expert 提供强大的局部二元先验，Fuse 负责把两侧信息整合成完整的三元复合物。

---

# 2. 实际遇到的核心问题

## 2.1 两个二元专家同时“拥有”同一个 PROTAC

问题的根源不是某个专家不够强，而是：

> **两个 binary expert 都试图根据各自的局部二元目标直接修改同一个共享 PROTAC。**

对于 A-side expert：

\[
E_A \rightarrow \Delta X_P^A
\]

它希望把 PROTAC 调整到最适合 \(A-W_A\) 二元结合的位置。

对于 B-side expert：

\[
E_B \rightarrow \Delta X_P^B
\]

它希望把同一个 PROTAC 调整到最适合 \(W_B-B\) 二元结合的位置。

但真实体系中只有一个 PROTAC：

\[
X_P
\]

因此两个专家实际上在对同一个几何对象施加两个不同、甚至相互矛盾的局部更新：

\[
\Delta X_P^A \neq \Delta X_P^B
\]

这会形成典型的“拔河”现象。

---

## 2.2 局部正确不代表全局兼容

两个 binary expert 各自都可能是正确的：

\[
A-W_A
\]

局部结合可以很好；

\[
W_B-B
\]

局部结合也可以很好。

但两套局部最优解未必同时存在于同一个完整三元构象中。

真正需要满足的是：

\[
\boxed{
\text{A-side local optimum}
+
\text{B-side local optimum}
+
\text{linker feasibility}
+
\text{A/B global compatibility}
}
\]

而不是简单地分别优化：

\[
A-\mathrm{PROTAC}
\]

和

\[
\mathrm{PROTAC}-B
\]

因此问题本质上不是单纯的信息融合问题，而是一个：

\[
\boxed{
\text{Geometry Consensus / Constraint Compatibility}
}
\]

问题。

---

# 3. 最危险的现象：正反馈死循环

当前结构很容易形成以下闭环：

\[
A/B\text{ 同时拉扯 PROTAC}
\]

导致：

\[
\mathrm{PROTAC\ geometry}\downarrow
\]

PROTAC 变差以后，两个 binary expert 下一步看到的输入也变差：

\[
E_A(X_P), E_B(X_P)
\]

由于输入状态偏离各自熟悉的二元结合分布，它们的预测开始恶化：

\[
\text{binary prediction quality}\downarrow
\]

于是新的更新更加不一致：

\[
\text{A/B conflict}\uparrow
\]

进一步造成：

\[
\mathrm{PROTAC\ geometry}\downarrow\downarrow
\]

最终形成：

\[
\boxed{
\text{PROTAC变差}
\rightarrow
\text{二元专家判断变差}
\rightarrow
\text{两侧冲突加剧}
\rightarrow
\text{PROTAC进一步变差}
}
\]

这是一个典型的 **positive feedback instability**。

---

# 4. 为什么这个问题会进一步破坏 Binary Expert 的优势

Binary expert 的价值在于它们已经学到了高质量的二元结合 vector field / score field。

它们擅长处理的是接近二元训练分布的状态：

\[
x_t \sim p_{\mathrm{binary}}
\]

但当另一侧 expert 和 Fuse 持续改变共享 PROTAC 后，当前状态可能逐渐变成：

\[
x_t \notin p_{\mathrm{binary}}
\]

即进入 **out-of-distribution state**。

于是：

\[
\mathrm{OOD}
\rightarrow
\mathrm{bad\ binary\ prediction}
\rightarrow
\mathrm{worse\ state}
\rightarrow
\mathrm{more\ OOD}
\]

这意味着：

> 两个原本很强的 pretrained binary experts，在错误的耦合方式下反而会逐渐失去其原有优势。

因此我们的设计目标不能只是“让两个 expert 能够互相交流”，还必须做到：

\[
\boxed{
\text{尽量保护原始 binary prior / binary vector field}
}
\]

---

# 5. 核心设计原则

整个问题可以浓缩成一句话：

\[
\boxed{
\textbf{Local experts propose; global Fuse decides.}
}
\]

即：

> **局部专家负责提出建议，Fuse 负责形成唯一的全局决策。**

而不能采用：

\[
\boxed{
\text{Local experts move; Fuse repairs.}
}
\]

后者意味着两个 binary experts 先各自修改状态，再让 Fuse 去修补冲突。

这恰恰是当前不稳定闭环的重要来源。

---

# 6. 单一全局状态原则

整个 ternary complex 在任何 diffusion / flow step 中，都必须只有一个全局状态：

\[
X_t
=
\{
X_A,\,
X_P,\,
X_B
\}
\]

其中：

\[
X_P
\]

是唯一的一份 PROTAC 坐标与构象。

Binary Expert A、Binary Expert B 和 Fuse 都只能读取这一个共享状态。

最关键的是：

\[
\boxed{
\text{每个 diffusion / flow step 最终只允许执行一次 global update}
}
\]

即：

\[
X_{t-\Delta t}
=
\operatorname{Update}
\left(
X_t,
\Delta X_{\mathrm{global}}
\right)
\]

不能出现：

\[
E_A\rightarrow X'
\]

然后：

\[
E_B\rightarrow X''
\]

最后：

\[
F\rightarrow X'''
\]

这样的连续多主体坐标修改。

---

# 7. 推荐架构：Binary Expert 只产生 Proposal

原始结构：

```text
Binary Expert A ──直接修改──┐
                            │
                         PROTAC
                            │
Binary Expert B ──直接修改──┘
             ↓
           Fuse 修补
```

推荐改成：

```text
Binary Expert A ── proposal / message ──┐
                                       │
Binary Expert B ── proposal / message ──┼── Fuse / Consensus
                                       │
Current Global State ──────────────────┘
                                               │
                                               ▼
                                      ONE Global Update
```

即两个 binary expert 输出：

\[
m_A = E_A(X_t)
\]

\[
m_B = E_B(X_t)
\]

这里的 \(m_A,m_B\) 可以包括：

- local score / vector field；
- SE(3) translation proposal；
- rotation proposal；
- torsion proposal；
- residue–ligand interaction representation；
- pocket geometry；
- binding confidence；
- uncertainty；
- local constraint embedding；
- desired warhead pose；
- cross-attention features。

但它们**不直接改变全局状态**。

Fuse 接收：

\[
F(
X_t,\,
m_A,\,
m_B,\,
t
)
\]

最后只输出唯一的：

\[
\Delta X_{\mathrm{global}}
\]

---

# 8. 模块职责重新划分

对于：

\[
A-W_A-L-W_B-B
\]

可以明确划分职责。

## 8.1 Binary Expert A

主要负责：

\[
A-W_A
\]

局部二元结合关系。

它最有资格判断：

- \(W_A\) 是否位于正确 pocket；
- local translation；
- local orientation；
- protein–warhead contact；
- local steric compatibility；
- A-side binding confidence。

但它**不应该单独决定整个 PROTAC 的最终姿态**。

---

## 8.2 Binary Expert B

对称地负责：

\[
W_B-B
\]

包括：

- \(W_B\) pocket；
- local orientation；
- contact geometry；
- local steric compatibility；
- B-side binding confidence。

同样不能独立决定完整 PROTAC。

---

## 8.3 Fuse

Fuse 应承担真正的 ternary-specific 问题：

### 1. Linker reconciliation

\[
W_A-L-W_B
\]

必须形成合理的连接结构。

### 2. A/B relative geometry

决定：

\[
A \leftrightarrow B
\]

的相对 translation / rotation。

### 3. Cross-side coupling

例如：

\[
A\text{-side information}
\rightarrow
B\text{-side decision}
\]

以及反方向的信息传递。

### 4. Conflict resolution

当：

\[
\Delta_A
\]

和

\[
\Delta_B
\]

存在冲突时，由 Fuse 判断如何折中或重新组织全局构象。

### 5. Global ternary consistency

确保整个：

\[
A-W_A-L-W_B-B
\]

是一个统一、物理合理的三元复合物，而不是两个二元复合物的机械拼接。

---

# 9. 推荐方案一：Fuse 直接产生唯一 Global Update

最直接的实现为：

\[
m_A=E_A(X_t)
\]

\[
m_B=E_B(X_t)
\]

然后：

\[
\Delta X_{\mathrm{global}}
=
F(X_t,m_A,m_B,t)
\]

最终：

\[
X_{t-\Delta t}
=
\operatorname{Update}
(X_t,\Delta X_{\mathrm{global}})
\]

Binary experts 完全不直接更新坐标。

这种方案简单、干净，而且能够从结构上消除“共享 PROTAC 被两个模块重复修改”的问题。

---

# 10. 推荐方案二：Residual Fuse（更推荐）

为了尽可能保留 pretrained binary experts 已经学好的 vector field，可以让 Fuse 不从零预测完整更新，而是学习：

\[
\boxed{
\text{binary prior 上的 ternary correction}
}
\]

两个专家首先输出：

\[
v_A
\]

和

\[
v_B
\]

通过适当的投影/坐标映射将它们转换到统一全局自由度：

\[
\tilde v_A=M_Av_A
\]

\[
\tilde v_B=M_Bv_B
\]

得到一个 binary prior：

\[
v_{\mathrm{prior}}
=
g_A\tilde v_A
+
g_B\tilde v_B
\]

Fuse 只预测：

\[
r_F
\]

即 ternary residual：

\[
r_F
=
F(
X_t,
m_A,
m_B,
t
)
\]

最终 vector field：

\[
\boxed{
v_{\mathrm{global}}
=
v_{\mathrm{prior}}
+
r_F
}
\]

然后统一更新：

\[
X_{t-\Delta t}
=
\operatorname{Integrate}
(X_t,v_{\mathrm{global}})
\]

这种设计的优点是：

1. 最大程度保留 binary expert 原有能力；
2. Fuse 不需要重新学习所有二元结合知识；
3. Fuse 专门学习“二元先验无法解释的三元耦合部分”；
4. 更容易解释；
5. 更容易做 ablation；
6. 更容易通过 residual regularization 防止 Fuse 破坏已有先验。

---

# 11. Confidence Gating：不同阶段不要相信两个专家同样多

两个 binary expert 在任意时刻不一定同样可靠。

定义：

\[
c_A,\quad c_B
\]

表示两个专家当前的 confidence。

Fuse 可以学习：

\[
g_A,g_B,g_F
\]

满足：

\[
g_A+g_B+g_F=1
\]

并令：

\[
[g_A,g_B,g_F]
=
G(
X_t,
m_A,
m_B,
c_A,
c_B,
t
)
\]

例如：

## Early stage

体系仍然非常 noisy：

\[
c_A,c_B\downarrow
\]

此时不应该过度相信某个局部专家。

---

## 一侧已经找到稳定 pocket

例如：

\[
c_A\gg c_B
\]

则：

\[
g_A\uparrow
\]

Fuse 应更多保留 A-side binary prior，同时让 B-side 和 linker 逐渐适配。

---

## 两侧都接近稳定

此时两个局部专家主要负责保持局部结合：

\[
A-W_A
\]

和：

\[
W_B-B
\]

Fuse 则主要负责：

- linker fine adjustment；
- protein–protein relative geometry；
- global clash elimination；
- ternary consistency。

---

# 12. Conflict-Aware Fuse

除了 confidence，还可以显式告诉 Fuse：

> 两个专家当前到底有多冲突。

例如可以定义 proposal conflict：

\[
C_{\mathrm{dir}}
=
1-
\cos(
\tilde v_A,
\tilde v_B
)
\]

或者：

\[
C_{\mathrm{norm}}
=
\|
\tilde v_A-\tilde v_B
\|
\]

还可以计算：

- warhead endpoint inconsistency；
- linker closure error；
- torsion incompatibility；
- predicted clash；
- local confidence disagreement。

这些量可以作为 Fuse 的输入：

\[
F(
X_t,
m_A,
m_B,
C_t,
t
)
\]

使模型显式知道：

\[
\boxed{
\text{什么时候两边意见一致，什么时候正在发生拔河}
}
\]

---

# 13. 不建议简单平均两个 Proposal

不能简单使用：

\[
v=
\frac12(v_A+v_B)
\]

因为两个 proposal 不是普通的独立噪声估计。

它们可能对应：

- 不同局部坐标系；
- 不同 binding confidence；
- 不同约束强度；
- 不同 diffusion stage；
- 完全相反的几何方向。

简单平均可能得到：

\[
\text{两个都不满足的中间态}
\]

因此应该由 learnable consensus / Fuse 决定如何组合。

---

# 14. 训练策略

## Stage 1：先冻结两个 Binary Expert

推荐首先：

\[
E_A,E_B
\]

保持 frozen。

只训练：

\[
F
\]

以及：

- gating；
- projection；
- ternary residual head；
- linker module；
- confidence / conflict module。

目的：

\[
\boxed{
\text{先保护 binary prior，只让 Fuse 学习 ternary correction}
}
\]

这也是判断整个架构是否成立的最干净实验。

---

## Stage 2：必要时小幅联合微调

如果 Stage 1 已经稳定，再考虑：

\[
E_A,E_B
\]

部分解冻。

例如：

- 只解冻最后几层；
- 极小 learning rate；
- Fuse 使用正常 learning rate；
- binary expert 使用 \(0.01\sim0.1\times\) Fuse 的 learning rate。

其核心目的仍然是：

\[
\text{adapt}
\]

而不是：

\[
\text{rewrite binary expert}
\]

---

# 15. 防止 Fuse 反过来破坏 Binary Prior

如果使用 residual Fuse：

\[
v_{\mathrm{global}}
=
v_{\mathrm{prior}}
+
r_F
\]

建议加入一定的 residual regularization：

\[
L_{\mathrm{res}}
=
\|r_F\|^2
\]

或者相对限制：

\[
L_{\mathrm{res}}
=
\frac{
\|r_F\|^2
}{
\|v_{\mathrm{prior}}\|^2+\epsilon
}
\]

意义是：

> Fuse 应当只在确实存在 ternary conflict 时进行较大修正，而不是无条件覆盖 binary expert。

注意该项不能过强，否则 Fuse 会失去解决真实三元耦合的能力。

---

# 16. Loss 设计建议

整体可以写成：

\[
L
=
L_{\mathrm{global}}
+
\lambda_{\mathrm{local}}L_{\mathrm{local}}
+
\lambda_{\mathrm{linker}}L_{\mathrm{linker}}
+
\lambda_{\mathrm{cons}}L_{\mathrm{consistency}}
+
\lambda_{\mathrm{res}}L_{\mathrm{res}}
\]

其中：

## Global denoising / flow loss

\[
L_{\mathrm{global}}
\]

监督最终唯一的 ternary vector field。

这是最核心的目标。

---

## Local binding preservation

\[
L_{\mathrm{local}}
=
L_A+L_B
\]

用于保护：

\[
A-W_A
\]

和：

\[
W_B-B
\]

的二元结合质量。

---

## Linker feasibility

\[
L_{\mathrm{linker}}
\]

约束：

- bond geometry；
- torsion；
- linker closure；
- chain continuity；
- steric validity。

---

## Ternary consistency

\[
L_{\mathrm{consistency}}
\]

用于保证：

\[
A-W_A-L-W_B-B
\]

整体的几何兼容性。

---

## Residual regularization

\[
L_{\mathrm{res}}
\]

防止 Fuse 无必要地覆盖 binary priors。

---

# 17. 推理过程推荐写法

每个 diffusion / flow step：

```text
Input:
    X_t = {A_t, PROTAC_t, B_t}

        │
        ├──────────────► Binary Expert A
        │                     │
        │                     ▼
        │                  m_A, v_A
        │
        ├──────────────► Binary Expert B
        │                     │
        │                     ▼
        │                  m_B, v_B
        │
        ▼
   Global / ternary features

m_A + m_B + X_t + t
        │
        ▼
      Fuse
        │
        ├── confidence / gating
        ├── conflict resolution
        ├── linker reconciliation
        └── ternary residual
        │
        ▼
v_global = binary_prior + ternary_residual
        │
        ▼
ONE integrator / ONE coordinate update
        │
        ▼
X_{t-Δt}
```

伪代码：

```python
for t in timesteps:

    proposal_A = expert_A(global_state, t)
    proposal_B = expert_B(global_state, t)

    global_update = fuse(
        global_state=global_state,
        proposal_A=proposal_A,
        proposal_B=proposal_B,
        timestep=t,
    )

    global_state = integrator(
        global_state,
        global_update,
        t,
    )
```

关键点是：

```python
expert_A(...)
expert_B(...)
```

只产生 proposal。

真正改变坐标的只有：

```python
integrator(...)
```

并且每个 step 只执行一次。

---

# 18. 推荐重点监控的诊断指标

为验证问题是否真正解决，不能只看总 loss。

建议单独监控：

### Binary-side quality

\[
Q_A,\quad Q_B
\]

例如：

- warhead RMSD；
- pocket contact；
- local clash；
- binding geometry。

### PROTAC quality

例如：

- PROTAC RMSD；
- linker geometry；
- torsion validity；
- molecular strain。

### Global ternary quality

例如：

- ternary RMSD；
- interface geometry；
- protein–protein relative pose；
- clash score。

### Expert conflict

\[
C_t
\]

例如：

\[
1-\cos(v_A,v_B)
\]

或其他 proposal disagreement 指标。

尤其值得观察：

\[
C_t
\]

是否随着 denoising / flow integration 逐渐下降。

一个健康的 ternary refinement 过程应该更接近：

\[
C_t\downarrow
\]

而不是：

\[
C_t\uparrow
\]

---

# 19. 建议的 Ablation

为了证明真正解决了问题，建议至少比较：

| 实验 | Binary Expert | Fuse | 更新方式                                |
| ---- | ------------- | ---- | --------------------------------------- |
| A    | ✓             | ×    | 两侧独立/简单组合                       |
| B    | ✓             | ✓    | 三个模块分别直接修改                    |
| C    | ✓             | ✓    | Proposal + Fuse 单次更新                |
| D    | ✓             | ✓    | Proposal + Residual Fuse                |
| E    | ✓             | ✓    | Residual Fuse + confidence gating       |
| F    | ✓             | ✓    | Residual Fuse + gating + conflict-aware |

最关键的是比较：

\[
B \quad vs \quad C
\]

因为它直接验证：

> **“多主体直接改坐标”与“单一全局 consensus update”之间的差异。**

随后：

\[
C \quad vs \quad D
\]

验证 residual formulation 是否能够更好地保留 binary prior。

---

# 20. EMA 在该架构中的位置

EMA 不解决上述架构性冲突，但可以提高训练后的稳定性。

如果：

- Fuse 可训练；
- projection / gating 可训练；
- binary experts 部分微调；

则可以对所有 **实际参与训练的参数** 维护 EMA。

标准方式仍然是：

```python
optimizer.step()
ema.update(model)
```

最终：

- raw model 用于继续训练；
- EMA model 用于 validation；
- best EMA checkpoint 用于最终推理。

但是必须强调：

\[
\boxed{
\text{EMA 只能平滑参数噪声，不能修复错误的多主体动力学。}
}
\]

因此应该先解决共享 PROTAC 的控制权问题，再讨论 EMA 带来的增益。

---

# 21. 最终推荐方案

综合来看，建议采用：

\[
\boxed{
\text{Frozen / weakly-finetuned Binary Experts}
+
\text{Proposal Interface}
+
\text{Residual Fuse}
+
\text{Confidence / Conflict Gating}
+
\text{Single Global Update}
}
\]

具体为：

\[
E_A(X_t)\rightarrow(m_A,v_A,c_A)
\]

\[
E_B(X_t)\rightarrow(m_B,v_B,c_B)
\]

然后：

\[
v_{\mathrm{prior}}
=
g_A M_Av_A
+
g_B M_Bv_B
\]

Fuse 预测：

\[
r_F
=
F(
X_t,
m_A,
m_B,
c_A,
c_B,
t
)
\]

最终：

\[
\boxed{
v_{\mathrm{global}}
=
v_{\mathrm{prior}}
+
r_F
}
\]

并且：

\[
\boxed{
X_{t-\Delta t}
=
\operatorname{Integrate}
(X_t,v_{\mathrm{global}})
}
\]

每个 step 只执行一次。

---

# 22. 一句话总结

当前失败的根本原因并不是两个 Binary Expert 不够强，也不是 Fuse 不够全面，而是：

\[
\boxed{
\text{两个局部专家同时获得了共享 PROTAC 的直接控制权}
}
\]

这导致局部目标冲突，并通过 PROTAC 质量下降形成正反馈死循环。

因此新的核心原则应该是：

\[
\boxed{
\textbf{Local experts propose; global Fuse decides.}
}
\]

Binary Expert 负责提供可靠的局部二元先验；

Fuse 负责理解两侧约束之间的冲突与三元耦合；

整个三元体系始终只维护一个全局状态，并在每个 diffusion / flow step 中只执行一次统一的几何更新。

这样才能同时获得：

\[
\boxed{
\text{Binary prior}
+
\text{Ternary coupling}
+
\text{Global geometric consistency}
}
\]

而不是让三个强模块互相破坏。
