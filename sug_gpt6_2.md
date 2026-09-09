# model-v2-dual-consensus 设计方案

## 0. 分支定位

`model-v2-dual-consensus` 是 `model-v2` 的平行实验分支。

它不把“双 PROTAC”视为对主线的简单改造，而是采用完全不同的动力学假设：

\[
\boxed{
\text{Dual Hypothesis}
+
\text{Cross-conditioned Negotiation}
+
\text{Annealed Consensus}
+
\text{Single Physical Collapse}
}
\]

核心思想：

> 两个 Binary Expert 不再从第一步开始争夺同一个 PROTAC，而是各自维护一个局部最适 PROTAC hypothesis，在 Fuse 的持续信息交换与共识控制下，最终收敛为唯一的 ternary-compatible PROTAC。

一句话总结：

\[
\boxed{
\textbf{允许局部不同意，但不允许最终不一致。}
}
\]

---

# 1. 动机

原始单 PROTAC 模型中：

\[
E_A \rightarrow P
\]

\[
E_B \rightarrow P
\]

两个 binary experts 同时修改同一个 PROTAC。

由于两个专家分别优化：

\[
A-W_A
\]

和：

\[
W_B-B
\]

它们的局部最优方向一般并不一致：

\[
\Delta P_A \neq \Delta P_B
\]

因此会形成：

\[
A/B\text{ 同时争夺 PROTAC}
\rightarrow
P\text{ 变差}
\rightarrow
E_A,E_B\text{ 判断变差}
\rightarrow
\text{冲突进一步增强}
\]

即：

\[
\boxed{
\text{positive feedback instability}
}
\]

`model-v2-dual-consensus` 的出发点是：

> 既然两个二元专家天然对应两个不同的局部条件分布，就不要强迫它们从一开始共享同一个瞬时几何状态。

---

# 2. 核心状态定义

在每一个 diffusion / flow time \(t\)，维护：

\[
X_t=
\{
A_t,\,
P_A^t,\,
P_B^t,\,
B_t
\}
\]

其中：

\[
P_A^t
\]

表示 A-side expert 当前维护的 PROTAC hypothesis，

\[
P_B^t
\]

表示 B-side expert 当前维护的 PROTAC hypothesis。

两者必须满足：

\[
\boxed{
\text{shared chemistry, separate geometry}
}
\]

即它们共享：

- atom identity；
- bond graph；
- warhead assignment；
- linker topology；
- rotatable bond definition；
- molecular features；
- diffusion / flow time；
- chemical constraints。

但允许不同：

- rigid pose；
- torsion；
- internal reachable geometry；
- instantaneous local hypothesis。

---

# 3. 两个 Binary Expert 的职责

## 3.1 A-side Expert

输入：

\[
(A_t,P_A^t,t)
\]

输出：

\[
m_A,\,
v_A,\,
c_A
\]

其中：

- \(m_A\)：A-side latent / interaction features；
- \(v_A\)：A-side proposal vector field；
- \(c_A\)：A-side confidence / reliability。

A-side 主要负责：

\[
A-W_A
\]

局部结合几何。

---

## 3.2 B-side Expert

输入：

\[
(P_B^t,B_t,t)
\]

输出：

\[
m_B,\,
v_B,\,
c_B
\]

主要负责：

\[
W_B-B
\]

局部结合。

---

# 4. Responsibility Mask

不能让每个 binary expert 对整条 PROTAC 都具有同等控制权。

定义 atom-wise responsibility：

\[
r_A(i),\quad r_B(i)
\]

满足：

\[
r_A(i)+r_B(i)\approx1
\]

推荐：

### A-side warhead

\[
r_A(i)\approx1,\qquad r_B(i)\approx0
\]

### B-side warhead

\[
r_A(i)\approx0,\qquad r_B(i)\approx1
\]

### Linker 中部

\[
r_A(i)\approx r_B(i)\approx0.5
\]

因此：

\[
v_A^{\mathrm{eff}}
=
r_A\odot v_A
\]

\[
v_B^{\mathrm{eff}}
=
r_B\odot v_B
\]

目的：

\[
\boxed{
\text{让每个 expert 强控制自己最有资格判断的区域}
}
\]

同时减少“远端乱拉”。

---

# 5. Fuse 的角色重新定义

在本分支中，Fuse 不只是 feature fusion。

Fuse 是：

\[
\boxed{
\textbf{Negotiation + Compatibility + Consensus Controller}
}
\]

输入：

\[
F(
X_t,
m_A,
m_B,
v_A,
v_B,
c_A,
c_B,
t
)
\]

输出至少包括：

\[
\Delta_A^F,\quad
\Delta_B^F
\]

用于对两个 hypothesis 分别施加 ternary-aware correction。

可扩展输出：

\[
g_A,\quad g_B
\]

\[
C_t
\]

\[
q_{\mathrm{ternary}}
\]

分别表示：

- expert gating；
- conflict level；
- global ternary compatibility。

---

# 6. 双轨更新方程

最基本形式：

\[
P_A^{t-\Delta t}
=
P_A^t
+
v_A^{\mathrm{eff}}
+
\lambda_F(t)\Delta_A^F
\]

\[
P_B^{t-\Delta t}
=
P_B^t
+
v_B^{\mathrm{eff}}
+
\lambda_F(t)\Delta_B^F
\]

其中：

\[
\lambda_F(t)
\]

控制 Fuse correction 强度。

与原模型的区别：

\[
\boxed{
P_A\text{ 和 }P_B\text{ 不再直接争夺同一个 state}
}
\]

而是在两个局部轨迹之间进行协调。

---

# 7. Cross-conditioned Binary Experts

两个 expert 不应该完全隔离。

推荐：

\[
E_A
=
E_A(
A,P_A,t
\mid
\operatorname{sg}(m_B)
)
\]

\[
E_B
=
E_B(
P_B,B,t
\mid
\operatorname{sg}(m_A)
)
\]

其中：

\[
\operatorname{sg}
\]

表示 stop-gradient 或受控信息通路。

这样：

> A expert 可以“知道 B 在想什么”，但不能直接修改 B-side hypothesis。

反之亦然。

这相当于：

\[
\boxed{
\text{conditional binary diffusion / flow}
}
\]

而不是两个完全独立的 binary dockers。

---

# 8. Annealed Consensus

这是本分支的核心。

定义：

\[
\lambda_{\mathrm{cons}}(t)
\]

控制两个 PROTAC hypothesis 之间的一致性强度。

---

## 8.1 Early Stage

早期状态噪声较大：

\[
\lambda_{\mathrm{cons}}(t)\approx0
\]

允许：

\[
P_A^t\neq P_B^t
\]

目的：

> 让两个 binary experts 各自先找到合理的 binding basin。

---

## 8.2 Middle Stage

逐渐提高：

\[
\lambda_{\mathrm{cons}}(t)\uparrow
\]

开始要求：

- linker geometry compatible；
- warhead endpoint relation compatible；
- torsion distribution 接近；
- global relative pose 可协调。

---

## 8.3 Late Stage

后期：

\[
\lambda_{\mathrm{cons}}(t)\gg0
\]

要求：

\[
P_A^t
\rightarrow
P^\*
\]

\[
P_B^t
\rightarrow
P^\*
\]

最终只允许一个物理 PROTAC。

整体过程：

\[
\boxed{
\text{Independent hypotheses}
\rightarrow
\text{Soft negotiation}
\rightarrow
\text{Hard consensus}
}
\]

---

# 9. 不建议直接使用全原子坐标 RMSD 作为早期 Consensus

直接：

\[
L=
\|P_A-P_B\|^2
\]

会过早把两个 hypothesis 拉回同一个状态。

这可能使模型重新退化成单 PROTAC 的“争夺问题”。

因此推荐分层 consensus。

---

# 10. Hierarchical Consensus Loss

定义：

\[
L_{\mathrm{cons}}
=
\lambda_{\mathrm{int}}L_{\mathrm{internal}}
+
\lambda_{\mathrm{tor}}L_{\mathrm{torsion}}
+
\lambda_{\mathrm{end}}L_{\mathrm{endpoint}}
+
\lambda_{\mathrm{pose}}L_{\mathrm{pose}}
\]

---

## 10.1 Internal Consensus

约束两份 PROTAC 的内部化学结构：

\[
L_{\mathrm{internal}}
\]

包括：

- bond geometry；
- local distance matrix；
- ring geometry；
- reachable internal state。

---

## 10.2 Torsion Consensus

\[
L_{\mathrm{torsion}}
\]

约束：

\[
\tau_A
\approx
\tau_B
\]

但可采用周期角距离，而非普通 MSE。

---

## 10.3 Endpoint Consensus

\[
L_{\mathrm{endpoint}}
\]

关注：

\[
W_A
\leftrightarrow
W_B
\]

两端几何是否能够由一条真实 linker 连接。

---

## 10.4 Global Pose Consensus

后期才加入更强的：

\[
L_{\mathrm{pose}}
\]

要求两个完整 PROTAC hypothesis 在统一 ternary frame 中收敛。

---

# 11. Consensus Schedule

推荐不同 consensus 项随时间分阶段增强：

### Early

\[
\lambda_{\mathrm{internal}},
\lambda_{\mathrm{torsion}}
\]

为主。

### Middle

加入：

\[
\lambda_{\mathrm{endpoint}}
\]

### Late

显著提高：

\[
\lambda_{\mathrm{pose}}
\]

形成：

\[
\boxed{
\text{internal}
\rightarrow
\text{endpoint}
\rightarrow
\text{global pose}
}
\]

的渐进式收敛。

---

# 12. Confidence-controlled Negotiation

两个 expert 不应该始终五五开。

根据：

\[
c_A,\quad c_B
\]

学习：

\[
g_A,\quad g_B
\]

例如：

\[
g_A+g_B=1
\]

如果：

\[
c_A\gg c_B
\]

则：

\[
g_A\uparrow
\]

意味着：

> A-side 已经形成可靠 anchor，应优先保护 A-side local geometry，让 B-side 和 linker 去适配。

反之亦然。

---

# 13. Dynamic Anchor

进一步允许模型动态选择当前 anchor。

定义：

\[
a_t\in\{A,B,\mathrm{soft}\}
\]

例如：

\[
c_A-c_B>\delta
\Rightarrow
A\text{ anchor}
\]

\[
c_B-c_A>\delta
\Rightarrow
B\text{ anchor}
\]

否则：

\[
\text{soft negotiation}
\]

这样模型可以表现为：

\[
\boxed{
\text{谁先可靠，谁先固定}
}
\]

而不是强制固定某一侧作为唯一 anchor。

---

# 14. Conflict-aware Fuse

定义：

\[
C_t
\]

表示两个 hypothesis 当前冲突程度。

可以包括：

\[
C_{\mathrm{pose}}
\]

\[
C_{\mathrm{torsion}}
\]

\[
C_{\mathrm{endpoint}}
\]

\[
C_{\mathrm{velocity}}
\]

例如：

\[
C_{\mathrm{velocity}}
=
1-
\cos(
v_A^{\mathrm{eff}},
v_B^{\mathrm{eff}}
)
\]

Fuse 根据：

\[
C_t
\]

决定：

- 当前是否允许继续独立探索；
- 是否应该加强 consensus；
- 哪一侧需要让步；
- 是否需要重新调整 linker；
- 是否需要切换 anchor。

---

# 15. 最重要的数学风险：Consensus 不等于 Correctness

这是本分支最重要的理论风险。

即使：

\[
P_A=P_B
\]

也不能推出：

\[
P_A=P_B=P_{\mathrm{native}}
\]

两个 hypothesis 完全可能：

\[
P_A
\rightarrow
P_c
\leftarrow
P_B
\]

但：

\[
P_c\neq P_{\mathrm{native}}
\]

因此：

\[
\boxed{
\text{agreement}\neq\text{correctness}
}
\]

---

# 16. Local Interface Success 不等于 Ternary Success

即使：

\[
Q_A\approx1
\]

\[
Q_B\approx1
\]

仍不保证：

\[
Q_{\mathrm{ternary}}\approx1
\]

原因包括：

- A/B relative pose 错误；
- linker 构象错误；
- global orientation 错误；
- long-range SE(3) error amplification；
- steric clash；
- strain；
- protein–protein interface 错误；
- 两个局部 interface 虽各自好，但不属于同一个 native ternary basin。

因此本模型不能只训练：

\[
L_A+L_B+L_{\mathrm{cons}}
\]

必须存在独立：

\[
\boxed{
L_{\mathrm{global\ ternary}}
}
\]

---

# 17. Ternary Compatibility Objective

推荐 Fuse 额外输出：

\[
q_{\mathrm{ternary}}
\]

或 global compatibility embedding：

\[
z_T
\]

用于显式学习：

\[
\boxed{
P(T\mid I_A,I_B)
}
\]

其中：

- \(I_A\)：A-side interface hypothesis；
- \(I_B\)：B-side interface hypothesis；
- \(T\)：正确 ternary complex。

换句话说：

> Binary experts 解决“各自能不能结合”，Fuse 解决“这两个局部结合能不能同时成立”。

这应成为本分支最核心的理论定义之一。

---

# 18. Global Ternary Loss

定义：

\[
L_{\mathrm{global}}
\]

可以监督：

- A/B relative translation；
- A/B relative rotation；
- whole-complex RMSD；
- ternary DockQ surrogate；
- protein–protein interface；
- global clash；
- linker strain；
- native ternary contacts。

推荐：

\[
L
=
L_A
+
L_B
+
\lambda_{\mathrm{cons}}L_{\mathrm{cons}}
+
\lambda_TL_{\mathrm{global}}
\]

其中：

\[
L_T
\]

不能被设置得过弱。

---

# 19. Final Collapse：不能简单平均

最终不能：

\[
P^\*
=
\frac{P_A+P_B}{2}
\]

因为坐标平均可能产生：

- bond distortion；
- linker distortion；
- nonphysical geometry；
- 两侧都不满足。

推荐学习：

\[
\boxed{
P^\*
=
\Pi_{\mathrm{ternary}}
(
P_A,
P_B,
m_A,
m_B,
A,
B
)
}
\]

其中：

\[
\Pi_{\mathrm{ternary}}
\]

是 learned projection / collapse head。

它的任务是：

> 将两个 binary-compatible hypotheses 投影到真实 ternary-compatible manifold。

---

# 20. Projection 的输出形式

可选三种形式。

---

## 20.1 Direct coordinate projection

直接预测：

\[
X_{P^\*}
\]

简单，但物理约束风险较高。

---

## 20.2 Residual projection

选择其中一个 hypothesis 作为 anchor：

\[
P^\*
=
P_{\mathrm{anchor}}
+
r_{\Pi}
\]

更稳定。

---

## 20.3 Internal-coordinate projection

预测：

- rigid pose；
- torsion；
- linker internal parameters；

再通过确定性 reconstruction 得到：

\[
P^\*
\]

最符合当前 DiffDock / torsional state-space 逻辑。

优先推荐第 2 或第 3 种。

---

# 21. Reachability 契约仍必须遵守

双 PROTAC 并不能解决：

\[
x_{\mathrm{native}}
\notin
\mathcal M(x_{\mathrm{ETKDG}})
\]

因此该分支仍必须采用：

\[
x_{\mathrm{target}}^{\mathrm{reachable}}
\]

而不能回到：

\[
x_{\mathrm{ETKDG}}
\rightarrow
x_{\mathrm{native}}
\]

的不可达监督。

也就是说：

\[
\boxed{
\text{Dual consensus 解决 expert conflict，
不解决 state-space mismatch}
}
\]

---

# 22. POI Rotation 问题仍需独立验证

双 hypothesis 不会自动解决：

\[
\cos_{\mathrm{POI,rot}}
:
0.332\rightarrow0.026
\]

因此仍必须独立监控：

\[
L_{\mathrm{POI,rot}}
\]

\[
\cos_{\mathrm{POI,rot}}
\]

\[
\theta_{\mathrm{POI,rot}}
\]

并做：

\[
N=1
\]

及：

\[
N=32/64
\]

的小数据 overfit sanity test。

---

# 23. 推荐训练阶段

---

## Stage 0：Single-side Validation

先分别验证：

\[
(A,P_A)
\]

和：

\[
(P_B,B)
\]

确保双状态复制后 binary expert 原始性能没有明显下降。

---

## Stage 1：Independent Dual Hypothesis

暂时：

\[
\lambda_{\mathrm{cons}}=0
\]

只验证两个 hypothesis 能否各自稳定工作。

此阶段不要求最终 ternary 成功。

目的：

\[
\boxed{
\text{确认双状态确实消除了直接共享 PROTAC 的破坏}
}
\]

---

## Stage 2：Feature Exchange

加入：

\[
m_A\leftrightarrow m_B
\]

但仍不强制 geometry consensus。

验证：

> 仅靠跨侧信息是否已经改善 local predictions。

---

## Stage 3：Soft Consensus

加入较弱：

\[
L_{\mathrm{internal}}
+
L_{\mathrm{torsion}}
+
L_{\mathrm{endpoint}}
\]

暂时不强迫完整 pose collapse。

---

## Stage 4：Annealed Consensus

引入随 \(t\) 变化的：

\[
\lambda_{\mathrm{cons}}(t)
\]

形成：

\[
\text{independent}
\rightarrow
\text{negotiation}
\rightarrow
\text{convergence}
\]

---

## Stage 5：Global Ternary Objective

强化：

\[
L_{\mathrm{global}}
\]

防止：

\[
\text{一致但错误}
\]

---

## Stage 6：Learned Collapse

训练：

\[
\Pi_{\mathrm{ternary}}
\]

将：

\[
P_A,P_B
\]

压缩成最终唯一：

\[
P^\*
\]

---

## Stage 7：Confidence / Anchor / Conflict-aware

最后才加入：

- confidence gating；
- dynamic anchor；
- conflict-aware Fuse；
- adaptive consensus schedule。

避免第一版系统过于复杂。

---

# 24. 第一版最简实现

第一版建议故意做简单。

维护：

\[
P_A,P_B
\]

两个独立状态。

Binary experts：

\[
v_A=E_A(A,P_A,t)
\]

\[
v_B=E_B(P_B,B,t)
\]

Fuse：

\[
\Delta_A^F,\Delta_B^F
=
F(
P_A,P_B,m_A,m_B,t
)
\]

更新：

\[
P_A'
=
P_A
+
v_A
+
\lambda_F\Delta_A^F
\]

\[
P_B'
=
P_B
+
v_B
+
\lambda_F\Delta_B^F
\]

loss：

\[
L
=
L_A
+
L_B
+
\lambda_{\mathrm{cons}}(t)
L_{\mathrm{cons}}
+
\lambda_TL_{\mathrm{global}}
\]

先不做：

- dynamic anchor；
- learned gating；
- adaptive conflict scheduler；
- complicated collapse。

先验证：

\[
\boxed{
\text{Dual hypothesis 本身是否真的减弱 shared-PROTAC conflict}
}
\]

---

# 25. 必须通过的 Sanity Checks

## Sanity 1：Binary Preservation

双状态后：

\[
Q_A^{\mathrm{dual}}
\approx
Q_A^{\mathrm{binary}}
\]

\[
Q_B^{\mathrm{dual}}
\approx
Q_B^{\mathrm{binary}}
\]

---

## Sanity 2：No Direct Cross-write

A expert 不能直接修改：

\[
P_B
\]

B expert 不能直接修改：

\[
P_A
\]

跨侧只能通过：

\[
m_A,m_B,F
\]

实现。

---

## Sanity 3：Consensus Schedule Works

要求：

Early：

\[
d(P_A,P_B)\text{ 可以较大}
\]

Late：

\[
d(P_A,P_B)\downarrow
\]

而不是从第一步就被强制贴合。

---

## Sanity 4：Consensus 不牺牲 Local Docking

随着：

\[
d(P_A,P_B)\downarrow
\]

不能出现：

\[
Q_A\downarrow,\quad Q_B\downarrow
\]

否则说明又形成了“为了统一而互相破坏”。

---

## Sanity 5：Global Ternary Improves

必须检查：

\[
d(P_A,P_B)\downarrow
\]

是否同时对应：

\[
Q_{\mathrm{ternary}}\uparrow
\]

否则说明：

\[
\boxed{
\text{model learns agreement but not correctness}
}
\]

---

## Sanity 6：POI Rotation Learnable

和主线一样：

\[
\cos_{\mathrm{POI,rot}}
\]

必须单独通过小数据 overfit。

---

# 26. 推荐监控指标

至少记录：

### Local

\[
Q_A,\quad Q_B
\]

### Balance

\[
Q_{\min}
=
\min(Q_A,Q_B)
\]

### Mean

\[
Q_{\mathrm{mean}}
=
\frac{Q_A+Q_B}{2}
\]

### Joint Success

\[
SR_{\mathrm{both}}
=
P(
Q_A>\tau
\land
Q_B>\tau
)
\]

### Consensus

\[
D_{\mathrm{cons}}
=
d(P_A,P_B)
\]

### Global

- ternary RMSD；
- A/B relative pose error；
- linker strain；
- clash；
- ternary DockQ / global compatibility score。

### Dynamics

\[
C_t
\]

expert / hypothesis conflict。

### Rotation

\[
\cos_{\mathrm{POI,rot}}
\]

---

# 27. 推荐 Ablation

| ID | Dual P | Cross-info | Consensus | Global loss | Gating | Collapse |
|---|---|---|---|---|---|---|
| A | × | - | - | ✓ | - | - |
| B | ✓ | × | × | × | × | simple |
| C | ✓ | ✓ | × | × | × | simple |
| D | ✓ | ✓ | soft | × | × | simple |
| E | ✓ | ✓ | annealed | ✓ | × | learned |
| F | ✓ | ✓ | annealed | ✓ | ✓ | learned |
| G | ✓ | ✓ | conflict-aware | ✓ | ✓ | learned |

最关键比较：

\[
A\ vs\ B
\]

验证：

> 双 PROTAC 是否确实缓解共享状态冲突。

\[
B\ vs\ C
\]

验证：

> 信息交换本身是否有效。

\[
C\ vs\ D
\]

验证：

> soft consensus 是否有益。

\[
D\ vs\ E
\]

验证：

> global ternary objective 是否解决“两个界面好但整体不对”。

---

# 28. 与 model-v2 的核心区别

## model-v2

\[
\boxed{
\text{One state}
+
\text{multiple proposals}
+
\text{one global decision}
}
\]

哲学：

> 冲突在更新前解决。

---

## model-v2-dual-consensus

\[
\boxed{
\text{Two hypotheses}
+
\text{controlled negotiation}
+
\text{late convergence}
}
\]

哲学：

> 允许局部轨迹暂时分歧，再逐步解决冲突。

---

# 29. 两条路线真正要回答的问题

`model-v2` 回答：

\[
\boxed{
\text{共享 PROTAC 本身是否没问题，只是原来的更新方式错了？}
}
\]

`model-v2-dual-consensus` 回答：

\[
\boxed{
\text{两个 binary priors 是否本来就应该拥有独立的中间几何 hypothesis？}
}
\]

两者不是版本替代关系，而是两种不同的建模假设。

---

# 30. 潜在优势

`model-v2-dual-consensus` 最有可能带来的优势：

1. 避免两个 experts 直接争夺同一个 PROTAC；
2. 更好保护 pretrained binary expert 的 in-distribution behavior；
3. 允许两个局部 binding basin 先独立形成；
4. Fuse 的任务从 repair 变成 negotiation；
5. 支持 confidence-aware / anchor-aware 决策；
6. 更适合 multi-hypothesis / multi-walker 扩展；
7. 可以显式建模：

\[
P(T\mid I_A,I_B)
\]

而不只是做 feature fusion。

---

# 31. 最大风险

最需要警惕：

### 1. Consensus collapse

\[
P_A=P_B
\]

但：

\[
P_A\neq P_{\mathrm{native}}
\]

---

### 2. Two perfect binaries, bad ternary

\[
Q_A\uparrow,\quad Q_B\uparrow
\]

但：

\[
Q_{\mathrm{ternary}}\downarrow
\]

---

### 3. Consensus too early

过早：

\[
\lambda_{\mathrm{cons}}\uparrow
\]

会退化回单状态争夺。

---

### 4. Consensus too late

两边各自陷入不同 basin，Fuse 无法重新拼合。

---

### 5. State duplication increases optimization complexity

状态维度增大，训练可能更不稳定。

---

### 6. Collapse head becomes bottleneck

如果最终：

\[
\Pi_{\mathrm{ternary}}
\]

过弱，两个 hypothesis 都很好也可能无法合成正确结构。

---

# 32. 推荐成功判据

不应只看：

\[
D_{\mathrm{cons}}\downarrow
\]

真正成功应该同时满足：

\[
\boxed{
Q_A\uparrow
}
\]

\[
\boxed{
Q_B\uparrow
}
\]

\[
\boxed{
D_{\mathrm{cons}}\downarrow
}
\]

\[
\boxed{
Q_{\mathrm{ternary}}\uparrow
}
\]

并且：

\[
\boxed{
\cos_{\mathrm{POI,rot}}\text{ 不再随训练退化}
}
\]

即：

\[
\boxed{
\text{local quality}
+
\text{agreement}
+
\text{global correctness}
}
\]

三者同时成立。

---

# 33. 最终推荐的成熟架构

最终可写成：

\[
(m_A,v_A,c_A)
=
E_A(
A,P_A,t
\mid m_B
)
\]

\[
(m_B,v_B,c_B)
=
E_B(
P_B,B,t
\mid m_A
)
\]

Fuse：

\[
(
\Delta_A^F,
\Delta_B^F,
g_A,
g_B,
C_t,
q_T
)
=
F(
A,
P_A,
P_B,
B,
m_A,
m_B,
c_A,
c_B,
t
)
\]

双轨更新：

\[
P_A'
=
P_A
+
r_A\odot v_A
+
\lambda_F(t)\Delta_A^F
\]

\[
P_B'
=
P_B
+
r_B\odot v_B
+
\lambda_F(t)\Delta_B^F
\]

共识：

\[
\lambda_{\mathrm{cons}}(t)
\uparrow
\]

最终：

\[
\boxed{
P^\*
=
\Pi_{\mathrm{ternary}}
(
P_A,
P_B,
m_A,
m_B,
A,
B
)
}
\]

输出唯一：

\[
\boxed{
A-P^\*-B
}
\]

---

# 34. 一句话总结

`model-v2-dual-consensus` 的核心不是：

> “复制两份 PROTAC。”

而是：

\[
\boxed{
\textbf{
让两个 binary experts 保留各自的局部几何假设，
通过持续信息交换、动态置信度和逐步增强的共识约束，
最终把两个 binary-compatible solutions 投影到同一个
ternary-compatible physical solution。
}
}
\]

因此它真正要验证的科学假设是：

\[
\boxed{
\text{局部不同意是否应该被视为一种需要保留的信息，而不是一种必须立即消除的错误。}
}
\]
