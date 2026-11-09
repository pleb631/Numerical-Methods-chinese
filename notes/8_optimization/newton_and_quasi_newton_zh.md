# 牛顿法与拟牛顿法

牛顿型方法利用局部曲率将梯度转化为更优缩放方向的搜索方向。

## 牛顿步

在 $x_k$ 处，用其二次泰勒模型近似目标函数

$$
m_k(p) = f(x_k) + g_k^\top p + \frac12 p^\top B_kp,
$$

其中

$$
g_k = \nabla f(x_k),
\qquad B_k = \nabla^2 f(x_k).
$$

将模型的梯度设为零可得

$$
B_kp_k = -g_k.
$$

实现时应求解该线性方程组；不应显式构造 $B_k^{-1}$。

## 牛顿法为何能二次收敛

在非退化极小值点附近，海森矩阵变化平滑，二次模型变得高度精确。在标准假设下，

$$
\|x_{k+1}-x^*\|_2\leq C\|x_k-x^*\|_2^2
$$

这种快速的局部收敛是牛顿法的主要吸引力。

### 更明确的二次收敛论证

设 $x^*$ 的梯度为零且海森矩阵非奇异，记 $e=x-x^*$，令 $B(x)=\nabla^2 f(x)$。沿线段应用微积分基本定理可得

$$
\nabla f(x)=\int_0^1B(x^*+te)e\,dt.
$$

因此，完整的牛顿步产生的误差为

$$
e_{\mathrm{new}}=B(x)^{-1}\int_0^1[B(x)-B(x^*+te)]e\,dt.
$$

若 $B$ 在局部满足利普希茨条件且常数为 $M$，且附近 $\|B(x)^{-1}\|\leq K$，则 $\|e_{\mathrm{new}}\|\leq KM\|e\|^2/2$。从足够近处开始可保持迭代在该邻域内。这证明了局部二次收敛界；阻尼步、不精确的导数或不定驻点需要单独解释。

在二次模型中，仅当海森矩阵正定时，将模型梯度设为零才能确定其极小值点。本文注中，$B_k$ 表示海森矩阵或其近似，而 $H_k$ 表示逆海森矩阵近似。保持这两种角色的区分可避免错误地对海森矩阵应用逆更新。

## 不定海森矩阵

若 $B_k$ 不定，牛顿方向可能满足

$$
g_k^\top p_k>0,
$$

这意味着它是一个上升方向。修正的牛顿法通过修改海森矩阵或使用信赖域来恢复可靠的步长。

## 拟牛顿思想

拟牛顿法避免使用精确的二阶导数。相反，它们根据迭代和梯度的变化构建矩阵近似：

$$
s_k = x_{k+1} - x_k,
\qquad y_k = g_{k+1} - g_k.
$$

割线方程为

$$
B_{k+1}s_k = y_k.
$$

它是将导数替换为有限割线斜率的多维类比。

![Newton curvature versus a BFGS approximation](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/newton_quasi_newton.svg)

## BFGS

对于逆海森矩阵近似 $H_k$，BFGS 使用

$$
H_{k+1} = (I - \rho s y^\top)H_k(I - \rho y s^\top) + \rho ss^\top,
$$

其中

$$
\rho = \frac{1}{y^\top s}.
$$

当 $y^\top s>0$ 时，正定性得以保持。满足沃尔夫曲率条件的线搜索保证每一步 $y^\top s>0$，这就是为何 BFGS 与 [Wolfe 线搜索](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/line_search.md) 配对而非简单的回溯。

### 验证 BFGS 割线方程

梯度变化是平均曲率作用，

$$
y_k=\int_0^1\nabla^2f(x_k+ts_k)s_k\,dt,
$$

这促使 $B_{k+1}s_k=y_k$，或等价地 $H_{k+1}y_k=s_k$。为检查逆更新，注意 $(I-\rho ys^\top)y=0$ 且 $\rho ss^\top y=s$（因为 $\rho(y^\top s)=1$）。代入可得 $H_{k+1}y=s$ 精确成立。

对于任意向量 $z$，更新还给出

$$
z^\top H_{k+1}z=w^\top H_kw+\rho(s^\top z)^2,\qquad w=(I-\rho ys^\top)z.
$$

若 $H_k\succ0$ 且 $y^\top s>0$，两项均非负，且仅当 $z=0$ 时才能同时为零。因此正定性得以保持。当 $y^\top s$ 很小时，计算其倒数可能放大舍入误差；即使精确曲率条件为正，跳过或阻尼更新也是一种数值保障措施。

## 成本比较

对于具有 $n$ 个变量的稠密问题：

- 精确海森矩阵存储成本为 $O(n^2)$；
- 稠密牛顿求解成本约为 $O(n^3)$；
- 稠密 BFGS 存储 $O(n^2)$ 但避免显式二阶导数；
- 有限内存 BFGS 利用仅最近更新对将存储降至 $O(mn)$。

## 何时优先选用各方法

当精确海森矩阵可负担且问题规模不大时使用牛顿法。当梯度可用但海森矩阵不便时使用 BFGS。对于非常大的问题，优先选用稀疏牛顿 -  Krylov 或有限内存变体。