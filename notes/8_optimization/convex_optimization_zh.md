# 凸优化

凸优化旨在凸可行集上最小化凸目标函数。其结构在选择算法之前提供了有用的保证：一个局部最优解同时也是全局最优解。支撑超平面不等式以及适用的 KKT 乘子将该几何性质转化为可数学验证的条件。核心陈述如下：

> 对于凸可行集上的凸目标函数，每一个局部最小值都是全局最小值。

这一事实将一阶或 KKT 条件转化为全局最优性的证明。

## 凸集

集合 $C\subseteq\mathbb{R}^n$ 是凸的，如果

$$
\theta x + (1 - \theta)y\in C
$$

对于每一个 $x,y\in C$ 和每一个 $\theta\in[0,1]$ 成立。

例子包括仿射子空间、半空间、欧几里得球、超立方体和多面体。

## 凸函数

函数 $f$ 是凸的，当且仅当

$$
f(\theta x + (1 - \theta)y) \le \theta f(x) + (1 - \theta)f(y).
$$

几何上，图上任意两点间的弦位于图的上方。

![Jensen inequality for a convex function](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/convex_jensen.svg)

对于可微函数 $f$，凸性等价于全局一阶下界

$$
f(y)\ge f(x) + \nabla f(x)^\top(y - x).
$$

对于定义在凸域上的二阶可微函数 $f$，充分且必要的条件是

$$
\nabla^2 f(x)\succeq0
$$

对于域中的每一个 $x$ 成立。

### 为何每一个局部最小值都是全局最小值

假设 $x^*$ 是一个局部最小值，但存在可行点 $y$ 使得 $f(y)<f(x^*)$。可行集的凸性保证了 $x_\theta=(1-\theta)x^*+\theta y$ 是可行的，而目标函数的凸性给出

$$
f(x_\theta)\leq(1-\theta)f(x^*)+\theta f(y)<f(x^*)
$$

对于每一个 $0<\theta\leq1$。取任意小的 $\theta$ 与局部最小性相矛盾。此证明依赖于两个凸性假设。

对于定义在凸可行集 $C$ 上的可微凸目标函数，最优性等价于

$$
\nabla f(x^*)^\top(y-x^*)\geq0\qquad\text{对于每一个 }y\in C.
$$

充分性直接由一阶下界得出。在无约束域中，这简化为 $\nabla f(x^*)=0$；在边界处通常不成立。

## 强凸性

可微函数是 $\mu$-强凸的，如果

$$
f(y)\ge f(x) + \nabla f(x)^\top(y - x) + \frac{\mu}{2}\| y - x\|_2^2.
$$

对于 $\mu>0$，强凸性意味着如果存在最小值，则该最小值是唯一的，并且允许为第一阶方法提供显式的收敛速率界限。定义在整个 $\mathbb R^n$ 上的连续强凸函数会达到其最小值；在受限的非闭域上，必须单独检查是否达到最小值。

## 平滑性与条件数

如果梯度是 $L$-Lipschitz 连续，

$$
\|\nabla f(x) - \nabla f(y)\|_2\le L\| x - y\|_2,
$$

那么步长为 $1/L$ 的梯度下降法满足标准的收敛保证。当 $f$ 既是 $\mu$-强凸又是 $L$-平滑时，比值

$$
\kappa = \frac{L}{\mu}
$$

扮演条件数的角色。

## 凸约束问题

标准凸问题的形式为

$$
\min_x f(x)
$$

受限于

$$
g_i(x)\le0,
\qquad Ax = b,
$$

其中 $f$ 和每一个 $g_i$ 都是凸的，且等式约束是仿射的。

![Convex level sets and a convex feasible region](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/convex_feasible_levels.svg)

在合适的约束规范（如 Slater 条件）下，KKT 条件是必要且充分的。

对于可微凸问题，任何满足 KKT 条件的点和乘子已经证明了全局最优性。Slater 条件等规范用于保证在最优解处存在此类乘子。Slater 条件的一种常见形式要求域相对内部的一个点满足所有仿射等式并严格满足所有不等式。

## 二次规划

凸二次规划的目标函数为

$$
q(x) = \frac12 x^\top Qx + c^\top x
$$

其中

$$
Q\succeq0.
$$

该存储库的实现检查 $Q$ 的特征值，然后使用投影梯度风格的迭代。详见 [二次规划](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/quadratic_programming.md)。

## 计算中的凸性检查

在采样点上随机测试 Jensen 不等式可以发现违反情况，但通过随机测试**不能**证明凸性。分析结构更可取：

- 已知的凸原子和组合规则；
- Hessian 半正定性；
- 表格外图论证；
- 保持凸性的变换。

因此，存储库中的 `is_convex_function` 助手应被视为数值诊断工具，而非证明。

## 计算示例

考虑

$$
f(x,y) = x^2 + 4y^2 + 2x - 8y.
$$

其 Hessian 为

$$
H=
\begin{bmatrix}
2&0\\
0&8
\end{bmatrix}\succ0,
$$

因此该函数是强凸的。将梯度设为零得到

$$
2x + 2 = 0,
\qquad 8y - 8 = 0,
$$

因此

$$
(x^*,y^*) = (-1,1).
$$

由于目标函数是凸的，该驻点自动即为唯一的全局最小值。

## 建模启示

在选择通用非线性求解器之前，询问问题是否可以表述为凸问题。正确的凸表述可以将看似困难的搜索问题转化为具有全局保证和可靠数值方法的问题。