# 优化

优化提出了一个简单但包含许多数值细微差别的问题：

> 在所有允许的选项中，哪一个给出了最小（或最大）的目标函数值？

最小化问题写作

$$
\min_{x\in\Omega} f(x),
$$

其中 $x\in\mathbb{R}^n$ 是决策向量，$f$ 是目标函数，$\Omega$ 是可行集。最大化问题总可以重写为 $-f$ 的最小化问题。

优化问题的难度不仅取决于其维度，更取决于其**几何性质**：凸性、光滑性、条件数、约束条件以及局部极小值的数量共同决定了哪些算法是合适的。

![非凸景观上的局部和全局极小值](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/optimization_landscape.svg)
## 一个有用的分类

| 属性 | 典型后果 | 常用方法 |
| --- | --- | --- |
| 光滑、无约束、凸 | 每个局部极小值都是全局极小值 | 梯度法、牛顿法、BFGS |
| 光滑、无约束、非凸 | 驻点可能是极小值、极大值或鞍点 | 带有保护措施的局部方法、多起点法 |
| 约束 | 迭代点必须满足或惩罚可行性 | 投影法、主动集法、KKT、SQP |
| 无导数/黑盒 | 梯度可能不可用或不可靠 | 网格搜索、模拟退火、进化算法 |
| 多峰 | 许多局部极小值相互竞争 | 全局或随机搜索 |

现有的实现章节围绕四个大类组织：

- [局部优化](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/local_optimization.md)
- [凸优化](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/convex_optimization.md)
- [约束优化](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/constrained_optimization.md)
- [全局优化](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/global_optimization.md)

详细笔记涵盖了这些家族内部的算法：

- [线搜索](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/line_search.md)
- [牛顿法和拟牛顿法](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/newton_and_quasi_newton.md)
- [线性规划](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/linear_programming.md)
- [拉格朗日乘子法和 KKT 条件](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/lagrange_multipliers_and_kkt.md)
- [二次规划](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/quadratic_programming.md)
- [模拟退火](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/simulated_annealing.md)
- [遗传算法](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/genetic_algorithms.md)

关于基本的最速下降法，请参阅早期的 [梯度下降法笔记](https://github.com/djeada/Numerical-Methods/blob/master/notes/1_root_and_extrema_finding/gradient_descent.md)。
## 一阶最优性条件

对于无约束的可微目标函数，位于内部的局部极小值点 $x^*$ 必须满足

$$
\nabla f(x^*) = 0.
$$

该条件是必要的，而非充分的。驻点也可能是极大值点或鞍点。

对于二阶可微函数，海森矩阵

$$
H(x) = \nabla^2 f(x)
$$

有助于对驻点进行分类。如果 $H(x^*)$ 是正定的，则 $x^*$ 是严格的局部极小值点。如果它是负定的，则 $x^*$ 是严格的局部极大值点。不定海森矩阵则表明存在鞍点。

### 奇异海森矩阵的含义与局限

在内部局部极小值点处，二阶可微目标函数的梯度为零且海森矩阵半正定。正定性是严格局部极小值的充分条件，但不是必要条件。在一维情况下，$x^4$、$-x^4$ 和 $x^3$ 在零点处的一阶和二阶导数均为零，但零点分别是极小值、极大值以及既非极大也非极小。因此，半定或零海森矩阵需要借助高阶分析或直接分析。

存在性是另一个问题。在非空紧可行集上的连续目标函数必能取得极小值。若无此类条件，下确界可能无法取得，例如 $f(x)=e^x$ 在 $\mathbb R$ 上的情况，或者目标函数可能无下界。当数学问题本身不存在解时，算法无法返回一个已取得的极优解。
## 为什么条件数很重要

考虑二次函数

$$
f(x) = \frac12 x^\top A x - b^\top x,
$$

其中 $A$ 为对称正定矩阵。其等高线为椭圆，条件数

$$
\kappa_2(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}
$$

衡量了椭圆的拉长程度。

当 $\kappa_2(A)$ 较大时，最速下降法倾向于在狭窄的谷底之字形摆动，而不是直接朝向最小值移动。

![Conditioning changes the path of gradient descent](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/optimization_conditioning.svg)

对于二次函数，固定步长的梯度下降法

$$
x_{k+1} = x_k - \alpha(Ax_k - b)
$$

在满足

$$
0<\alpha< \frac{2}{\lambda_{\max}(A)}
$$

时收敛。

使用针对谱的最佳固定步长，收缩因子依赖于 $\kappa$；较差的条件性直接减缓收敛速度。

为了推导该因子，从梯度更新中减去 $Ax^*=b$：$e_{k+1}=(I-\alpha A)e_k$。在正交归一化的特征基中，每个误差分量乘以 $1-\alpha\lambda_i$。平衡极端因子可得

$$
\alpha_* = \frac{2}{\lambda_{\min}+\lambda_{\max}},\qquad q_* = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}}=\frac{\kappa-1}{\kappa+1}.
$$

对于较大的 $\kappa$，该因子接近 1，因此需要多次微小的误差缩减。变量缩放和预条件旨在改善该谱。
## 局部与全局保证

**局部最小值** $x^*$ 满足

$$
f(x^*)\le f(x)
$$

对于 $x^*$ 附近的某个邻域内的所有 $x$。**全局最小值** 对每个可行点都满足该不等式。

凸性连接了这两个概念：如果 $f$ 是凸函数且 $\Omega$ 是凸集，则每个局部最小值都是全局最小值。如果没有凸性，局部求解器只能在适当假设下保证收敛到附近的驻点。
## 约束改变最优性条件

对于

$$
\min_x f(x)
$$

受限于

$$
g_i(x)\le0,
\qquad h_j(x) = 0,
$$

我们通常不能令 $\nabla f=0$。梯度可能指向可行域之外。相反，活跃约束的梯度通过拉格朗日函数和 KKT 条件引入。
## 选择方法

实用的决策顺序如下：

1. **优先利用结构。** 确定问题是凸的、二次的、线性的、可分离的、稀疏的，或是其他特殊形式。
2. **在可信时使用导数。** 它们通常能大幅减少函数求值次数。
3. **缩放变量和约束。** 不良缩放可能模拟出困难的几何结构。
4. **使用全局化保障措施。** 线搜索和信赖域思想可防止牛顿类方法在远离解时采取破坏性步长。
5. **将全局方法保留给真正多模态的问题。** 它们通常昂贵得多。
6. **验证结果。** 检查梯度、KKT 残差、可行性和敏感性，而不仅仅依赖求解器的成功标志。
## 停止准则

常见的终止测试包括

$$
\|\nabla f(x_k)\|\le\varepsilon_g,
$$

$$
\| x_{k+1} - x_k\|\le\varepsilon_x(1 + \| x_k\|),
$$

以及

$$
|f(x_{k+1}) - f(x_k)|\le\varepsilon_f(1 + |f(x_k)|).
$$

没有单一的准则在每个问题中都可靠。小步长可能是由于步长选择不当造成的，而在梯度并不小的平坦高原上，目标函数值的变化也可能很小。