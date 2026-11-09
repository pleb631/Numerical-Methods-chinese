# 局部优化

局部优化方法利用目标函数在当前迭代点附近的局部信息来改进初始点。由于它们能以远少于全局搜索的评估次数实现快速收敛，因此是光滑数值优化的默认工具。

根本性的权衡在于：在非凸目标函数上，解可能依赖于初始点。

## 下降方向

对于可微目标函数，若向量 $p_k$ 满足

$$
\nabla f(x_k)^\top p_k<0,
$$

则称 $p_k$ 为 $x_k$ 处的下降方向。

更新步骤具有通用形式

$$
x_{k+1} = x_k + \alpha_k p_k,
$$

其中 $\alpha_k>0$ 为步长。

最速下降法选择

$$
p_k = -\nabla f(x_k).
$$

牛顿法求解

$$
\nabla^2 f(x_k)p_k = -\nabla f(x_k),
$$

而 BFGS 使用近似逆海森矩阵 $H_k$：

$$
p_k = -H_k\nabla f(x_k).
$$

### 验证方向是否真正下降

沿 $p$ 的泰勒展开给出 $f(x+\alpha p)=f(x)+\alpha\nabla f(x)^\top p+o(\alpha)$，当 $\alpha\to0^+$ 时。因此，负的下降方向导数保证了对于足够小的正步长会有改进，尽管这并不能证明全步长是安全的。

对于最速下降法，只要 $g\ne0$，就有 $g^\top p=-\|g\|_2^2<0$。对于海森矩阵 $B$ 为正定的牛顿法，$p=-B^{-1}g$ 给出 $g^\top p=-g^\top B^{-1}g<0$。当 BFGS 的逆近似为正定时，同样的论证也适用。不定曲率矩阵会破坏这一保证，并解释了引入保护措施的必要性。

在收敛时，小梯度仅建立近似驻点，本身并不能确立全局甚至局部最小值。在约束问题中，适当的测试是投影梯度或 KKT 残差，因为在边界最优解处普通梯度不必为零。

## 罗森布罗克谷

罗森布罗克函数

$$
f(x,y) = (1 - x)^2 + 100(y - x^2)^2
$$

在 $(1,1)$ 处具有唯一的全球最小值，但其狭窄的弯曲谷地使得那些未考虑曲率的方法难以处理。

![Representative local-optimization paths on the Rosenbrock function](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/local_rosenbrock_paths.svg)

梯度下降法计算廉价但常呈之字形路径。牛顿法和拟牛顿法则花费更多计算量来学习谷地的几何结构。

## 收敛速率

在足够正则的最小值附近，常见的渐近收敛速率如下：

- 梯度下降：在强凸性下配合合适的步长，为**线性**收敛；
- BFGS：配合合适的线搜索，通常为**超线性**收敛；
- 牛顿法：当海森矩阵非奇异且迭代点足够接近时，为**二次**收敛。

若 $e_k=\|x_k-x^*\|$，这些理想化的速率表现为

$$
e_{k+1}\le c e_k,
$$

$$
\frac{e_{k+1}}{e_k}\to0,
$$

以及

$$
e_{k+1}\le c e_k^2,
$$

分别对应。

对于线性收敛，因子满足 $0<c<1$。在二次界中，固定的正常数不必小于 1：正是已经很小的误差的平方带来了快速缩减。这些是局部的渐近陈述。由于真实最小值在求解过程中通常未知，实际诊断使用梯度、接受的步长和残差，而不是假设误差 $e_k$ 可直接观测。

![Idealized local convergence rates](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/local_convergence_rates.svg)

## 为何普通牛顿法可能失败

当海森矩阵不定（indefinite）时，牛顿方向不能保证是下降方向。远离最小值时，它可能指向最大值、鞍点，或者仅仅采取巨大的步长。

因此，实用的牛顿型求解器会使用一种或多种保护措施：

- [线搜索](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/line_search.md)；
- 海森矩阵修正或阻尼；
- 信赖域；
- 拟牛顿正定近似。

该存储库的实现刻意保持算法紧凑：固定步长梯度下降、使用 `numpy.linalg.solve` 的普通牛顿法，以及带有强沃尔夫线搜索（先括区间后二分缩放，参见 [线搜索](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/line_search.md)）的 BFGS。

## BFGS 更新

定义

$$
s_k = x_{k+1} - x_k,
\qquad y_k = \nabla f(x_{k+1}) - \nabla f(x_k),
$$

以及

$$
\rho_k = \frac{1}{y_k^\top s_k}.
$$

逆海森 BFGS 更新公式为

$$
H_{k+1} = (I - \rho_k s_k y_k^\top)H_k(I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top.
$$

若 $H_k$ 为正定且 $y_k^\top s_k>0$，则更新保持正定性。

推导及数值解释参见 [牛顿和拟牛顿方法](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/newton_and_quasi_newton.md)。

## 二次函数工作示例

对于

$$
f(x,y) = \frac12(8x^2 + 2y^2) - 4x - 2y,
$$

我们有

$$
\nabla f=
\begin{bmatrix}
8x-4\\
2y-2
\end{bmatrix},
\qquad
H=
\begin{bmatrix}
8&0\\
0&2
\end{bmatrix}.
$$

最小值解为 $H x=b$：

$$
(x^{\ast}, y^{\ast}) = (0.5, 1).
$$

牛顿法从任意起点出发，因二次模型即函数本身，故只需一步精确到达。梯度下降法通常需要多步，其速率由特征值比 $8/2=4$ 控制。

## 实用诊断

对于局部求解器，除最终点外还需报告更多信息。有用的诊断包括：

- 梯度范数；
- 目标函数减少量；
- 接受的步长；
- 函数和梯度评估次数；
- 海森矩阵特征值或拟牛顿曲率测试；
- 对初始点的敏感性。