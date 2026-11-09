# 约束优化

约束优化为决策变量添加了限制条件：

$$
\min_x f(x)
$$

受限于

$$
g_i(x)\le0,
\qquad h_j(x) = 0.
$$

约束条件定义了可行域。即使目标函数梯度非零，约束最优解也可能位于其边界上。

![可行域上的目标函数等高线](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/constrained_feasible_region.svg)

## 等式约束与相切

对于单个等式约束

$$
h(x) = 0,
$$

可行的一阶运动 $d$ 需满足

$$
\nabla h(x)^\top d = 0.
$$

在正则约束最优解处，没有任何可行切线方向能使目标函数减小。因此，目标函数梯度必须位于约束梯度的张成空间中：

$$
\nabla f(x^*) + \nu\nabla h(x^*) = 0.
$$

这就是拉格朗日乘子条件。

等式乘子 $\nu$ 可正可负。下文将 $\lambda_i$ 保留用于表示不等式 $g_i\leq0$ 的非负乘子。将这些约定区分开来，可使平稳性和灵敏度公式保持一致。

## 不等式约束

对于不等式 $g_i(x)\le0$，当 $g_i(x^*)=0$ 时，该约束在 $x^*$ 处是**活跃**的。非活跃约束具有局部松弛量，不必影响平稳性条件。

这一区别由互补松弛性捕捉：

$$
\lambda_i g_i(x^*) = 0.
$$

## KKT 条件

定义拉格朗日函数

$$
\mathcal{L}(x,\lambda,\nu) = f(x) + \sum_i\lambda_i g_i(x) + \sum_j\nu_j h_j(x).
$$

在合适的约束规范下，局部最优解必须满足

$$
\nabla_x\mathcal{L}(x^*,\lambda^*,\nu^*) = 0,
$$

$$
g_i(x^*)\le0,
\qquad h_j(x^*) = 0,
$$

$$
\lambda_i^*\ge0,
$$

以及

$$
\lambda_i^*g_i(x^*) = 0.
$$

![活跃边界处的 KKT 几何](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/constrained_kkt_geometry.svg)

对于具有凸不等式和仿射等式的可微凸问题，这些条件是全局最优性的充分条件。需要标准约束规范来确保最优解具有合适的乘子，而不是为了验证提供的 KKT 证书。

参见 [拉格朗日乘子与 KKT 条件](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/lagrange_multipliers_and_kkt.md) 以获取详细推导。

## 线性规划

当目标函数和约束均为线性时，问题变为线性规划。可行域是一个多面体，在具有非负变量的标准形式下，达到的有限最优解可以用一个最优极值点表示。一般的具有线性的多面体不一定具有任何极值点。

该仓库包含一个紧凑的两阶段单纯形法实现，可处理负右端项、等式约束，并报告不可行和无界问题。参见 [线性规划](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/linear_programming.md)。

## 数值 KKT 系统

等式约束牛顿法和 SQP 型方法会导致形如下列的鞍点线性系统：

$$
\begin{bmatrix}
H & J_h^\top\\
J_h & 0
\end{bmatrix}
\begin{bmatrix}
p\\
\Delta\nu
\end{bmatrix}
{}=
-\begin{bmatrix}
\nabla f+J_h^\top\nu\\
h(x)
\end{bmatrix}.
$$

当约束梯度近乎线性相关或问题缩放不良时，这些系统可能病态。

该仓库的等式约束例程精确求解此系统，其中 $H$ 为拉格朗日函数的海森矩阵（若提供解析形式则使用解析形式，否则使用中心差分估计），从最小二乘乘子估计开始，并缩减步长直至 KKT 残差减小。在正则解附近，这给出二次收敛。生产级求解器会增加不定 $H$ 的海森矩阵修正、 merit 函数以及不等式处理，而此紧凑例程省略了这些内容。

### 推导等式 KKT 线性系统

将等式平稳性和可行性方程收集为

$$
F(x,\nu)=\begin{bmatrix}\nabla f(x)+J_h(x)^\top\nu\\h(x)\end{bmatrix}.
$$

其对 $(x,\nu)$ 的雅可比矩阵正是显示的块矩阵，其中 $H=\nabla_{xx}^2\mathcal L$，且 $J_h$ 的行等于约束梯度。牛顿线性化 $F+DF\,(p,\Delta\nu)=0$ 给出鞍点求解。下块使步长满足 $h(x)+J_hp=0$，从而强制线性化约束；非线性约束通常在一次迭代后不能精确满足。

二次局部收敛需要非奇异 KKT 雅可比矩阵、足够平滑的导数、精确的线性求解以及最终的全步长接受。近乎相关的约束梯度即使目标函数海森矩阵条件数良好，也可能使该系统敏感。在 KKT 残差上的线搜索有助于控制迭代，但不能单独证明全局收敛或将平稳点分类为最小值。

## 计算示例

最小化

$$
f(x,y) = x^2 + y^2
$$

受限于

$$
x + y = 4.
$$

拉格朗日函数为

$$
\mathcal{L} = x^2 + y^2 + \nu(x + y - 4).
$$

平稳性给出

$$
2x + \nu = 0,
\qquad 2y + \nu = 0,
$$

因此 $x=y$。约束条件随后给出

$$
x = y = 2.
$$

因此约束最小值为

$$
f(2,2) = 8.
$$

此处 $\nu=-4$。通过参数化约束 $x=2+t$，$y=2-t$ 得到 $f=8+2t^2\geq8$，直接验证了沿可行线的全局最小性。等价地，目标函数海森矩阵 $2I$ 在每个非零切线方向 $(d,-d)$ 上都是正定的。

## 可行性与缩放

在实践中，需同时监控最优性和可行性。即使梯度很小，若违反约束，该点仍不可接受。

有用的残差包括

$$
r_{\text{eq}} = \| h(x)\|,
$$

$$
r_{\text{ineq}} = \|\max(g(x),0)\|,
$$

以及平稳性残差

$$
r_{\text{stat}} = \|\nabla_x\mathcal{L}\|.
$$