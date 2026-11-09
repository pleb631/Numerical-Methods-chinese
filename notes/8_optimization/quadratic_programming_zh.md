# 二次规划

二次规划（QP）具有二次目标函数和线性约束：

$$
\min_x \frac12x^\top Qx + c^\top x
$$

受限于

$$
A_{\text{ub}}x\le b_{\text{ub}},
\qquad A_{\text{eq}}x = b_{\text{eq}}.
$$

当

$$
Q\succeq0,
$$

该问题是凸的。

## 无约束解

如果 $Q$ 是正定的且没有约束，平稳性给出

$$
Qx^* + c = 0,
$$

因此

$$
Qx^* = -c.
$$

同样，直接求解该系统，而不是构造 $Q^{-1}$。

## 投影梯度思想

对于凸可行集 $C$，投影梯度下降使用

$$
y_k = x_k - \alpha\nabla f(x_k),
$$

然后

$$
x_{k+1} = \Pi_C(y_k),
$$

其中 $\Pi_C$ 表示欧几里得投影。

![Gradient step followed by projection to a feasible set](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/quadratic_programming_projection.svg)

仓库实现采用了这种思想的简化形式：它执行一个梯度步，然后通过 Dykstra 算法投影到可行多面体上，该算法遍历各个半空间和仿射等式集，并在扫描之间传递校正项。

## 二次函数的步长

对于

$$
f(x) = \frac12x^\top Qx + c^\top x,
$$

梯度为

$$
\nabla f(x) = Qx + c.
$$

如果 $Q\succeq0$，最大特征值 $\lambda_{\max}(Q)=\|Q\|_2$ 是梯度的李普希茨常数 $L$。仓库使用的固定步长因此为

$$
\alpha = \frac{1}{L} = \frac{1}{\|Q\|_2}.
$$

如果 $Q$ 数值上为零，则该问题是一个线性规划：梯度是常数，没有曲率可供推导步长，并且例程会报错而不是猜测步长。在这种情况下，请使用线性规划求解器。

### 投影必须计算的内容

到非空闭凸集的欧几里得投影求解

$$
\Pi_C(y)=\underset{z\in C}{\operatorname{arg\,min}}\ \tfrac12\|z-y\|_2^2.
$$

对于半空间 $a^\top z\leq b$（其中 $a\ne0$），

$$
\Pi(y)=y-\frac{\max(a^\top y-b,0)}{\|a\|_2^2}a.
$$

对于满行秩等式 $Az=b$，它是 $y-A^\top(AA^\top)^{-1}(Ay-b)$，通过线性求解进行计算。单次遍历多个此类集合可能会违反之前访问过的集合。Dykstra 算法为每个集合存储校正项：在投影之前添加存储的校正项，然后用校正后的输入与其投影之间的差值替换它。在标准非空闭凸假设下，重复扫描收敛到交集上的投影。

投影梯度的停止量是

$$
G_\alpha(x)=\frac{x-\Pi_C(x-\alpha\nabla f(x))}{\alpha}.
$$

对于可行 $x$ 和凸可微 $f$，$G_\alpha(x)=0$ 等价于一阶约束最优性条件。普通梯度不必消失。例如，在 $x\geq1$ 上最小化 $x^2/2$ 给出 $x^*=1$：其梯度为 $1$，而投影将试探点 $0$ 送回 $1$，且梯度映射为零。

仅正半定性并不能保证有限的最优解。具有 $Q=\operatorname{diag}(1,0)$ 和 $c=(0,-1)^\top$ 的无约束问题在其第二个坐标上是凸的但无下界。可行性和 attainment 仍然是独立的要求。

### 为什么投影步可以减少目标函数

设当前点 $x$ 可行，令 $z=\Pi_C(x-\alpha g)$，并写 $d=z-x$。投影的一阶条件，针对可行点 $x$ 测试，给出 $g^\top d\leq-\|d\|_2^2/\alpha$。将此与 $L$-平滑上界结合得到

$$
f(z)\leq f(x)-\left(\frac1\alpha-\frac L2\right)\|d\|_2^2.
$$

因此，精确的投影步在 $0<\alpha<2/L$ 时减少非平稳目标函数。常见的选择 $\alpha=1/L$ 留下了方便的余量。这一论点解释了为什么投影必须是最近的可行点，而不是任意的可行修复，以及为什么不准确的内部投影会影响外部收敛。它不要求普通梯度在边界解处消失。

## 等式约束 QP 的 KKT 系统

对于

$$
Ax = b,
$$

最优性方程为

$$
Qx + c + A^\top\nu = 0,
$$

$$
Ax = b.
$$

它们共同构成

$$
\begin{bmatrix}
Q&A^\top\\
A&0
\end{bmatrix}
\begin{bmatrix}
x\\
\nu
\end{bmatrix} =
\begin{bmatrix}
-c\\
b
\end{bmatrix}.
$$

这种线性代数观点是主动集法和内点法的核心。

## 数值注意事项

单次循环投影扫描甚至可能无法离开可行点。重复未校正的投影可能会到达一个不是最近的可行点，这会改变投影梯度算法。Dykstra 的校正项在极限下恢复真正的投影；有限内部求解必须监控可行性和投影精度。当约束集以不利的角度相交时，收敛可能很慢，因此大型或困难的 QP 通常从专用求解器中受益。等式 KKT 系统是非奇异的，例如，当 $A$ 具有满行秩且 $Q$ 在 $\ker A$ 上是正定的；$Q$ 的完全正定性是充分的，但比必要的更强。