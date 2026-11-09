# 高斯径向基函数插值

高斯径向基函数 (RBF) 插值利用以数据站点为中心的高斯函数构建平滑插值函数。

对于一维节点

$$
x_0,\ldots,x_n,
$$

插值函数具有如下形式

$$
s(x) = \sum_{j=0}^{n} \lambda_j e^{-\varepsilon^2(x-x_j)^2}
$$

其中系数 $\lambda_j$ 的选择使得

$$
s(x_i) = y_i
$$

对每一个数据点成立。

与分段样条不同，该表示法是全局的：每一个基函数都可以对每一次查询做出贡献。

![以数据站点为中心的高斯径向基函数](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/6_regression/resources/plots/gaussian_rbf_basis_functions.svg)

该图展示了**未加权的基函数**，其中心为 $x_j=0,1,2,3$，共享形状参数 $\varepsilon=1.2$。每条曲线在其自身中心处恰好达到 $1$，关于该中心对称，并随远离中心而趋于零。高度为 $1$ 的点标记峰值；高度为 $0$ 的点标记数据站点，而非观测值 $y_j$。

单个基函数并不直接插值观测值。插值函数 $s(x)$ 是它们的加权和，且系数 $\lambda_j$ 不必等于数据值或为正数。基函数也不必求和为一。

## 高斯核

高斯径向基函数为

$$
\phi(r) = e^{-(\varepsilon r)^2},
$$

其形状参数为

$$
\varepsilon>0.
$$

距离为

$$
r = |x - x_j|.
$$

较大的 $\varepsilon$ 产生较窄的高斯函数。较小的 $\varepsilon$ 产生更宽、更平坦的高斯函数。

另一种常见的参数化形式为

$$
\phi(r) = e^{-r^2/(2\sigma^2)}.
$$

当满足

$$
\varepsilon = \frac{1}{\sqrt{2}\sigma}
$$

时，这两种形式等价。

阅读代码或论文时，务必检查使用的是哪种约定。

## 插值系统

应用条件

$$
s(x_i) = y_i
$$

可得

$$
\sum_{j=0}^{n} \lambda_j e^{-\varepsilon^2(x_i-x_j)^2} = y_i.
$$

定义矩阵

$$
A_{ij} = e^{-\varepsilon^2(x_i-x_j)^2}.
$$

则系数满足

$$
A\boldsymbol{\lambda} = \mathbf y.
$$

对于互异的数据站点和 $\varepsilon>0$，高斯核是严格正定的。在精确算术下，该矩阵因此是对称正定的，且插值问题有唯一解。

## 不要显式构造逆矩阵

数学恒等式

$$
\boldsymbol{\lambda} = A^{-1}\mathbf y
$$

在符号上很有用，但数值代码应直接求解线性方程组。

对于稠密对称正定矩阵，Cholesky 分解是一种自然的直接方法。对于小规模示例，通用稠密求解器也适用。

显式计算 $A^{-1}$ 通常成本更高，存储的信息多于所需，且可能放大数值误差。

## 计算示例

使用

$$
(0,0),
\qquad(1,0.5),
\qquad(2,0)
$$

且

$$
\varepsilon = 1.
$$

则

$$
A =
\begin{bmatrix}
1 & e^{-1} & e^{-4}\\
e^{-1} & 1 & e^{-1}\\
e^{-4} & e^{-1} & 1
\end{bmatrix}.
$$

数值上，

$$
A
\approx
\begin{bmatrix}
1 & 0.367879 & 0.018316\\
0.367879 & 1 & 0.367879\\
0.018316 & 0.367879 & 1
\end{bmatrix}.
$$

求解

$$
A\boldsymbol{\lambda} =
\begin{bmatrix}
0\\0.5\\0
\end{bmatrix}
$$

得到近似解

$$
\lambda_0 = -0.246025,
$$

$$
\lambda_1 = 0.681015,
$$

$$
\lambda_2 = -0.246025.
$$

因此，

$$
s(x) = -0.246025e^{-x^2} + 0.681015e^{-(x-1)^2} - 0.246025e^{-(x-2)^2}.
$$

在

$$
x = 0.5,
$$

结果约为

$$
\boxed{s(0.5)\approx0.31284}.
$$

在精确算术下，插值函数仍精确通过所有三个给定的数据值。

## 形状参数为何重要

形状参数既影响插值函数的视觉形状，也影响线性系统的条件数。

![高斯形状参数的影响](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/6_regression/resources/plots/gaussian_rbf_shape_parameter.svg)

当 $\varepsilon$ 较大时，每个基函数都很窄。若站点分离良好，插值矩阵将更接近单位矩阵。

当 $\varepsilon$ 较小时，基函数变得宽泛且彼此日益相似。即使矩阵在精确算术下保持非奇异，其条件数也可能变得严重恶化。

这构成了经典的 RBF 权衡：

- 更平坦的基函数可以产生优异的逼近性质；
- 但相应的系数求解可能变得数值困难。

$\varepsilon$ 的取值不应仅凭视觉偏好来选择。

### 读取矩阵与缩放形状参数

在对称三点示例中，令 $a=e^{-1}$ 和 $b=e^{-4}$。由对称性可知 $\lambda_0=\lambda_2=u$ 且 $\lambda_1=v$，因此矩阵方程简化为

$$
(1+b)u+av=0,\qquad 2au+v=\tfrac12.
$$

由此得 $v=\tfrac12/[1-2a^2/(1+b)]$ 且 $u=-av/(1+b)$。这补充了报告系数背后的缺失代数推导，并解释了为何需要负的端点系数来重现零端点值。

对于固定的互异站点，$\varepsilon\to0$ 会使每个矩阵元素趋于 $1$。极限全一矩阵的秩为一，这解释了平坦高斯函数条件数丧失的原因。当 $\varepsilon\to\infty$ 时，非对角元素趋于零且 $A$ 趋于单位矩阵，但非常窄的核函数仍可能导致站点之间产生较差的数值结果，尽管矩阵条件数有利。

有意义的量是无量纲距离 $\varepsilon|x_i-x_j|$。若坐标缩放为 $\widetilde x=cx$（其中 $c>0$），则选择 $\widetilde\varepsilon=\varepsilon/c$ 以保持相同的核函数。在多维情况下，坐标缩放不仅决定距离几何，也影响数值条件数。

## 条件数

条件数

$$
\kappa(A)
$$

衡量系数解对扰动和舍入的敏感程度。

较大的条件数意味着矩阵 $A$ 或向量 $\mathbf y$ 中的微小数值误差可能导致计算出的系数产生大得多的相对变化。

一个有用的工作流程是同时检查：

- 节点处的插值精度；
- 核矩阵的条件数。

即使最终插值结果看起来合理，大系数伴随大量抵消也可能是一个警告信号。

## 导数

高斯插值函数具有无限阶可导性。

一阶导数为

$$
s'(x) = \sum_{j=0}^{n} - 2\varepsilon^2(x - x_j) \lambda_j e^{-\varepsilon^2(x-x_j)^2}.
$$

二阶导数为

$$
s''(x) = \sum_{j=0}^{n} \lambda_j
\left[4\varepsilon^4(x - x_j)^2 - 2\varepsilon^2 \right] e^{-\varepsilon^2(x-x_j)^2}.
$$

由于不存在分段连接，无需显式建立导数匹配方程。

## 算法

I. 验证所有数据站点互异；

II. 选择 $\varepsilon>0$；

**步骤 3.** 构造稠密矩阵

$$
A_{ij} = e^{-\varepsilon^2(x_i-x_j)^2};
$$

**步骤 4.** 求解

$$
A\boldsymbol{\lambda} = \mathbf y;
$$

**步骤 5.** 利用下式评估新查询：

$$
s(x) = \sum_j \lambda_j e^{-\varepsilon^2(x-x_j)^2}.
$$

## 复杂度

对于 $N=n+1$ 个站点，直接的稠密实现大约需要：

- $O(N^2)$ 内存用于存储矩阵；
- $O(N^3)$ 工作量用于直接分解；
- 在已知系数后，单次标量查询的工作量为 $O(N)$。

这些成本对于大型数据集变得重要。此时可能需要专门的近似方法、局部 RBF 方法、迭代求解器或快速核技术。

## 插值不同于平滑

上述系统强制精确插值。如果数据存在噪声，重现每一个观测值可能并不理想。

正则化 RBF 拟合修改了该问题，例如通过求解如下形式的系统

$$
(A + \alpha I)\boldsymbol{\lambda} = \mathbf y,
$$

其中 $\alpha>0$。

这不再是精确插值，但它可以用微小的数据不匹配换取更好的稳定性和平滑性。

站点处的数据残差使这一变化显式化：$\mathbf r=\mathbf y-A\boldsymbol\lambda=\alpha\boldsymbol\lambda$。不匹配由正则化系统控制，而非强制为零。这种核正则化也应与直接在 $\|A\boldsymbol\lambda-\mathbf y\|_2^2$ 上施加欧几里得系数惩罚区分开来：后者给出正规方程 $(A^\top A+\alpha I)\boldsymbol\lambda=A^\top\mathbf y$。陈述系统或目标函数可防止“正则化”一词掩盖模型的变化。

## 高维情形

高斯 RBF 插值可直接扩展至点

$$
\mathbf x_j\in\mathbb R^d.
$$

将一维距离替换为欧几里得范数：

$$
r = \|\mathbf x - \mathbf x_j\|_2.
$$

则

$$
s(\mathbf x) = \sum_j \lambda_j e^{-\varepsilon^2\|\mathbf x-\mathbf x_j\|_2^2}.
$$

这种与维度无关的形式是 RBF 插值对散乱数据有用的原因之一。

## 本仓库中的代码

文件夹 `src/6_regression/gaussian_interpolation` 包含两种不同的方法：

- `implementation/gaussian_rbf.py` (`GaussianRBFInterpolator`) 实现了本说明中描述的高斯径向基函数插值，而 `examples/plot_gaussian_rbf.py` 对其绘图；
- `implementation/gaussian_interpolation.py` 连同 `examples/example.ipynb` 和 `examples/plot.py`，实现了等距节点的**高斯中心差分公式**。那是一个多项式插值函数（它复现与拉格朗日或牛顿插值相同的多项式），与高斯核无关。

共享的文件夹名称是历史遗留；两者不应混淆。