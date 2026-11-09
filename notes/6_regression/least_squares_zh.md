# 最小二乘法

最小二乘法通过最小化残差平方和来拟合模型。

对于线性模型

$$
\mathbf y\approx X\boldsymbol{\beta},
$$

残差向量为

$$
\mathbf r = \mathbf y - X\boldsymbol{\beta}.
$$

普通最小二乘法选择使以下目标函数最小化的系数：

$$
\mathrm{RSS}(\boldsymbol{\beta}) = \|\mathbf y -
X\boldsymbol{\beta}\|_2^2 = \sum_{i=1}^{N}r_i^2.
$$

等价地，

$$
\hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta}}{\mathrm{arg\,min}} \;
\mathrm{RSS}(\boldsymbol{\beta}).
$$

与插值不同，最小二乘法通常不要求拟合模型穿过每一个数据点。

![Least-squares line and its residuals](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/6_regression/resources/plots/least_squares_residuals.svg)

用 $N$ 表示观测值数量，用 $p$ 表示拟合系数数量，因此 $X\in\mathbb R^{N\times p}$，$\boldsymbol\beta\in\mathbb R^p$，且 $\mathbf y,\mathbf r\in\mathbb R^N$。行对应观测值；列对应模型特征，如果使用了截距项则包括截距。下面的正规方程包含 $p$ 个系数空间方程，即使残差是在更多观测值上测量的。

## 设计矩阵形式

假设直线模型为

$$
y\approx\beta_0 + \beta_1x.
$$

对于测量值 $(x_i,y_i)$，设计矩阵为

$$
X =
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_N
\end{bmatrix},
$$

系数向量为

$$
\boldsymbol{\beta} =
\begin{bmatrix}
\beta_0\\
\beta_1
\end{bmatrix},
$$

观测值为

$$
\mathbf y =
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_N
\end{bmatrix}.
$$

该框架同样适用于多项式回归以及任何对其未知系数呈线性的模型。

## 推导正规方程

展开目标函数：

$$
\begin{aligned}
  \mathrm{RSS}(\boldsymbol{\beta}) = (\mathbf y - X\boldsymbol{\beta})^\top (\mathbf y - X\boldsymbol{\beta}) \\
  = \mathbf y^\top\mathbf y - 2\boldsymbol{\beta}^\top X^\top\mathbf y + \boldsymbol{\beta}^\top X^\top X\boldsymbol{\beta}.
\end{aligned}
$$

对 $\boldsymbol{\beta}$ 求导：

$$
\nabla \mathrm{RSS} = -2X^\top\mathbf y + 2X^\top X\boldsymbol{\beta}.
$$

在最小值点，

$$
\nabla\mathrm{RSS} = 0,
$$

由此得到**正规方程**

$$
X^\top X\hat{\boldsymbol{\beta}} = X^\top\mathbf y.
$$

如果 $X$ 的列线性无关，则 $X^\top X$ 是正定的，解是唯一的。

## 几何解释

拟合向量

$$
\hat{\mathbf y} = X\hat{\boldsymbol{\beta}}
$$

属于 $X$ 的列空间。

正规方程可以重写为

$$
X^\top (\mathbf y - X\hat{\boldsymbol{\beta}}) = 0
$$

因此，

$$
X^\top\mathbf r = 0.
$$

残差向量与 $X$ 的每一列正交。从几何上看，最小二乘拟合是 $\mathbf y$ 在设计矩阵列空间上的正交投影。

这种解释比熟悉的最佳拟合直线图景更为通用。

### 为什么驻点是最小值

设 $\widehat{\boldsymbol\beta}$ 满足正规方程，并将任意其他系数向量写为 $\widehat{\boldsymbol\beta}+d$。由于残差与 $X$ 的列正交，交叉项消失：

$$
\|\mathbf y-X(\widehat{\boldsymbol\beta}+d)\|_2^2=\|\mathbf r\|_2^2+\|Xd\|_2^2.
$$

第二项非负，证明了全局最优性。如果 $X$ 具有满列秩，则仅当 $d=0$ 时该项为零，证明了唯一性。如果 $X$ 秩亏，非零零空间向量会给出具有完全相同拟合值的其他系数最小化解。因此，即使系数向量不唯一，预测向量也是唯一的。

对于具有共同方差 $\sigma^2$ 的独立零均值误差以及固定的满秩 $X$，$\operatorname{Cov}(\widehat{\boldsymbol\beta})=\sigma^2(X^\top X)^{-1}$。这种统计不确定性与数值舍入误差是分开的。在 $p$ 个拟合系数且 $N>p$ 的情况下，$\mathrm{RSS}/(N-p)$ 在这些假设下估计 $\sigma^2$；训练均方误差 $\mathrm{RSS}/N$ 服务于不同的目的。

## 计算示例

拟合直线到

$$
(0,1),
\qquad (1,2),
\qquad (2,2),
\qquad (3,4).
$$

模型为

$$
y\approx\beta_0 + \beta_1x.
$$

设计矩阵和响应向量为

$$
X =
\begin{bmatrix}
1&0\\
1&1\\
1&2\\
1&3
\end{bmatrix},
\qquad
\mathbf y =
\begin{bmatrix}
1\\2\\2\\4
\end{bmatrix}.
$$

正规方程为

$$
\begin{bmatrix}
4&6\\
6&14
\end{bmatrix}
\begin{bmatrix}
\beta_0\\
\beta_1
\end{bmatrix} =
\begin{bmatrix}
9\\18
\end{bmatrix}.
$$

求解得到

$$
\hat\beta_0 = 0.9,
\qquad \hat\beta_1 = 0.9.
$$

拟合直线为

$$
\boxed{\hat y=0.9+0.9x}.
$$

预测值为

$$
0.9,\quad1.8,\quad2.7,\quad3.6,
$$

残差为

$$
0.1,\quad0.2,\quad - 0.7,\quad0.4.
$$

它们的总和为零，因为模型包含截距项，并且它们与预测变量列正交，这是正规方程所要求的。

## 目标函数是凸的

对于线性最小二乘问题，

$$
\mathrm{RSS}(\boldsymbol{\beta})
$$

是系数的二次函数。

![Contours of the least-squares objective](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/6_regression/resources/plots/least_squares_objective_contours.svg)

当 $X$ 具有满列秩时，二次碗状函数具有唯一最小值。如果 $X$ 的列线性相关，则存在多个产生相同拟合值的系数向量，除非施加额外标准。

## 为什么直接求解正规方程可能很危险

正规方程在理论上很重要，但显式构造

$$
X^\top X
$$

可能会恶化数值条件数。

在 2-范数下，

$$
\kappa_2(X^\top X) = \kappa_2(X)^2
$$

当 $X$ 具有满列秩时。

因此，中等条件数较差的设计矩阵在平方条件数后可能变得难以准确求解。

对于数值工作，建议：

- 使用 **QR 分解** 进行稳健的一般最小二乘求解；
- 当担心秩亏或严重条件数较差时，使用 **SVD**。

正规方程对于小型、条件数良好的教学示例仍然可以接受。

## QR 分解

如果

$$
X = QR,
$$

其中 $Q$ 具有标准正交列，$R$ 是上三角矩阵，则

$$
\| X\boldsymbol{\beta} - \mathbf y\|_2^2 = \| R\boldsymbol{\beta} - Q^\top\mathbf y\|_2^2 + \|(I - QQ^\top)\mathbf y\|_2^2,
$$

其中最后一项是 $\mathbf y$ 中垂直于列空间的部分，且不依赖于 $\boldsymbol{\beta}$。

系数向量可以通过求解

$$
R\hat{\boldsymbol{\beta}} = Q^\top\mathbf y.
$$

这避免了显式构造 $X^\top X$。

## 秩亏

如果 $X$ 的列线性相关，则系数向量不是唯一确定的。

SVD

$$
X = U\Sigma V^\top
$$

通过 $\Sigma$ 中的奇异值使数值秩显式化。

常见的选择是使用伪逆计算最小范数最小二乘解：

$$
\hat{\boldsymbol{\beta}} = X^+\mathbf y.
$$

在实践中，库程序使用容差来决定哪些奇异值被视为有效零。

## 缩放与中心化

多项式和多元设计矩阵可能缩放不良。

有用的预处理包括：

- 将预测变量以零为中心；
- 将列缩放到可比较的量级；
- 当次数较高时，使用正交多项式基而不是原始幂次。

缩放会改变数值表示，只要随后适当解释或转换系数，就不会改变底层拟合关系。

## 加权最小二乘法

如果观测值具有不同的已知可靠性，使用权重

$$
w_i>0
$$

并最小化

$$
\sum_{i=1}^{N} w_i r_i^2.
$$

对于对角矩阵

$$
W = \mathrm{diag}(w_1,\ldots,w_N),
$$

目标函数变为

$$
(\mathbf y - X\boldsymbol{\beta})^\top W (\mathbf y - X\boldsymbol{\beta}).
$$

相应的正规方程为

$$
X^\top W X\hat{\boldsymbol{\beta}} = X^\top W\mathbf y.
$$

为避免构造加权正规方程，设 $\widetilde X=W^{1/2}X$ 和 $\widetilde{\mathbf y}=W^{1/2}\mathbf y$，然后对 $\|\widetilde{\mathbf y}-\widetilde X\boldsymbol\beta\|_2^2$ 应用 QR 或 SVD。如果独立观测值具有已知方差 $\sigma_i^2$，则逆方差权重 $w_i=1/\sigma_i^2$ 给出通常的加权最小二乘模型。较大的权重代表更高的精度，而不是更大的误差棒。

## 最小二乘法对异常值敏感

平方残差会使大误差具有不成比例的高影响。单个极端异常值可能会显著改变普通最小二乘拟合。

稳健回归方法用平方损失替换或迭代重新加权观测值，以使极端残差的影响减小。

## 实际检查

对于重要的最小二乘模型：

1. 检查设计矩阵的秩；
2. 检查预测变量缩放；
3. 优先使用 QR 或 SVD 而不是手动构建的逆矩阵；
4. 检查残差而不仅仅是报告系数；
5. 检查异常值和杠杆点；
6. 将数值拟合质量与预测有效性分开。

## 关于仓库实现的注释

`src/6_regression/least_squares` 通过 $X^\top X$ 的伪逆求解正规方程，用于教学目的；在实践中，对 $X$ 本身进行 QR 或 SVD 更可取。该函数要求 $m\ge n$ 且具有满列秩，否则抛出 `ValueError`。$X$ 中全为零的行会被接受：它对 $X^\top X$ 或 $X^\top\mathbf y$ 没有贡献，并且仅向残差添加常数 $y_i^2$。