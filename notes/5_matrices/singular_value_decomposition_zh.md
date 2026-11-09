# 奇异值分解 (SVD)

奇异值分解 (SVD) 将一个矩阵表示为围绕矩形对角缩放的正交或酉坐标变换。它存在于每一个实数或复数矩阵中，包括矩形矩阵和秩亏矩阵，而可对角化的特征值分解则需要一个可对角化的方阵。奇异值量化了变换沿配对的输入和输出方向的作用强度。这使得 SVD 在降维、噪声过滤、图像压缩和最小二乘问题中非常有用，特别是在秩或条件数不确定的情况下。该分解将这些问题分离为显式量：正奇异值描述了矩阵的非零作用，而零奇异值标识了将其映射为零的方向。

![Orthogonal factors and singular values of a matrix](https://github.com/user-attachments/assets/1479f467-1d30-4f80-84e2-4c996d4b09fc)

## 数学公式

给定任意 $m \times n$ 矩阵 $A$，SVD 将 $A$ 分解为：

$$
A = U \Sigma V^{T}
$$

其中：

I. **$U$** 是一个 $m \times m$ 正交矩阵（即 $U^{T}U = I$）。其列称为 $A$ 的左奇异向量。

II. **$\Sigma$** (Sigma) 是一个 $m \times n$ 对角矩阵，对角线上为非负实数。这些数字 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_p \geq 0$（其中 $p = \min(m,n)$）称为 $A$ 的奇异值。

III. **$V$** 是一个 $n \times n$ 正交矩阵（即 $V^{T}V = I$）。其列称为 $A$ 的右奇异向量。

综合起来：

$$
A = U \begin{bmatrix}
\sigma_1 & 0 & \cdots & 0 \\
0 & \sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_p \\
\vdots & \vdots & & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix} V^{T}
$$

这里，$\Sigma$ 是矩形对角矩阵：$\Sigma_{ii}=\sigma_i$ 对于 $i\leq p$，其余条目为零。显示的零行填充是用于高矩阵的示意图；宽矩阵则在 $p\times p$ 对角块之外具有零列。其完整形状始终为 $m\times n$。$A$ 的秩是正奇异值的数量。

## 推导

SVD 与矩阵 $A^{T}A$ 和 $AA^{T}$ 的特征值分解密切相关。$A^{T}A$ 和 $AA^{T}$ 都是对称、半正定矩阵，因此具有非负实特征值。

**步骤 1。** 计算 $A^{T}A$ 的特征值。这些特征值 $\lambda_i$ 是非负的。

II. $A$ 的奇异值定义为 $\sigma_i = \sqrt{\lambda_i}$。

III. $A^{T}A$ 的特征向量构成 $V$ 的列，$AA^{T}$ 的特征向量构成 $U$ 的列（符号选择一致，例如 $u_i = A v_i / \sigma_i$）。

由于每个实矩阵 $A$ 都会产生一个非负、对称矩阵 $A^{T}A$，并且此类矩阵总是可对角化的，因此 SVD 保证对任何矩阵 $A$ 都存在。

### 完成推导并处理秩亏

选择 $A^\top A$ 的正交特征向量 $v_i$。对于每个 $\sigma_i>0$，定义 $u_i=Av_i/\sigma_i$。则

$$
u_i^\top u_j=\frac{v_i^\top A^\top Av_j}{\sigma_i\sigma_j}=\frac{\sigma_j^2v_i^\top v_j}{\sigma_i\sigma_j}=\delta_{ij}.
$$

因此，构造出的左奇异向量是正交的。如果 $\sigma_i=0$，则 $\|Av_i\|_2^2=v_i^\top A^\top Av_i=0$，所以 $Av_i=0$；除以 $\sigma_i$ 是不必要且无效的。完成可用的正交列以形成完整因子。对于秩 $r$，相同的分解可以写为不含未使用列的形式：

$$
A=U_r\Sigma_rV_r^\top=\sum_{i=1}^r\sigma_i u_iv_i^\top,
$$

其中 $U_r$ 是 $m\times r$，$\Sigma_r$ 是 $r\times r$，$V_r$ 是 $n\times r$。对于复数矩阵，用共轭转置代替转置。

下面的 Gram 矩阵构造解释了存在性，并且适用于手工计算，但不是首选的通用数值算法：形成 $A^\top A$ 会平方非零奇异值条件比，并可能丢失关于小奇异值的信息。生产级 SVD 方法通过对 $A$ 进行正交变换来工作，通常将其简化为双对角形式。

伪逆是 $A^+=V_r\Sigma_r^{-1}U_r^\top$，并给出最小范数最小二乘解 $A^+b$。仅保留最大的 $k$ 个奇异值可得 $A_k=\sum_{i=1}^k\sigma_i u_iv_i^\top$，当 $k<r$ 时，$\|A-A_k\|_2=\sigma_{k+1}$。这量化了压缩误差，而不是假设每个小奇异值都是噪声。

## 算法步骤

**步骤 1：形成 $A^{T}A$。**

计算 $n \times n$ 矩阵 $A^{T}A$。

**步骤 2：计算 $A^{T}A$ 的特征值和特征向量。**

求解 $(A^{T}A) v = \lambda v$ 以找到所有特征值 $\lambda_i \ge 0$ 和相应的特征向量 $v_i$。

**步骤 3：获取奇异值。**

按降序排列特征值并取 $\sigma_i = \sqrt{\lambda_i}$。它们构成 $\Sigma$ 的对角条目。

**步骤 4：形成 $V$。**

$A^{T}A$ 的特征向量 $v_i$ 排列成列以形成矩阵 $V$。

**步骤 5：形成 $U$。**

类似地，找到 $AA^{T}$ 的特征向量，对每个正奇异值使用 $u_i=Av_i/\sigma_i$。将这些列完成为正交基以获得完整的 $U$。矩形或秩亏的 $\Sigma$ 没有普通矩阵逆。

**步骤 6：组装 SVD。**

计算出 $U, \Sigma, V$ 后，$A = U \Sigma V^{T}$。

## 示例

考虑 $2 \times 2$ 矩阵：

$$
A = \begin{bmatrix}3 & 4 \\ 2 & 1\end{bmatrix}.
$$

**步骤 1。** 计算 $A^{T}A$：

$$
A^{T}A = \begin{bmatrix} 3 & 2 \\ 4 & 1 \end{bmatrix}\begin{bmatrix}3 & 4 \\ 2 & 1\end{bmatrix} = \begin{bmatrix}13 & 14 \\ 14 & 17\end{bmatrix}.
$$

**步骤 2。** 找出 $A^{T}A$ 的特征值：

求解 $\det(A^{T}A - \lambda I) = 0$：

$$
(13 - \lambda)(17 - \lambda) - 14^2 = \lambda^2 - 30\lambda + 25 = 0.
$$

求解得到 $\lambda_1 = 15 + 10\sqrt{2} \approx 29.1421$ 和 $\lambda_2 = 15 - 10\sqrt{2} \approx 0.8579$。

III. 奇异值：

$$
\sigma_1 = \sqrt{\lambda_1} \approx 5.3983, \quad \sigma_2 = \sqrt{\lambda_2} \approx 0.9262.
$$

(检查：$\sigma_1 \sigma_2 = |\det A| = 5$。)

IV. $A^{T}A$ 的特征向量（归一化）：

$$
v_1 \approx \begin{bmatrix}0.6552 \\ 0.7555\end{bmatrix}, \quad v_2 \approx \begin{bmatrix}-0.7555 \\ 0.6552\end{bmatrix}.
$$

因此：

$$
V \approx \begin{bmatrix}0.6552 & -0.7555 \\ 0.7555 & 0.6552\end{bmatrix}.
$$

V. 为了找到 $U$，对非零奇异值使用 $u_i = \frac{1}{\sigma_i} A v_i$：

$$
u_1 \approx \begin{bmatrix}0.9239 \\ 0.3827\end{bmatrix}, \quad u_2 \approx \begin{bmatrix}0.3827 \\ -0.9239\end{bmatrix},
$$

所以

$$
U \approx \begin{bmatrix}0.9239 & 0.3827 \\ 0.3827 & -0.9239\end{bmatrix}.
$$

**步骤 6。** 组装 SVD：

$$
\Sigma \approx \begin{bmatrix}5.3983 & 0 \\ 0 & 0.9262\end{bmatrix}
$$

$$
A = U \Sigma V^{T}.
$$

## 优势

I. **通用性**：

SVD 存在于任何 $m \times n$ 矩阵 $A$ 中，无论其秩如何，使其比 EVD 更广泛适用。

II. **噪声抑制和压缩**：

通过截断小奇异值，可以获得接近原始但更简单的 $A$ 的低秩近似，这在数据压缩和去噪中很有用。

III. **数值稳定性**：

SVD 在数值上是稳定的，并广泛用于稳健的数值方法中，例如伪逆计算和求解最小二乘问题。

## 局限性

I. **计算复杂度**：

对于大型方阵，计算 SVD 通常比特征值分解计算成本更高。存在高效算法，但对于非常大的数据集，成本仍然可能很高。

II. **因子的解释**：

虽然分解产生正交因子和非负奇异值，但解释这些分量的物理或特定应用含义可能需要额外的洞察力。

III. **原始 $A$ 的直接特征值信息**：

奇异值与 $A^{T}A$（或 $AA^{T}$）的特征值相关，而不是直接与 $A$ 本身的特征值相关。对于正规矩阵（包括实对称矩阵），奇异值等于其特征值的模。它们仍然不能确定这些特征值的符号或复数相位。例如，$\operatorname{diag}(2,1)$ 和 $\operatorname{diag}(-2,1)$ 具有相同的奇异值但不同的特征值。即使在這種特殊情况下，特征值问题和奇异值问题也保留了不同的信息。