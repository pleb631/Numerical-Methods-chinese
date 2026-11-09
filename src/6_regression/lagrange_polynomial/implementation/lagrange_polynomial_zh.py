# 本文件为 lagrange_polynomial.py 的中文注释版本（代码逻辑与英文版完全一致）
# lagrange_polynomial.py
import numpy as np


def lagrange_polynomial(x_data: np.ndarray, y_data: np.ndarray, point: float) -> float:
    """在 *point* 处评估拉格朗日插值多项式。

给定 (n+1) 个数据点，通过所有这些点的唯一次数 ≤ n 的多项式为：

    L(x) = Σᵢ yᵢ · Πⱼ≠ᵢ (x − xⱼ) / (xᵢ − xⱼ)

参数
------
x_data : 一维不同 x 坐标数组。
y_data : 对应的 y 值一维数组（与 *x_data* 长度相同）。
point  : 评估多项式的 x 值。

返回
----
float – L(point)。

"""
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("X and Y vectors must have equal number of elements.")
    if x_data.shape[0] < 1:
        raise ValueError("X and Y vectors must contain at least one element.")
    if len(np.unique(x_data)) != x_data.shape[0]:
        raise ValueError("X data must contain unique values.")

    n = x_data.shape[0]
    P = 0.0
    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if j != i:
                L_i *= (point - x_data[j]) / (x_data[i] - x_data[j])
        P += y_data[i] * L_i
    return P
