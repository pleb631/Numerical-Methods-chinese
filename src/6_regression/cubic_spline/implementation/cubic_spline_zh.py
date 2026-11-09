# 本文件为 cubic_spline.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import bisect
from typing import Callable


def cubic_spline(x_data: np.ndarray, y_data: np.ndarray) -> Callable[[float], float]:
    """返回通过给定数据的自然三次样条插值函数。

构建唯一的分段三次函数 S(x)，使其经过每个 (x_i, y_i) 点，在每个内部节点处具有连续的一阶和二阶导数，并满足自然边界条件 S''(x_0) = S''(x_n) = 0。

二阶导数 M_i 通过求解三对角方程组求得

    h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1}
        = 6 [(y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1}]

随后使用托马斯算法，之后每个片段被评估为

    S_i(x) = M_i/(6h_i)(x_{i+1}-x)³ + M_{i+1}/(6h_i)(x-x_i)³
           + (y_i - M_i h_i²/6)(x_{i+1}-x)/h_i
           + (y_{i+1} - M_{i+1} h_i²/6)(x-x_i)/h_i

参数
-----
x_data : 严格递增的 x 坐标的一维数组（≥ 3 个点）。
y_data : 对应的 y 值的一维数组。

返回
-----
可调用函数，用于在 [x_data[0], x_data[-1]] 范围内的任意 x 处评估样条函数。



"""
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("X and Y vectors must have equal number of elements.")
    if x_data.shape[0] < 3:
        raise ValueError("At least three data points are required.")
    if len(np.unique(x_data)) != x_data.shape[0]:
        raise ValueError("X data must contain unique values.")
    if not np.all(np.diff(x_data) > 0):
        raise ValueError("X data must be strictly increasing.")

    x_data = x_data.astype(float)
    y_data = y_data.astype(float)
    n = x_data.shape[0]
    h = np.diff(x_data)

    # 三对角系统右侧的向量（系数为 6，而非 3）
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = 6 * (
            (y_data[i + 1] - y_data[i]) / h[i]
            - (y_data[i] - y_data[i - 1]) / h[i - 1]
        )

    # 托马斯算法——前向消元
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    for i in range(1, n - 1):
        l[i] = 2 * (x_data[i + 1] - x_data[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    # 后向代入 → 二阶导数 M_i
    M = np.zeros(n)
    for j in range(n - 2, 0, -1):
        M[j] = z[j] - mu[j] * M[j + 1]

    def spline(X: float) -> float:
        if X < x_data[0] or X > x_data[-1]:
            raise ValueError(
                f"Query point {X} outside data range [{x_data[0]}, {x_data[-1]}]."
            )
        i = bisect.bisect_right(x_data, X) - 1
        i = min(max(i, 0), n - 2)
        dx = X - x_data[i]
        return (
            M[i] / (6 * h[i]) * (x_data[i + 1] - X) ** 3
            + M[i + 1] / (6 * h[i]) * dx**3
            + (y_data[i] - M[i] * h[i] ** 2 / 6) * (x_data[i + 1] - X) / h[i]
            + (y_data[i + 1] - M[i + 1] * h[i] ** 2 / 6) * dx / h[i]
        )

    return spline
