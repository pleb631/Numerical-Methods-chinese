# 本文件为 gaussian_interpolation.py 的中文注释版本（代码逻辑与英文版完全一致）
# gaussian_interpolation.py
#
# 高斯前向/后向中心差分多项式插值
# 用于等间距节点。这并非高斯径向基函数（RBF）插值
# 该方法位于本目录下的 gaussian_rbf.py 中，是
# notes/6_regression/gaussian_interpolation.md 的主题。
import numpy as np
from math import factorial


def _build_difference_table(y_data: np.ndarray) -> np.ndarray:
    """构建等间距数据的向前差分表。

返回一个 (n+1) x (n+1) 数组，其中条目 [i][k] 存储 Δ^k y_i。
"""
    n = len(y_data) - 1
    table = np.zeros((n + 1, n + 1))
    table[:, 0] = y_data
    for k in range(1, n + 1):
        for i in range(n + 1 - k):
            table[i][k] = table[i + 1][k - 1] - table[i][k - 1]
    return table


def _gauss_forward_coeff(t: float, k: int) -> float:
    """k 阶高斯向前公式的系数。

(t - j) 的乘积，其中 j = -floor((k-1)/2) .. floor(k/2)，除以 k!。
"""
    lo = -((k - 1) // 2)
    hi = k // 2
    coeff = 1.0
    for j in range(lo, hi + 1):
        coeff *= t - j
    return coeff / factorial(k)


def _gauss_backward_coeff(t: float, k: int) -> float:
    """k 阶高斯向后公式的系数。

(t - j) 的乘积，其中 j = -floor(k/2) .. floor((k-1)/2)，除以 k!。
"""
    lo = -(k // 2)
    hi = (k - 1) // 2
    coeff = 1.0
    for j in range(lo, hi + 1):
        coeff *= t - j
    return coeff / factorial(k)


def _gauss_forward(diff_table: np.ndarray, m: int, t: float, n: int) -> float:
    """评估高斯向前中心差分公式。

差分表中的之字形路径：
    Δy_m, Δ²y_{m-1}, Δ³y_{m-1}, Δ⁴y_{m-2}, ...
k 阶的基础索引：m - floor(k/2)。
"""
    result = diff_table[m][0]
    for k in range(1, n + 1):
        base = m - k // 2
        if base < 0 or base + k > n:
            break
        result += _gauss_forward_coeff(t, k) * diff_table[base][k]
    return result


def _gauss_backward(diff_table: np.ndarray, m: int, t: float, n: int) -> float:
    """评估高斯向后中心差分公式。

差分表中的之字形路径：
    Δy_{m-1}, Δ²y_{m-1}, Δ³y_{m-2}, Δ⁴y_{m-2}, ...
k 阶的基础索引：m - ceil(k/2)。
"""
    result = diff_table[m][0]
    for k in range(1, n + 1):
        base = m - (k + 1) // 2
        if base < 0 or base + k > n:
            break
        result += _gauss_backward_coeff(t, k) * diff_table[base][k]
    return result


def gaussian_interpolation(
    x_data: np.ndarray, y_data: np.ndarray, point: float
) -> float:
    """高斯中心差分插值，用于等间距数据。

当目标点位于中心节点右侧（t >= 0）时使用高斯向前公式，位于左侧（t < 0）时使用向后公式。两者均恢复通过所有数据点的唯一多项式插值。

参数
-----
x_data : 等间距横坐标数组
y_data : 对应的纵坐标数组
point  : 进行插值的横坐标

返回
-----
*point* 处的插值结果。



"""
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("X and Y vectors must have equal number of elements.")
    if x_data.shape[0] < 2:
        raise ValueError("At least two points are required for interpolation.")
    if len(np.unique(x_data)) != x_data.shape[0]:
        raise ValueError("X data must contain unique values.")
    if point < np.min(x_data) or point > np.max(x_data):
        raise ValueError("Point is out of bounds.")

    n = len(x_data) - 1
    h = float(x_data[1] - x_data[0])

    # 缩放容差：固定的绝对容差会接受
    # 节点极小（例如 ~1e-9）时会出现任意不均匀的间距。
    diffs = np.diff(x_data.astype(float))
    tol = 1e-9 * abs(h) + 8 * np.finfo(float).eps * np.max(np.abs(x_data))
    if not np.all(np.abs(diffs - h) <= tol):
        raise ValueError("X data must be equally spaced for Gaussian interpolation.")

    diff_table = _build_difference_table(y_data.astype(float))

    # 向前公式在 m = n//2 时达到满阶。
    # 向后公式在 n 为奇数时需要 m = (n+1)//2。
    m_fwd = n // 2
    t_fwd = (float(point) - float(x_data[m_fwd])) / h

    if t_fwd >= 0:
        return _gauss_forward(diff_table, m_fwd, t_fwd, n)
    else:
        m_bwd = (n + 1) // 2
        t_bwd = (float(point) - float(x_data[m_bwd])) / h
        return _gauss_backward(diff_table, m_bwd, t_bwd, n)
