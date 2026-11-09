# 本文件为 thin_plate_spline_interpolation.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from typing import Tuple


def _tps_kernel(r: float) -> float:
    """薄板样条径向基函数：φ(r) = r²ln(r)。当 r = 0 时定义为 0（即 r → 0⁺ 时 r²ln(r) 的极限）。


"""
    if r == 0.0:
        return 0.0
    return r * r * np.log(r)


def thin_plate_spline_interpolation(
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
    point: Tuple[float, float],
) -> float:
    """二维散点数据的薄板样条插值。

寻找满足所有数据点且最小化弯曲能量的曲面 f(x,y) = α₀ + α₁x + α₂y + Σ wᵢ φ(‖(x,y)−(xᵢ,yᵢ)‖)，
其中 φ(r) = r² ln( r ) 是二维的标准 TPS 核。

参数
----
x_data, y_data : 空间坐标的一维数组，每个条目对应一个数据点（传入的是与 z_data 长度相同的扁平化坐标数组；不存在隐式的 meshgrid 模式）。
    点必须互不相同，且不能全部共线。
z_data : 函数值的一维数组
point  : 用于评估插值函数的 (x, y) 元组

返回
----
在 *point* 处的插值结果。

异常
----
ValueError
    如果数组长度不同，给定的点数少于三个，点重复，或所有点共线（此时曲面的仿射部分无法确定且系统奇异）。





"""
    if x_data.shape[0] != y_data.shape[0] or x_data.shape[0] != z_data.shape[0]:
        raise ValueError("X, Y, and Z vectors must have equal number of elements.")
    if x_data.shape[0] < 3:
        raise ValueError(
            "At least three points are required for thin plate spline interpolation."
        )
    if len(set(zip(x_data, y_data))) != x_data.shape[0]:
        raise ValueError("Duplicate (x, y) points detected.")

    x_data = x_data.astype(float)
    y_data = y_data.astype(float)
    z_data = z_data.astype(float)

    N = x_data.shape[0]

    # 共线站点会使 [K P; P^T 0] 奇异。中心化矩阵的秩
    # 坐标是缩放后的（基于最大奇异值的 SVD 容差）
    # 奇异值），因此它不依赖于 x 和 y 的单位。
    centred = np.column_stack((x_data - x_data.mean(), y_data - y_data.mean()))
    if np.linalg.matrix_rank(centred) < 2:
        raise ValueError(
            "Data points are collinear; thin plate spline interpolation "
            "requires points that do not all lie on one line."
        )

    # 核矩阵 K_{ij} = φ(‖pᵢ − pⱼ‖)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                r = np.hypot(x_data[i] - x_data[j], y_data[i] - y_data[j])
                K[i, j] = _tps_kernel(r)

    # 多项式矩阵 P = [1, x, y]
    P = np.vstack((np.ones(N), x_data, y_data)).T

    # 增强系统：[K P; P^T 0] [w; α] = [z; 0]
    system = np.vstack((np.hstack((K, P)), np.hstack((P.T, np.zeros((3, 3))))))
    b = np.hstack((z_data, np.zeros(3)))

    try:
        coefficients = np.linalg.solve(system, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Thin plate spline system is singular for the given data."
        ) from exc
    w = coefficients[:N]
    a = coefficients[N:]

    # 在查询点处求值
    x, y = point
    U = np.zeros(N)
    for i in range(N):
        r = np.hypot(x - x_data[i], y - y_data[i])
        U[i] = _tps_kernel(r)

    return a[0] + a[1] * x + a[2] * y + np.dot(w, U)
