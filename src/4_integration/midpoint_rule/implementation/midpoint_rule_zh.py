# 本文件为 midpoint_rule.py 的中文注释版本（代码逻辑与英文版完全一致）
from typing import Callable
import numpy as np


def midpoint_rule(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    if n <= 0:
        raise ValueError("Number of intervals must be positive.")
    h = (b - a) / n
    midpoints = a + h * (np.arange(n) + 0.5)
    values = f(midpoints)
    if np.isscalar(values):
        values = np.full_like(midpoints, values, dtype=float)
    return h * np.sum(values)


def midpoint_rule_multidim(
    f: Callable[[np.ndarray], float], bounds: list, n: int
) -> float:
    if n <= 0:
        raise ValueError("Number of intervals must be positive.")
    h = np.array([(b - a) / n for a, b in bounds])
    grids = [np.linspace(a + h_i / 2, b - h_i / 2, n) for (a, b), h_i in zip(bounds, h)]
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack(mesh, axis=-1).reshape(-1, len(bounds))
    # 1. 尝试对所有点同时应用 f；仅接受标量或每个点一个值。
    # 2. 否则，针对每个点的 f（例如 lambda x: x[0] + x[1]）会静默地
    # 3. 错误解释 (N, dims) 数组，因此回退到逐行评估。
    try:
        values = np.asarray(f(points), dtype=float)
        if values.ndim == 0:
            values = np.full(points.shape[0], float(values))
        elif values.shape != (points.shape[0],):
            raise ValueError("f is not vectorized")
    except Exception:
        values = np.apply_along_axis(f, 1, points)
    return np.prod(h) * np.sum(values)
