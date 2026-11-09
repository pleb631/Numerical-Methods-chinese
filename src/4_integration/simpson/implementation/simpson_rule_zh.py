# 本文件为 simpson_rule.py 的中文注释版本（代码逻辑与英文版完全一致）
from typing import Callable, List
import numpy as np


def simpson_rule(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    if n <= 0 or n % 2 != 0:
        raise ValueError("Number of intervals must be a positive even integer.")
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    h = (b - a) / n
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


def simpson_rule_multidim(
    f: Callable[[np.ndarray], np.ndarray], bounds: List[tuple], n: int
) -> float:
    if n <= 0 or n % 2 != 0:
        raise ValueError("Number of intervals must be a positive even integer.")
    dims = len(bounds)
    grids = [np.linspace(a, b, n + 1) for a, b in bounds]
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack(mesh, axis=-1)
    flat_points = points.reshape(-1, dims)
    # 仅接受标量或每个点一个值；否则 f 为每点函数。
    try:
        flat_vals = np.asarray(f(flat_points), dtype=float)
        if flat_vals.ndim == 0:
            flat_vals = np.full(flat_points.shape[0], float(flat_vals))
        elif flat_vals.shape != (flat_points.shape[0],):
            raise ValueError("f is not vectorized")
    except Exception:
        flat_vals = np.apply_along_axis(f, 1, flat_points)
    vals = flat_vals.reshape([n + 1] * dims)
    h = [(b - a) / n for a, b in bounds]

    w_1d = np.ones(n + 1)
    w_1d[1:-1:2] = 4
    w_1d[2:-2:2] = 2
    W = w_1d
    for _ in range(dims - 1):
        W = np.outer(W, w_1d)
    W = W.reshape([n + 1] * dims)

    return np.sum(vals * W) * np.prod(h) / 3**dims
