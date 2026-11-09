# 本文件为 euler.py 的中文注释版本（代码逻辑与英文版完全一致）
# euler_solver.py
import numpy as np
from typing import Callable, Tuple


def euler_method(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    y0: np.ndarray,
    t_end: float,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用显式欧拉法从 t0 积分 y' = f(t, y) 至 t_end。

步长为 h，但最后一步会缩短，使得 (t_end - t0) / h 不是整数时，t[-1] 恰好等于 t_end。

返回 (t, y)。当 y0 为单分量时，y 的形状为 (n + 1,)；当 y0 有 d > 1 个分量时，y 的形状为 (n + 1, d)；y[i] 近似 y(t[i])。


"""
    if h <= 0:
        raise ValueError("Step size h must be positive.")
    if t_end < t0:
        raise ValueError("t_end must be greater than or equal to t0.")
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_full = int(np.floor((t_end - t0) / h + 1e-9))
    t = t0 + h * np.arange(n_full + 1)
    if t_end - t[-1] > 1e-9 * h:
        t = np.append(t, t_end)  # 缩短的最终步骤
    else:
        t[-1] = t_end
    y = np.zeros((len(t), y0.size))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i + 1] = y[i] + (t[i + 1] - t[i]) * f(t[i], y[i])
    return t, (y.reshape(-1) if y0.size == 1 else y)
