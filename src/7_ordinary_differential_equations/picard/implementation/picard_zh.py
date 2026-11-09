# 本文件为 picard.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from typing import Callable, Tuple


def picard_method(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    y0: np.ndarray,
    t_end: float,
    h: float,
    tol: float = 1e-6,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """1. 使用定点（皮卡型）迭代按步长积分 y' = f(t, y)。

这不是对整个函数的皮卡迭代（参见说明）。每一步求解隐式中点方程
y_{n+1} = y_n + h f(t_n + h/2, (y_n + y_{n+1})/2)（通过重复代入求解，
当 h*L/2 < 1 时收敛），因此该方法为二阶。

步长长度为 h，但最后一步在 (t_end - t0) / h 不是整数时会缩短，
使得 t[-1] 恰好等于 t_end。

返回 (t, y)。当 y0 为单分量时，y 的形状为 (n + 1,)；
当 y0 有 d > 1 个分量时，y 的形状为 (n + 1, d)；y[i] 近似 y(t[i])。

"""
    if h <= 0:
        raise ValueError("Step size h must be positive.")
    if t_end < t0:
        raise ValueError("t_end must be greater than or equal to t0.")
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_full = int(np.floor((t_end - t0) / h + 1e-9))
    t = t0 + h * np.arange(n_full + 1)
    if t_end - t[-1] > 1e-9 * h:
        t = np.append(t, t_end)  # 2. 缩短的最后一步
    else:
        t[-1] = t_end
    y = np.zeros((len(t), y0.size))
    y[0] = y0
    for i in range(len(t) - 1):
        hi = t[i + 1] - t[i]
        y_prev = y[i]
        y_next = y_prev.copy()
        for _ in range(max_iterations):
            y_new = y_prev + hi * f(t[i] + 0.5 * hi, 0.5 * (y_prev + y_next))
            if np.linalg.norm(y_new - y_next, ord=np.inf) < tol:
                y_next = y_new
                break
            y_next = y_new
        else:
            raise ValueError(f"Picard iteration did not converge at step {i}.")
        y[i + 1] = y_next
    return t, (y.reshape(-1) if y0.size == 1 else y)
