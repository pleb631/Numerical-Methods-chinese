# 本文件为 bisection_search.py 的中文注释版本（代码逻辑与英文版完全一致）
# bisection_search.py
import numpy as np
from typing import Callable


def bisection_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> float:
    fa: float = f(a)
    fb: float = f(b)

    # 检查任一端点是否为根
    if np.abs(fa) < tol:
        return a
    if np.abs(fb) < tol:
        return b

    if fa * fb > 0:
        raise ValueError("Function must have opposite signs at endpoints a and b.")

    for _ in range(max_iterations):
        c: float = (a + b) / 2
        fc: float = f(c)

        if np.abs(fc) < tol or (b - a) / 2 < tol:
            return c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    raise ValueError(
        "Bisection method did not converge within the maximum number of iterations."
    )
