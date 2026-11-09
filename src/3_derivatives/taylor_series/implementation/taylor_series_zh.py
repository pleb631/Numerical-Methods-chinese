# 本文件为 taylor_series.py 的中文注释版本（代码逻辑与英文版完全一致）
from typing import Callable
import numpy as np


def taylor_series(
    f: Callable[[float], float],
    a: float,
    n: int,
    x: float,
    derivatives: Callable[[int], float],
) -> float:
    k = np.arange(n)
    # 1. 浮点运算：int64 阶乘/幂在 n > 20 时静默溢出。
    factorial = np.cumprod(np.append(1.0, k[1:].astype(float)))
    terms = derivatives(k) / factorial * (x - a) ** k.astype(float)
    return np.sum(terms)
