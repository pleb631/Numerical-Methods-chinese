# 本文件为 golden_ratio_search.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from typing import Callable


def golden_ratio_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> float:
    """在区间 [a, b] 上对单峰函数进行黄金分割法搜索以求得最小值。
    
每次迭代进行一次新的函数评估：幸存的内部点（在收缩后保留）与其存储的值一起被重新用作较小区间的另一个内部点。
    
    当 b - a < tol 时停止，或者如果 f(c) == f(d) 时立即停止（两侧均无信息可丢弃），返回当前区间的中点。
    
异常：
        ValueError：如果方法在 max_iterations 次迭代内未收敛。
    

    
    
"""
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = f(c)
    fd = f(d)
    for _ in range(max_iterations):
        if fc < fd:
            # 最小值在 [a, d] 区间：旧的 c 变为新的 d。
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            if np.abs(b - a) < tol:
                return (b + a) / 2
            fc = f(c)
        elif fc > fd:
            # 最小值在 [c, b] 区间：旧的 d 变为新的 c。
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            if np.abs(b - a) < tol:
                return (b + a) / 2
            fd = f(d)
        else:
            # 如果 fc == fd，返回中点
            return (a + b) / 2
    raise ValueError(
        "Golden ratio search did not converge within the maximum number of iterations."
    )
