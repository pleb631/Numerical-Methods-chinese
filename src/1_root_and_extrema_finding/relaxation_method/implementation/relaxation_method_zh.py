# 本文件为 relaxation_method.py 的中文注释版本（代码逻辑与英文版完全一致）
# relaxation_method.py
import numpy as np
from typing import Callable, Optional


def relaxation_method(
    func: Callable[[float], float],
    initial_guess: float,
    omega: Optional[float] = None,
    tol: float = 1e-6,
    max_iterations: int = 100000,
) -> float:
    """松弛法（定点法）用于求解 x = func(x)。每一步为 x_new = (1 - w) * x + w * func(x)。如果 omega 为 None，则 w 在每次迭代中根据 func 的数值斜率 g' 自适应选择：当 |g'| >= 1 时，w = -1 / g'；否则 w = 1（普通定点迭代）。否则，使用提供的 omega 作为每一步的值。当 |x_new - x| < tol 时停止。异常：ValueError：如果方法在 max_iterations 次迭代内未收敛。
    

    
    
    
    

    

    
    
"""
    x = initial_guess
    delta = 1e-8
    for _ in range(max_iterations):
        if omega is None:
            # 基于 func 在 x 处斜率的自适应松弛因子
            f_prime = (func(x + delta) - func(x - delta)) / (2 * delta)
            w = -1.0 / f_prime if abs(f_prime) >= 1 else 1.0
        else:
            w = omega
        x_new = (1 - w) * x + w * func(x)
        if np.abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError(
        "Relaxation method did not converge within the maximum number of iterations."
    )
