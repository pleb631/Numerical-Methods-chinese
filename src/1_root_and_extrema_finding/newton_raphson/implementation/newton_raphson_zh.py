# 本文件为 newton_raphson.py 的中文注释版本（代码逻辑与英文版完全一致）
# newton_raphson.py
import numpy as np
from typing import Callable, Optional


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> float:
    """牛顿法（牛顿 - 拉夫逊法）用于求实值函数的根。
    
参数：
        f (Callable[[float], float]): 用于求根的函数。
        df (Callable[[float], float]): 函数 f 的导数。
        x0 (float): 根的初始猜测值。
        tol (float, 可选): 收敛容差。默认为 1e-8。
        max_iterations (int, 可选): 最大迭代次数。默认为 1000。
    
返回：
        float: 函数的估计根。
    
异常：
        ValueError: 如果导数恰好为零（或牛顿步长不是有限值）。
        ValueError: 如果方法在最大迭代次数内未收敛。
    
"""
    x = x0
    for iteration in range(1, max_iterations + 1):
        fx = f(x)
        dfx = df(x)
        # 仅当导数恰好为零（或小到足以使步长
        # （溢出）将被拒绝；固定的绝对阈值会错误地
        # 拒绝自然尺度较小的函数，例如 1e-20 * (x - 2)。
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            step = fx / dfx if dfx != 0.0 else np.inf
        if not np.isfinite(step):
            raise ValueError(f"Derivative is zero at iteration {iteration}, x = {x}.")
        x_new = x - step
        if np.abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError(
        "Newton-Raphson method did not converge within the maximum number of iterations."
    )


def newton_raphson_system(
    F: Callable[[np.ndarray], np.ndarray],
    J: Callable[[np.ndarray], np.ndarray],
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> np.ndarray:
    """牛顿法（牛顿 - 拉夫逊法）用于求解非线性方程组的根。
    
参数：
        F (Callable[[np.ndarray], np.ndarray]): 方程组。
        J (Callable[[np.ndarray], np.ndarray]): 函数 F 的雅可比矩阵。
        x0 (Optional[np.ndarray], 可选): 根的初始猜测值。默认为 None。
        tol (float, 可选): 收敛容差。默认为 1e-8。
        max_iterations (int, 可选): 最大迭代次数。默认为 1000。
    
返回：
        np.ndarray: 方程组的估计根。
    
异常：
        ValueError: 如果在任何迭代中雅可比矩阵奇异。
        ValueError: 如果方法在最大迭代次数内未收敛。
        ValueError: 如果未提供初始猜测 x0。
    
"""
    if x0 is None:
        raise ValueError("Initial guess x0 must be provided")
    x = x0.astype(float)
    for iteration in range(1, max_iterations + 1):
        Fx = F(x)
        Jx = J(x)
        try:
            delta = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            raise ValueError(f"Jacobian is singular at iteration {iteration}, x = {x}.")
        x_new = x + delta
        if np.linalg.norm(delta, ord=np.inf) < tol:
            return x_new
        x = x_new
    raise ValueError(
        "Newton-Raphson method did not converge within the maximum number of iterations."
    )
