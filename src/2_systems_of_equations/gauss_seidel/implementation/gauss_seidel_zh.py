# 本文件为 gauss_seidel.py 的中文注释版本（代码逻辑与英文版完全一致）
import warnings
import numpy as np
from typing import Optional


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    epsilon: float = 1e-8,
    max_iterations: int = 100,
    max_iter: Optional[int] = None,
) -> np.ndarray:
    """1. 高斯 - 赛德尔迭代求解 Ax = b。
    
当 ||x_new - x_prev||_2 < epsilon 时停止。
    
参数:
        max_iterations: 最大迭代次数（与 jacobi_method 中的名称相同）。
        max_iter: max_iterations 的已弃用别名；如果提供则覆盖 max_iterations。
    
异常:
        ValueError: 如果 A 不是方阵、对角线元素为零，或迭代在 max_iterations 次迭代内未收敛。
    
    
"""
    if max_iter is not None:
        warnings.warn(
            "'max_iter' is deprecated; use 'max_iterations' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        max_iterations = max_iter
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if np.any(np.diag(A) == 0):
        raise ValueError("Matrix A has zero diagonal elements.")
    n: int = A.shape[0]
    x: np.ndarray = (
        np.zeros_like(b, dtype=np.double) if x0 is None else x0.astype(float)
    )
    for _ in range(max_iterations):
        x_prev: np.ndarray = x.copy()
        for i in range(n):
            x[i] = (
                b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1 :], x_prev[i + 1 :])
            ) / A[i, i]
        if np.linalg.norm(x - x_prev) < epsilon:
            return x
    raise ValueError(
        "Gauss-Seidel method did not converge within the maximum number of iterations."
    )
