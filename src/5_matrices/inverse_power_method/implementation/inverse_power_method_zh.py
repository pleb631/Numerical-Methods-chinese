# 本文件为 inverse_power_method.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from typing import Optional, Tuple


def inverse_power_method(
    A: np.ndarray,
    tol: float = 1e-8,
    max_iterations: int = 10000,
    x0: Optional[np.ndarray] = None,
    shift: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """移位逆迭代：寻找最接近 `shift` 的 A 的特征值（当 shift = 0 时为模最小的特征值）以及对应的特征向量。
    
    每一步求解 (A - shift*I) y = x 并归一化 y。误差在每次迭代中大约按 |lambda_closest - shift| / |lambda_next - shift| 缩小，其中 lambda_next 是第二接近的特征值。返回的特征值为瑞利商 y^T A y，迭代在变化量小于 `tol` 时停止（对于对称矩阵 A，此时特征向量精度约为 sqrt(tol)）。若存在复数特征值、A - shift*I 奇异或 `max_iterations` 耗尽，则抛出 ValueError。

    
    
    
    
    
    
    
"""
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    eigenvalues = np.linalg.eigvals(A)
    if not np.all(np.abs(np.imag(eigenvalues)) < 1e-10):
        raise ValueError("Matrix has complex eigenvalues.")

    n = A.shape[0]
    I = np.eye(n)
    A_shifted = A - shift * I

    if x0 is None:
        x = np.random.rand(n)
    else:
        x = x0.astype(float)

    x /= np.linalg.norm(x)
    eigenvalue = 0.0

    for iteration in range(max_iterations):
        try:
            y = np.linalg.solve(A_shifted, x)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Matrix (A - shift * I) is singular and cannot be inverted."
            )

        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            raise ValueError("Encountered zero vector during iterations.")
        y /= y_norm

        eigenvalue_new = np.dot(y, A @ y)

        # 1. 此外要求残差较小：特征值距离相等时
        # 2. 从移位处，瑞利商可能收敛到非特征值。
        residual = np.linalg.norm(A @ y - eigenvalue_new * y)
        if np.abs(eigenvalue_new - eigenvalue) < tol and residual <= np.sqrt(
            tol
        ) * np.linalg.norm(A, 2):
            return eigenvalue_new, y
        eigenvalue = eigenvalue_new
        x = y

    raise ValueError(
        "Inverse Power method did not converge within the maximum number of iterations."
    )
