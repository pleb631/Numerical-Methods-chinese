# 本文件为 power_method.py 的中文注释版本（代码逻辑与英文版完全一致）
# power_method.py
import numpy as np
from typing import Optional, Tuple


def power_method(
    A: np.ndarray,
    tol: float = 1e-8,
    max_iterations: int = 1000,
    x0: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """幂法：估算实矩阵 A 的模最大的特征值及其对应的单位特征向量。收敛为线性，收敛速率为 |lambda_2| / |lambda_1|，且要求存在唯一的占优特征值，以及初始向量在其特征向量方向上具有分量。若迭代次数为零或 `max_iterations` 耗尽（对于复数占优特征值，此时估计值会振荡，也会发生此情况），则抛出 ValueError。
    
    
    
    
    
    
"""
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    n = A.shape[0]
    if x0 is None:
        x = np.ones(n)
    else:
        x = x0.astype(float)
    x /= np.linalg.norm(x)
    eigenvalue = 0.0
    for _ in range(max_iterations):
        x_old = x.copy()
        y = A @ x
        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            raise ValueError("Encountered zero vector during iterations.")
        x_new = y / y_norm
        if np.dot(x_old, x_new) < 0:
            x_new = -x_new
        eigenvalue_new = np.dot(A @ x_old, x_new)
        # 特征值估计本身不足以（对于 a
        # 旋转时它卡在 0，因此还需要一个小的残差。
        residual = np.linalg.norm(A @ x_new - eigenvalue_new * x_new)
        if np.abs(eigenvalue_new - eigenvalue) < tol and residual <= np.sqrt(
            tol
        ) * np.linalg.norm(A, 2):
            x_new[np.abs(x_new) < 1e-10] = 0.0
            return float(eigenvalue_new), x_new
        eigenvalue = eigenvalue_new
        x = x_new
    raise ValueError(
        "Power method did not converge within the maximum number of iterations."
    )
