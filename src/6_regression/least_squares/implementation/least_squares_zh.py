# 本文件为 least_squares.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from numpy.linalg import LinAlgError


def least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """通过正规方程 A^T A x = A^T b 求解 min ||A x - b||_2。

A 的行数必须不少于列数，且列满秩。
允许 A 中存在全零行：它们对 A^T A 或 A^T b 没有贡献，因此仅对残差贡献一个常数 b_i^2。秩亏（包括全零列）将被拒绝并抛出 ValueError。



"""
    if A.ndim != 2 or A.shape[0] < A.shape[1]:
        raise ValueError("Matrix A must have at least as many rows as columns.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("The number of rows in A must match the size of vector b.")
    if np.linalg.matrix_rank(A) < A.shape[1]:
        raise ValueError("Matrix A does not have full column rank.")
    A = A.astype(float)
    b = b.astype(float)
    At = A.T
    AtA = At @ A
    Atb = At @ b
    try:
        # 显式使用正规方程以保持实现的教学性，
        # 尽管直接从 A 求解通常数值稳定性更高。
        U, s, Vt = np.linalg.svd(AtA, full_matrices=False)
        tol = max(AtA.shape) * np.finfo(s.dtype).eps * max(s)
        s_inv = np.where(s > tol, 1 / s, 0.0)
        AtA_pseudo_inv = Vt.T @ (s_inv[:, np.newaxis] * U.T)
        x = AtA_pseudo_inv @ Atb
    except LinAlgError:
        raise ValueError(
            "Singular matrix encountered during least squares computation."
        )
    return x
