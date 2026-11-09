# 本文件为 singular_value_decomposition.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from typing import Tuple


def _truncate_small_singular_values(A: np.ndarray, S: np.ndarray) -> np.ndarray:
    tol = max(A.shape) * np.finfo(S.dtype).eps * np.max(S, initial=0.0)
    return np.where(S > tol, S, 0.0)


def singular_value_decomposition(
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    S = _truncate_small_singular_values(A, S)
    S_matrix = np.zeros((U.shape[0], Vt.shape[0]))
    np.fill_diagonal(S_matrix, S)
    return U, S_matrix, Vt


def singular_value_decomposition_reduced(
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S = _truncate_small_singular_values(A, S)
    S_matrix = np.diag(S)  # 紧凑对角矩阵
    return U, S_matrix, Vt
