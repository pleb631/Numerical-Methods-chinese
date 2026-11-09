# 本文件为 qr_method.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from scipy.linalg import hessenberg


def qr_decomposition(A: np.ndarray):
    return np.linalg.qr(A)


def _eigenvalues_2x2(block: np.ndarray):
    trace = np.trace(block)
    det = np.linalg.det(block)
    discriminant = (trace / 2) ** 2 - det
    if discriminant >= 0:
        return [trace / 2 + np.sqrt(discriminant), trace / 2 - np.sqrt(discriminant)]
    return [
        trace / 2 + 1j * np.sqrt(-discriminant),
        trace / 2 - 1j * np.sqrt(-discriminant),
    ]


def _wilkinson_shift(block: np.ndarray):
    """威尔金森位移：末尾 2x2 块中距离其右下角元素最近的特征值。如果该块具有复共轭对，则返回 None，此时不存在实位移，转而使用双重位移步。
    
    
    
"""
    corner = block[-1, -1]
    mu1, mu2 = _eigenvalues_2x2(block)
    if np.iscomplexobj(mu1):
        return None
    return float(mu1 if abs(mu1 - corner) <= abs(mu2 - corner) else mu2)


def _shifted_qr_step(A_k: np.ndarray) -> np.ndarray:
    """实矩阵上的单次位移 QR 步。给定实威尔金森位移 mu，执行 A - mu I = QR，随后 A <- RQ + mu I。如果末尾 2x2 块具有复特征值 mu 和 conj(mu)，则同时使用两者（弗朗西斯双重位移）：Q R = (A - mu I)(A - conj(mu) I) = A^2 - trace*A + det*I 保持为实矩阵，且 A <- Q^T A Q。
    
    
    
    
    
"""
    block = A_k[-2:, -2:]
    n = A_k.shape[0]
    I = np.eye(n)
    shift = _wilkinson_shift(block)
    if shift is not None:
        Q, R = qr_decomposition(A_k - shift * I)
        return R @ Q + shift * I
    M = A_k @ A_k - np.trace(block) * A_k + np.linalg.det(block) * I
    Q, _ = qr_decomposition(M)
    return Q.T @ A_k @ Q


def qr_algorithm(A: np.ndarray, tol: float = 1e-8, max_iterations: int = 2000):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    A_k = hessenberg(A).astype(float)
    eigenvalues = []

    iterations = 0  # 跟踪迭代次数
    while A_k.shape[0] > 0:
        for _ in range(max_iterations - iterations):
            iterations += 1
            if A_k.shape[0] == 1:
                eigenvalues.append(A_k[0, 0])
                A_k = np.empty((0, 0))
                break

            if abs(A_k[-1, -2]) < tol:
                eigenvalues.append(A_k[-1, -1])
                A_k = A_k[:-1, :-1]
                break

            if A_k.shape[0] == 2:
                eigenvalues.extend(_eigenvalues_2x2(A_k))
                A_k = np.empty((0, 0))
                break

            if abs(A_k[-2, -3]) < tol:
                # 末尾 2x2 块已解耦（例如为复共轭对）
                eigenvalues.extend(_eigenvalues_2x2(A_k[-2:, -2:]))
                A_k = A_k[:-2, :-2]
                break

            Q, R = qr_decomposition(A_k)
            A_k = R @ Q

        if A_k.shape[0] > 0 and iterations >= max_iterations:
            raise ValueError(
                "QR algorithm did not converge within the maximum number of iterations."
            )

    return np.sort(eigenvalues)


def qr_algorithm_with_shifts(
    A: np.ndarray, tol: float = 1e-8, max_iterations: int = 2000
):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    A_k = hessenberg(A).astype(float)
    eigenvalues = []

    iterations = 0  # 跟踪迭代次数
    while A_k.shape[0] > 0:
        for _ in range(max_iterations - iterations):
            iterations += 1
            if A_k.shape[0] == 1:
                eigenvalues.append(A_k[0, 0])
                A_k = np.empty((0, 0))
                break

            if abs(A_k[-1, -2]) < tol:
                eigenvalues.append(A_k[-1, -1])
                A_k = A_k[:-1, :-1]
                break

            if A_k.shape[0] == 2:
                eigenvalues.extend(_eigenvalues_2x2(A_k))
                A_k = np.empty((0, 0))
                break

            if abs(A_k[-2, -3]) < tol:
                # 末尾 2x2 块已解耦（例如为复共轭对）
                eigenvalues.extend(_eigenvalues_2x2(A_k[-2:, -2:]))
                A_k = A_k[:-2, :-2]
                break

            A_k = _shifted_qr_step(A_k)

        if A_k.shape[0] > 0 and iterations >= max_iterations:
            raise ValueError(
                "QR algorithm with shifts did not converge within the maximum number of iterations."
            )

    return np.sort(eigenvalues)
