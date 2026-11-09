# 本文件为 eigenvalues_and_eigenvectors.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np


def find_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """通过求解特征多项式 det(A - λI) = 0 计算特征值。
    
"""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    coeffs = np.poly(matrix)  # 1. 多项式系数，用于 det(A - λI) = 0
    eigenvalues = np.roots(coeffs)  # 2. 多项式的根
    return eigenvalues


def _cluster_eigenvalues(eigenvalues: np.ndarray, scale: float):
    """["将计算出的特征值分组为数值上重复的根簇。", "重数为 m 的根仅被 np.roots 恢复至约 eps**(1/m) * scale 的精度，因此当它们彼此都在（安全因子乘以）该精度的范围内时，m 个特征值被声明为一个 m 重特征值。", "返回 (index_list, radius) 对的列表；radius 也是对于该簇，应将 A - mu*I 视为秩亏的精度。"]
    

    
    
    
    
    
"""
    eps = np.finfo(float).eps
    remaining = list(range(len(eigenvalues)))
    clusters = []
    while remaining:
        seed = eigenvalues[remaining[0]]
        for m in range(len(remaining), 0, -1):
            radius = 10 * scale * eps ** (1.0 / m)
            members = [i for i in remaining if abs(eigenvalues[i] - seed) <= radius]
            if len(members) >= m:
                break
        clusters.append((members, radius))
        remaining = [i for i in remaining if i not in members]
    return clusters


def find_eigenvectors(matrix: np.ndarray) -> np.ndarray:
    """计算单位特征向量；第 i 列属于 find_eigenvalues(matrix)[i]。数值上重复的特征值被聚类，且 (A - mu*I) 的零空间对每个簇仅通过具有尺度感知秩容差的 SVD 计算一次。其标准正交基分布在簇的列上，因此可对角化的重复特征值会获得一组完整的独立特征向量。对于亏损特征值（独立特征向量数量少于代数重数），基向量被循环复用，因此返回的矩阵是奇异的。
    

    
    
    
    
    
    
    
"""
    eigenvalues = find_eigenvalues(matrix)
    n = matrix.shape[0]
    A = np.asarray(matrix, dtype=complex if np.iscomplexobj(matrix) else float)
    scale = np.linalg.norm(A, 2) if n else 0.0
    eigenvectors = np.zeros((n, n), dtype=complex)

    for members, radius in _cluster_eigenvalues(eigenvalues, scale):
        mu = np.mean(eigenvalues[members])
        if abs(mu.imag) <= radius:  # 实特征值：保持在实数运算中
            mu = mu.real
        _, s, vh = np.linalg.svd(A - mu * np.eye(n))
        tol = max(radius, n * np.finfo(float).eps * scale)
        # 零奇异值（数值上）的右奇异向量张成
        # 零空间；总是至少有一个，且最多有 len(members) 个。
        nullity = int(np.clip(np.sum(s <= tol), 1, len(members)))
        basis = vh[n - nullity :].conj()
        for k, col in enumerate(members):
            eigenvectors[:, col] = basis[k % nullity]

    if np.allclose(eigenvectors.imag, 0):
        return eigenvectors.real
    return eigenvectors
