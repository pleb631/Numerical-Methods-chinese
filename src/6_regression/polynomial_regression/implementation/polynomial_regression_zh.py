# 本文件为 polynomial_regression.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    augmented = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))
    for i in range(n):
        pivot = np.argmax(np.abs(augmented[i:, i])) + i
        if np.isclose(augmented[pivot, i], 0.0, atol=1e-12):
            raise ValueError("Matrix is singular or nearly singular.")
        if pivot != i:
            augmented[[i, pivot]] = augmented[[pivot, i]]
        augmented[i] = augmented[i] / augmented[i, i]
        for j in range(i + 1, n):
            augmented[j] -= augmented[j, i] * augmented[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i, -1] - np.dot(augmented[i, i + 1 : n], x[i + 1 : n])
    return x


def polynomial_regression(
    x_data: np.ndarray, y_data: np.ndarray, degree: int
) -> np.ndarray:
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("X and Y vectors must have equal number of elements.")
    if x_data.shape[0] < degree + 1:
        raise ValueError("Number of data points must be at least degree + 1.")
    if degree < 0:
        raise ValueError("Degree must be non-negative.")
    # 使用浮点数，以防止整数输入在求幂时溢出。
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    V = np.vander(x_data, N=degree + 1, increasing=True)
    # 将列缩放为单位范数，以便在 solve_linear_system 中进行奇异性测试。
    # 不依赖于 x 的大小（x 的幂次变化极大）。
    col_norms = np.linalg.norm(V, axis=0)
    if np.any(col_norms == 0.0):
        raise ValueError("Matrix is singular or nearly singular.")
    V = V / col_norms
    AtA = V.T @ V
    AtY = V.T @ y_data
    coefficients = solve_linear_system(AtA, AtY) / col_norms
    return coefficients
