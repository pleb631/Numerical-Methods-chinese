# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# 该实现仅返回最终解，因此这里重新计算收敛图所需的历史数据（使用相同的更新和停止规则）。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from implementation.jacobi_method import jacobi_method  # noqa: E402


def jacobi_error_history(A, b, x0=None, max_iterations=100, tol=1e-8):
    """记录每次雅可比迭代中的 ||x_new - x||_inf。
    

    
    
"""
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.astype(float)
    errors = []
    for _ in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        x = x_new
        if error < tol:
            break
    return errors


if __name__ == "__main__":
    # 对角占优 2x2 系统
    A = np.array([[5, -1], [7, 8]], dtype=float)
    b = np.array([6, 20], dtype=float)
    x0 = np.array([0, 0], dtype=float)

    x = jacobi_method(A, b, x0)
    errors = jacobi_error_history(A, b, x0)
    x_exact = np.linalg.solve(A, b)
    print(f"Jacobi solution: {x}")
    print(f"Exact solution:  {x_exact}")

    plt.figure(figsize=(8, 6))
    plt.semilogy(range(1, len(errors) + 1), errors, "b-o", markersize=4, linewidth=2)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Error (log scale)", fontsize=14)
    plt.title("Jacobi Method Convergence", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
