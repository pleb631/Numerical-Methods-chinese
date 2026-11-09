# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# 当作为脚本运行时，使同级 `implementation` 包可导入。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from implementation.gauss_seidel import gauss_seidel  # noqa: E402


def _error_history(A, b, x0, method, max_iterations=100):
    """记录雅可比或高斯 - 赛德尔迭代中每一次扫描的 ||x_new - x||_inf。实现仅返回最终解，因此此处重新计算绘图所需的历​​史数据。
    
    
    
"""
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.astype(float)
    errors = []
    for _ in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            src = x if method == "gauss_seidel" else x_old
            sum1 = np.dot(A[i, :i], src[:i])
            sum2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        error = np.linalg.norm(x - x_old, ord=np.inf)
        errors.append(error)
        if error < 1e-8:
            break
    return errors


def plot_convergence(errors_gs, errors_jacobi):
    """绘制高斯 - 赛德尔和雅可比方法的收敛性。参数：errors_gs (list): 高斯 - 赛德尔方法的误差。errors_jacobi (list): 雅可比方法的误差。
    

    
    
    
"""
    plt.figure(figsize=(8, 6))
    plt.semilogy(errors_gs, label="Gauss-Seidel", linewidth=2)
    plt.semilogy(errors_jacobi, label="Jacobi", linestyle="--", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Error (log scale)")
    plt.title("Convergence Comparison: Gauss-Seidel vs Jacobi")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 示例：系数矩阵和右端向量
    A = np.array([[5, -1], [7, 8]], dtype=float)
    b = np.array([6, 20], dtype=float)

    # 初始猜测
    x0 = np.array([0, 0], dtype=float)

    print("Solving system using Gauss-Seidel method:")
    x_gs = gauss_seidel(A, b, x0)
    errors_gs = _error_history(A, b, x0, "gauss_seidel")
    print(f"Gauss-Seidel Solution: {x_gs} ({len(errors_gs)} sweeps)")

    print("\nSolving system using Jacobi method:")
    errors_jacobi = _error_history(A, b, x0, "jacobi")
    print(f"Jacobi converged in {len(errors_jacobi)} iterations")
    print(f"Exact solution: {np.linalg.solve(A, b)}")

    # 绘制收敛性比较
    plot_convergence(errors_gs, errors_jacobi)
