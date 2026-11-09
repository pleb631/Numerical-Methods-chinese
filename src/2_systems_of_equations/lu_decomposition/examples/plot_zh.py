# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# 当作为脚本运行时，使同级 `implementation` 包可导入。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from implementation.lu_decomposition import lu_decomposition, solve_lu  # noqa: E402


def plot_matrix_decomposition(L, U):
    """可视化 LU 分解（PA = LU）的 L 和 U 矩阵。
    

    
    
    
"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(L, cmap="coolwarm", interpolation="none")
    axes[0].set_title("Lower Triangular Matrix L")
    axes[0].axis("off")

    axes[1].imshow(U, cmap="coolwarm", interpolation="none")
    axes[1].set_title("Upper Triangular Matrix U")
    axes[1].axis("off")

    plt.suptitle("LU Decomposition Visualization")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例系统
    A = np.array([[2, 3, -4], [3, -3, 2], [-2, 6, -1]], dtype=float)
    b = np.array([1, -2, 3], dtype=float)

    print("Solving system using LU decomposition (with partial pivoting):")
    P, L, U = lu_decomposition(A)
    x = solve_lu(P, L, U, b)
    print(f"Solution: {x}")
    print("P (Permutation Matrix):")
    print(P)
    print("L (Lower Triangular Matrix):")
    print(L)
    print("U (Upper Triangular Matrix):")
    print(U)
    print(f"max |PA - LU| = {np.abs(P @ A - L @ U).max():.2e}")

    # 绘制 LU 分解
    plot_matrix_decomposition(L, U)
