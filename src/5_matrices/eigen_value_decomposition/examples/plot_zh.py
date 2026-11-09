# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv


def eigen_decomposition_visualize(A):
    """执行并可视化方阵 A 的特征值分解（EVD）。
    
参数：
        A (ndarray): 一个方阵。
    
"""
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = eig(A)
    D = np.real_if_close(np.diag(eigenvalues))  # 对角矩阵 D
    P = eigenvectors  # 具有特征向量作为列的矩阵 P
    P_inv = inv(P)  # P 的逆矩阵

    # 验证分解 A = P D P^{-1}
    A_reconstructed = np.real_if_close(P @ D @ P_inv)

    # 打印结果
    print("Original Matrix A:")
    print(A)
    print("\nEigenvalues (D):")
    print(D)
    print("\nEigenvectors (P):")
    print(P)
    print("\nReconstructed A (P D P^{-1}):")
    print(A_reconstructed)

    # 绘图：特征向量、特征值和重构
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 图 1：原始矩阵 A 变换（概念）
    axs[0].set_title("Original Matrix A")
    axs[0].imshow(A, cmap="viridis")
    axs[0].axis("off")
    axs[0].set_xlabel("A transformation")

    # 图 2：特征向量和特征值
    axs[1].set_title("Eigenvectors (P) and Eigenvalues (D)")
    axs[1].imshow(D, cmap="coolwarm")
    axs[1].set_xlabel("Diagonal Eigenvalues (D)")

    # 图 3：重构 A
    axs[2].set_title("Reconstructed Matrix A")
    axs[2].imshow(A_reconstructed, cmap="viridis")
    axs[2].axis("off")
    axs[2].set_xlabel("A = P D P^{-1}")

    plt.tight_layout()
    plt.show()


# 示例矩阵 A
A = np.array([[4, 1], [2, 3]])

# 执行并可视化特征值分解
eigen_decomposition_visualize(A)
