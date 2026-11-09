# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd


def plot_svd(A):
    """计算并可视化矩阵 A 的奇异值分解 (SVD)
    

    
    
"""
    # 执行 SVD
    U, Sigma, VT = svd(A)

    # 构造对角矩阵 Sigma 用于可视化
    Sigma_matrix = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma_matrix, Sigma)

    # 打印结果
    print("Original Matrix A:")
    print(A)
    print("\nLeft Singular Vectors (U):")
    print(U)
    print("\nSingular Values (Sigma):")
    print(Sigma)
    print("\nRight Singular Vectors (V^T):")
    print(VT)

    # 绘制原始矩阵及其 SVD 分量
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    # 绘制原始矩阵 A
    axs[0].imshow(A, cmap="viridis", aspect="auto")
    axs[0].set_title("Original Matrix A")
    axs[0].axis("off")

    # 绘制左奇异向量 U
    axs[1].imshow(U, cmap="coolwarm", aspect="auto")
    axs[1].set_title("Left Singular Vectors (U)")
    axs[1].axis("off")

    # 绘制奇异值 Sigma
    axs[2].imshow(Sigma_matrix, cmap="coolwarm", aspect="auto")
    axs[2].set_title("Singular Values (Sigma)")
    axs[2].axis("off")

    # 绘制右奇异向量 VT
    axs[3].imshow(VT, cmap="coolwarm", aspect="auto")
    axs[3].set_title("Right Singular Vectors (V^T)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


# 示例矩阵 A
A = np.array([[3, 4], [2, 1], [0, 5]])

# 执行并绘制 SVD
plot_svd(A)
