# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def plot_eigenvector_transformation(A):
    """可视化矩阵如何变换向量，并突出显示特征向量。
    
"""
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # 生成单位圆上的点。
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])

    # 变换圆。
    transformed = A @ circle

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 带有特征向量的原始空间。
    axes[0].plot(circle[0], circle[1], "b-", alpha=0.3, label="Unit circle")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i].real
        axes[0].quiver(
            0, 0, v[0], v[1],
            angles="xy", scale_units="xy", scale=1,
            color=f"C{i}", linewidth=2,
            label=f"v{i+1} (λ={eigenvalues[i].real:.2f})",
        )
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].set_aspect("equal")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].set_title("Original Space")
    axes[0].legend()

    # 变换后的空间。
    axes[1].plot(transformed[0], transformed[1], "r-", alpha=0.3, label="Transformed")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i].real
        Av = A @ v
        axes[1].quiver(
            0, 0, Av[0], Av[1],
            angles="xy", scale_units="xy", scale=1,
            color=f"C{i}", linewidth=2,
            label=f"Av{i+1} = {eigenvalues[i].real:.2f}·v{i+1}",
        )
    axes[1].set_xlim(-6, 6)
    axes[1].set_ylim(-6, 6)
    axes[1].set_aspect("equal")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].set_title("Transformed Space (A @ v)")
    axes[1].legend()

    plt.suptitle("Eigenvector Transformation Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    A = np.array([[4, 1], [2, 3]])
    plot_eigenvector_transformation(A)
