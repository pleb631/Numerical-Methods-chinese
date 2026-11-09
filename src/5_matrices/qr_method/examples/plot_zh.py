# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def qr_method_visualization(A, max_iterations=50, tol=1e-8):
    """可视化 QR 方法将矩阵迭代转换为上三角形式的过程。
    
参数：
        A (numpy.ndarray): 输入方阵。
        max_iterations (int): 最大迭代次数。
        tol (float): 非对角线元素的收敛容差。
    
"""
    A_k = A.copy()
    n = A.shape[0]
    off_diagonal_norms = []

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    step_intervals = max_iterations // 5

    for i in range(max_iterations):
        # 执行 QR 分解
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q

        # 计算非对角线范数以检查收敛性
        off_diag = A_k - np.diag(np.diag(A_k))
        off_diag_norm = np.linalg.norm(off_diag)
        off_diagonal_norms.append(off_diag_norm)

        # 若非对角线元素足够小则停止
        if off_diag_norm < tol:
            print(f"Converged at iteration {i+1}")
            break

        # 在特定间隔绘制进度
        if i % step_intervals == 0:
            ax_idx = i // step_intervals
            axes[ax_idx].imshow(A_k, cmap="viridis", interpolation="nearest")
            axes[ax_idx].set_title(f"Iteration {i}")
            axes[ax_idx].axis("off")

    # 最终收敛可视化
    axes[-1].imshow(A_k, cmap="viridis", interpolation="nearest")
    axes[-1].set_title(f"Iteration {i+1} (Final)")
    axes[-1].axis("off")

    plt.suptitle("QR Method: Matrix Evolution Towards Upper-Triangular Form")
    plt.tight_layout()
    plt.show()

    # 绘制非对角线元素的收敛情况
    plt.figure(figsize=(8, 6))
    plt.semilogy(off_diagonal_norms, label="Off-diagonal Norm")
    plt.xlabel("Iteration")
    plt.ylabel("Log Scale: Off-diagonal Norm")
    plt.title("Convergence of Off-diagonal Elements")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 示例：输入矩阵
    A = np.array([[4, 1, 2], [1, 3, 1], [2, 1, 5]], dtype=float)

    print("Initial Matrix A:")
    print(A)
    qr_method_visualization(A)
