# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# 使作为脚本运行时，同级 `implementation` 包可被导入。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from implementation.gaussian_elimination import gaussian_elimination  # noqa: E402


def plot_gaussian_elimination_process(A, b):
    """可视化高斯消元法的逐步过程。
    
该实现仅返回解，因此此处重复前向消元过程（相同的部分主元选取和主元归一化）以记录中间增广矩阵。
    
    参数：
        A (numpy.ndarray): 系数矩阵。
    b (numpy.ndarray): 右端向量。
    
    
    
"""
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    steps = [Ab.copy()]

    # 带可视化步骤的前向消元
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:n, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(i + 1, n):
            Ab[j] -= Ab[j, i] * Ab[i]
        steps.append(Ab.copy())

    # 绘制高斯消元法的每一步
    fig, axes = plt.subplots(1, len(steps), figsize=(20, 4))
    for idx, step in enumerate(steps):
        axes[idx].imshow(step, cmap="coolwarm", aspect="auto")
        axes[idx].set_title(f"Step {idx}")
        axes[idx].axis("off")
    plt.suptitle("Gaussian Elimination: Forward Elimination Steps")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例系统
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    print("Solving system using Gaussian elimination:")
    solution = gaussian_elimination(A, b)
    print(f"Solution: {solution}")
    print(f"Check (np.linalg.solve): {np.linalg.solve(A, b)}")

    # 可视化消元过程
    plot_gaussian_elimination_process(A, b)
