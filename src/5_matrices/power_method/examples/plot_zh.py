# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


def power_method(A, max_iterations=100, tolerance=1e-6):
    """幂法用于寻找主特征值及其对应的特征向量。
    
参数：
        A (ndarray): 方阵 numpy 矩阵。
        max_iterations (int): 最大迭代次数。
        tolerance (float): 特征值的收敛阈值。
    
返回：
        dominant_eigenvalue (float): 模最大的特征值。
        dominant_eigenvector (ndarray): 对应的特征向量。
    
"""
    n = A.shape[0]
    x = np.random.rand(n)  # 1. 初始随机向量
    x = x / norm(x)  # 2. 归一化初始向量

    eigenvalues = []  # 跟踪特征值收敛

    for i in range(max_iterations):
        # 矩阵 - 向量乘法
        y = np.dot(A, x)
        # 使用瑞利商估算主特征值
        eigenvalue = np.dot(x.T, y)
        eigenvalues.append(eigenvalue)

        # 归一化所得向量
        x_next = y / norm(y)

        # 检查收敛性
        if norm(x_next - x) < tolerance:
            print(f"Converged after {i+1} iterations.")
            break

        x = x_next

    dominant_eigenvalue = eigenvalue
    dominant_eigenvector = x

    # 绘制收敛图
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o", linestyle="-")
    plt.title("Convergence of Dominant Eigenvalue")
    plt.xlabel("Iteration")
    plt.ylabel("Eigenvalue Estimate")
    plt.grid(True)
    plt.show()

    return dominant_eigenvalue, dominant_eigenvector


# 示例矩阵 A
A = np.array([[4, 1], [2, 3]])

# 执行幂法以计算主特征值和特征向量
eigenvalue, eigenvector = power_method(A)
print("\nDominant Eigenvalue (Power Method):", eigenvalue)
print("Dominant Eigenvector (Power Method):", eigenvector)
