# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def gradient_descent_minimize(f, grad_f, x0, learning_rate=0.1, tol=1e-6, max_iter=1000):
    """使用梯度下降法最小化函数。
    
参数：
        f (callable): 目标函数。
        grad_f (callable): 目标函数的梯度。
        x0 (numpy.ndarray): 初始点。
        learning_rate (float): 步长。
        tol (float): 收敛容差。
        max_iter (int): 最大迭代次数。
    
返回：
        path (list): 优化过程中访问的点列表。
    
"""
    x = x0.copy()
    path = [x.copy()]

    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - learning_rate * g
        path.append(x.copy())

    return path


if __name__ == "__main__":
    # 二次函数：f(x, y) = 3x^2 + y^2
    def f(x):
        return 3 * x[0] ** 2 + x[1] ** 2

    def grad_f(x):
        return np.array([6 * x[0], 2 * x[1]])

    x0 = np.array([3.0, 4.0])
    path = gradient_descent_minimize(f, grad_f, x0, learning_rate=0.15)
    path = np.array(path)

    print(f"Starting point: {x0}")
    print(f"Final point:    ({path[-1][0]:.6f}, {path[-1][1]:.6f})")
    print(f"Iterations:     {len(path) - 1}")

    # 创建等高线图
    x_range = np.linspace(-4, 4, 200)
    y_range = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 3 * X ** 2 + Y ** 2

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8)

    # 绘制优化路径
    ax.plot(path[:, 0], path[:, 1], "r.-", markersize=6, linewidth=1.5, label="Gradient descent path")
    ax.plot(path[0, 0], path[0, 1], "go", markersize=10, label="Start")
    ax.plot(path[-1, 0], path[-1, 1], "r*", markersize=15, label="Minimum")

    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_title("Gradient Descent on $f(x, y) = 3x^2 + y^2$", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
