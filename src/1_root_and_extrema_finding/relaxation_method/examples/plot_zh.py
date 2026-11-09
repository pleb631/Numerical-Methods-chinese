# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import matplotlib.pyplot as plt
import numpy as np


def g(x):
    """定点迭代函数：g(x) = (x^2 + 2) / 3"""
    return (x ** 2 + 2) / 3


def fixed_point_iteration_visualize(g_func, x0, tol=1e-6, max_iterations=50):
    """可视化定点迭代法求根的进度。
    
参数：
        g_func (function): 定点迭代函数。
        x0 (float): 初始起点。
        tol (float): 停止条件的容差。
        max_iterations (int): 允许的最大迭代次数。
    
"""
    # 存储迭代值以用于绘图
    x_values = [x0]
    iteration = 0

    while iteration < max_iterations:
        x_new = g_func(x0)
        x_values.append(x_new)

        # 检查收敛性
        if abs(x_new - x0) < tol:
            break

        x0 = x_new
        iteration += 1

    # 为平滑曲线绘图生成 x 值
    x = np.linspace(min(x_values) - 0.5, max(x_values) + 0.5, 400)
    y_gx = g_func(x)

    # 绘制不动点函数和步骤
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_gx, label="$g(x)$", linestyle="--", color="orange")
    plt.plot(x, x, label="$y = x$", color="gray")
    plt.scatter(
        x_values, [g_func(x) for x in x_values], color="red", label="Iteration Steps"
    )

    # 突出起点和终点
    plt.scatter(x_values[0], g_func(x_values[0]), color="green", label="Starting Point")
    plt.scatter(x_values[-1], g_func(x_values[-1]), color="purple", label="Final Point")

    plt.title("Fixed-Point Iteration Method: Progress of Steps")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例不动点迭代函数
def g(x):
    return (x ** 2 + 2) / 3


# 固定点迭代的初始猜测值
x0 = 0
fixed_point_iteration_visualize(g, x0)
