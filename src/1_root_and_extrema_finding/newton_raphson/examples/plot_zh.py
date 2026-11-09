# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """示例函数：f(x) = x^2 - 4"""
    return x ** 2 - 4


def df(x):
    """示例函数的导数：f'(x) = 2x"""
    return 2 * x


def newton_method_visualize(func, dfunc, x0, tol=1e-6, max_iterations=20):
    """在单个图表上可视化牛顿法求根的进度。
    
参数：
        func (function): 需要求根的目标函数。
        dfunc (function): 函数的导数。
        x0 (float): 初始起点。
        tol (float): 停止条件的容差。
        max_iterations (int): 允许的最大迭代次数。
    
"""
    # 存储用于绘图的数值
    x_values = [x0]
    y_values = [func(x0)]
    iteration = 0

    while iteration < max_iterations:
        f_x = func(x0)
        df_x = dfunc(x0)

        # 避免除以零
        if df_x == 0:
            print("Derivative is zero. Stopping iteration.")
            break

        # 牛顿更新
        x_new = x0 - f_x / df_x
        x_values.append(x_new)
        y_values.append(func(x_new))

        # 检查收敛性
        if abs(x_new - x0) < tol:
            break

        x0 = x_new
        iteration += 1

    # 为平滑曲线绘图生成 x 值
    x = np.linspace(min(x_values) - 1, max(x_values) + 1, 400)
    y = func(x)

    # 绘制函数图像和牛顿法步骤
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="$f(x) = x^2 - 4$", color="blue")
    plt.scatter(x_values, y_values, color="red", label="Newton's Method Steps")
    plt.axhline(0, color="black", linewidth=0.5)

    # 突出起点和终点
    plt.scatter(x_values[0], y_values[0], color="green", label="Starting Point")
    plt.scatter(x_values[-1], y_values[-1], color="purple", label="Final Point")

    plt.title("Newton's Method: Progress of Steps")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例函数及其导数
def f(x):
    return x ** 2 - 4


def df(x):
    return 2 * x


# 1. 牛顿法的初始条件
x0 = 3  # 2. 起始点
newton_method_visualize(f, df, x0)
