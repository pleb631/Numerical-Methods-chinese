# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """示例函数：f(x) = x^2 - 4"""
    return x ** 2 - 4


def secant_method_visualize(f_func, x0, x1, tol=1e-6, max_iterations=50):
    """可视化割线法求根的进度。
    
参数：
        f_func (function): 需要求根的函数。
        x0 (float): 第一个初始猜测值。
        x1 (float): 第二个初始猜测值。
        tol (float): 停止条件的容差。
        max_iterations (int): 允许的最大迭代次数。
    
"""
    # 存储迭代值以进行绘图
    x_values = [x0, x1]
    y_values = [f_func(x0), f_func(x1)]
    iteration = 0

    while iteration < max_iterations:
        f_x0 = f_func(x0)
        f_x1 = f_func(x1)

        # 避免除以零
        if f_x1 - f_x0 == 0:
            print("Division by zero encountered. Stopping iteration.")
            break

        # 割线法更新
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        x_values.append(x_new)
        y_values.append(f_func(x_new))

        # 检查收敛性
        if abs(x_new - x1) < tol or abs(f_func(x_new)) < tol:
            break

        # 更新点
        x0, x1 = x1, x_new
        iteration += 1

    # 生成用于绘制平滑曲线的 x 值
    x = np.linspace(min(x_values) - 1, max(x_values) + 1, 400)
    y = f_func(x)

    # 绘制函数和割线法步骤
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="$f(x) = x^2 - 4$", color="blue")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.scatter(x_values, y_values, color="red", label="Secant Method Steps")

    # 突出显示起始点和终点
    plt.scatter(x_values[0], y_values[0], color="green", label="Starting Point")
    plt.scatter(x_values[-1], y_values[-1], color="purple", label="Final Point")

    plt.title("Secant Method: Progress of Steps")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# 割线法的示例函数
def f(x):
    return x ** 2 - 4


# 割线法的初始猜测值
x0 = 0
x1 = 1
secant_method_visualize(f, x0, x1)
