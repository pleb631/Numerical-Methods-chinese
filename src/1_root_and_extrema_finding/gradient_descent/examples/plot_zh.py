# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """示例函数：f(x) = x^2"""
    return x ** 2


def gradient_descent_visualize(
    func, grad, x0, learning_rate=0.1, tol=1e-6, max_iterations=100
):
    """在单个图表中可视化梯度下降用于函数最小化的进度。
    
参数：
        func (function): 要最小化的函数。
        grad (function): 函数的梯度。
        x0 (float): 初始起点。
        learning_rate (float): 步长。
        tol (float): 停止条件的容差。
        max_iterations (int): 允许的最大迭代次数。
    
"""
    # 存储用于绘图的数值
    x_values = [x0]
    y_values = [func(x0)]
    iteration = 0

    while iteration < max_iterations:
        gradient = grad(x0)
        x_new = x0 - learning_rate * gradient
        x_values.append(x_new)
        y_values.append(func(x_new))

        # 检查收敛性
        if abs(x_new - x0) < tol:
            break

        x0 = x_new
        iteration += 1

    # 生成用于平滑曲线绘图的 x 值
    x = np.linspace(min(x_values) - 1, max(x_values) + 1, 400)
    y = func(x)

    # 绘制函数和梯度下降步骤
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="$f(x) = x^2$", color="blue")
    plt.scatter(x_values, y_values, color="red", label="Gradient Descent Steps")
    plt.axhline(0, color="black", linewidth=0.5)

    # 突出显示起点和终点
    plt.scatter(x_values[0], y_values[0], color="green", label="Starting Point")
    plt.scatter(x_values[-1], y_values[-1], color="purple", label="Final Point")

    plt.title("Gradient Descent: Progress of Steps")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例函数和梯度
def f(x):
    return x ** 2


def grad_f(x):
    return 2 * x


# 梯度下降的初始条件
x0 = 5  # 起点
learning_rate = 0.1
gradient_descent_visualize(f, grad_f, x0, learning_rate)
