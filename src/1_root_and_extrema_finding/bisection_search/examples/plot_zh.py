# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """示例函数：f(x) = x^2 - 4"""
    return x ** 2 - 4


def bisection_method_visualize(func, a, b, tol=1e-6, max_iterations=100):
    """在单个图表上可视化二分法求根法的初始和最终步骤。
    
参数：
        func (function): 用于求根的函数。
        a (float): 区间的初始下界。
        b (float): 区间的初始上界。
        tol (float): 停止条件的容差。
        max_iterations (int): 允许的最大迭代次数。
    
"""
    # 存储初始 a 和 b 用于绘图
    a_initial, b_initial = a, b

    # 用于可视化的区间列表
    intervals = [(a, b)]
    iteration = 0

    while (b - a > tol) and (iteration < max_iterations):
        c = (a + b) / 2.0  # 中点
        if func(a) * func(c) < 0:
            b = c  # 根位于 [a, c]
        else:
            a = c  # 根位于 [c, b]
        intervals.append((a, b))
        iteration += 1

    # 最终 a 和 b
    a_final, b_final = a, b

    # 生成用于绘图的 x 值
    x = np.linspace(a_initial - 1, b_initial + 1, 400)
    y = func(x)

    # 在同一图表上绘制初始和最终区间
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="$f(x) = x^2 - 4$", color="blue")
    plt.axhline(0, color="black", linewidth=0.5)

    # 初始步骤
    plt.scatter(
        [a_initial, b_initial],
        [func(a_initial), func(b_initial)],
        color="red",
        label="Initial Brackets [a, b]",
    )
    c_initial = (a_initial + b_initial) / 2.0
    plt.scatter(c_initial, func(c_initial), color="green", label="Initial Midpoint c")

    # []
    plt.scatter(
        [a_final, b_final],
        [func(a_final), func(b_final)],
        color="purple",
        label="Final Brackets [a, b]",
    )
    c_final = (a_final + b_final) / 2.0
    plt.scatter(c_final, func(c_final), color="orange", label="Final Midpoint c")

    plt.title("Bisection Method: Initial and Final Steps")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# 二分法初始条件
a = 0
b = 5
bisection_method_visualize(f, a, b)
