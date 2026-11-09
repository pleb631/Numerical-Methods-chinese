# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """示例函数：f(x) = x^2"""
    return x ** 2


def golden_ratio_search_visualize(func, a, b, tol=1e-6, max_iterations=100):
    """在单个图表上可视化黄金分割法（黄金比例搜索）用于函数最小化的初始和最终步骤。
    
参数：
        func (function): 要最小化的函数。
        a (float): 区间的初始下界。
        b (float): 区间的初始上界。
        tol (float): 停止条件的容差。
        max_iterations (int): 允许的最大迭代次数。
    
"""
    # 黄金分割常数
    phi = (1 + np.sqrt(5)) / 2  # ~1.618
    inv_phi = 1 / phi  # ~0.618

    # 存储初始 a 和 b 用于绘图
    a_initial, b_initial = a, b

    # 计算初始内部点
    x1 = b - (b - a) * inv_phi
    x2 = a + (b - a) * inv_phi
    f1, f2 = func(x1), func(x2)

    iteration = 0
    intervals = [(a, b, x1, x2)]  # 用于可视化存储

    while (b - a > tol) and (iteration < max_iterations):
        if f1 > f2:  # 最小值位于 [x1, b]
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * inv_phi
            f2 = func(x2)
        else:  # 最小值在 [a, x2] 中
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) * inv_phi
            f1 = func(x1)
        intervals.append((a, b, x1, x2))
        iteration += 1

    # 最终 a 和 b
    a_final, b_final = a, b

    # 生成用于绘图的 x 值
    x = np.linspace(a_initial - 1, b_initial + 1, 400)
    y = func(x)

    # 在同一图中绘制初始和最终区间
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="$f(x) = x^2$", color="blue")
    plt.axhline(0, color="black", linewidth=0.5)

    # 初始步骤
    plt.scatter(
        [a_initial, b_initial],
        [func(a_initial), func(b_initial)],
        color="red",
        label="Initial Brackets [a, b]",
    )
    plt.scatter(
        [intervals[0][2], intervals[0][3]],
        [func(intervals[0][2]), func(intervals[0][3])],
        color="green",
        label="Initial Points x1, x2",
    )

    # []
    plt.scatter(
        [a_final, b_final],
        [func(a_final), func(b_final)],
        color="purple",
        label="Final Brackets [a, b]",
    )
    plt.scatter(
        [x1, x2], [func(x1), func(x2)], color="orange", label="Final Points x1, x2"
    )

    plt.title("Golden Ratio Search: Initial and Final Steps")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# 黄金分割法的初始条件
a = -2
b = 2
golden_ratio_search_visualize(f, a, b)
