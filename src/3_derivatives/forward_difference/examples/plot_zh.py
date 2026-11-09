# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def forward_difference(f, x, h=1e-5):
    """使用前向差分法近似计算 f 在 x 处的导数。
    

    
    
    
    

    
    
"""
    return (f(x + h) - f(x)) / h


if __name__ == "__main__":
    # 定义函数及其精确导数
    def f(x):
        return x ** 3

    def df(x):
        return 3 * x ** 2

    # 定义域和评估点
    x_vals = np.linspace(0, 1, 1000)
    x0 = 0.3
    x1 = x0 + 0.5  # 前向点

    # 绘制函数图像
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, f(x_vals), label="$f(x) = x^3$")

    # 绘制前向差分割线
    plt.plot([x0, x1], [f(x0), f(x1)], "r", linewidth=2, label="Forward difference")

    # 绘制精确导数切线
    h = 0.2
    plt.plot(
        [x0 - h, x0 + h],
        [f(x0) - h * df(x0), f(x0) + h * df(x0)],
        "g--",
        linewidth=2,
        label="Exact derivative",
    )

    # 添加坐标轴标签和刻度
    plt.xticks((x0, x1), labels=("$x_0$", "$x_0 + \\Delta x$"))
    plt.plot([x0, x0], [-0.1, f(x0)], "g:")
    plt.plot([x1, x1], [-0.1, f(x1)], "g:")

    plt.yticks((f(x0), f(x1)), labels=("$f(x_0)$", "$f(x_0 + \\Delta x)$"))
    plt.plot([-0.1, x0], [f(x0), f(x0)], "g:")
    plt.plot([-0.1, x1], [f(x1), f(x1)], "g:")

    plt.title("Forward Difference Method", fontsize=16)
    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$f(x) = x^3$", fontsize=16)
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()
