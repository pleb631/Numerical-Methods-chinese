# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def simpson_rule(f, a, b, n):
    """使用辛普森法近似计算 f 从 a 到 b 的积分。
    
参数：
        f (callable): 待积分的函数。
        a (float): 下界。
        b (float): 上界。
        n (int): 子区间数量（必须为偶数）。
    
返回：
        float: 近似积分值。
    
"""
    if n <= 0 or n % 2 != 0:
        raise ValueError("Number of intervals must be a positive even integer.")
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    h = (b - a) / n
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


if __name__ == "__main__":
    def f(x):
        return x * np.sin(x)

    a = 0
    b = np.pi
    n = 10  # 必须是偶数

    result = simpson_rule(f, a, b, n)
    exact = np.pi
    print(f"Simpson's rule result: {result:.10f}")
    print(f"Exact value: {exact:.10f}")

    # 绘制函数和抛物线段
    x = np.linspace(a - 0.5, b + 0.5, 500)
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "blue", linewidth=2, label="f(x) = x*sin(x)")
    ax.set_ylim(bottom=0)

    h = (b - a) / n
    nodes = np.linspace(a, b, n + 1)
    for i in range(0, n, 2):
        x0, x1, x2 = nodes[i], nodes[i + 1], nodes[i + 2]
        xs = np.linspace(x0, x2, 100)
        # 通过三个点的拉格朗日插值
        y0, y1, y2 = f(x0), f(x1), f(x2)
        ys = (
            y0 * (xs - x1) * (xs - x2) / ((x0 - x1) * (x0 - x2))
            + y1 * (xs - x0) * (xs - x2) / ((x1 - x0) * (x1 - x2))
            + y2 * (xs - x0) * (xs - x1) / ((x2 - x0) * (x2 - x1))
        )
        ax.fill_between(xs, 0, ys, alpha=0.3, color="orange", edgecolor="black")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Simpson's Rule")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
