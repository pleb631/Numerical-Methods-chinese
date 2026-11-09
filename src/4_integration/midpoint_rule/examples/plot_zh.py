# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def midpoint_rule(f, a, b, n):
    """使用**中点法**近似计算 f 从 a 到 b 的积分。
    

    
    
    
    
    

    
    
"""
    h = (b - a) / n
    midpoints = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(midpoints))


if __name__ == "__main__":
    def f(x):
        return x * np.sin(x)

    a = 0
    b = np.pi
    n = 20

    result = midpoint_rule(f, a, b, n)
    print(f"Midpoint rule result: {result:.10f}")
    print(f"Exact value: {np.pi:.10f}")

    # 绘制函数及中点矩形
    x = np.linspace(a - 0.5, b + 0.5, 500)
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "blue", linewidth=2)
    ax.set_ylim(bottom=0)

    dx = (b - a) / n
    for i in range(n):
        xi = a + i * dx
        mid = xi + dx / 2
        xs = [xi, xi, xi + dx, xi + dx]
        ys = [0, f(mid), f(mid), 0]
        ax.fill(xs, ys, "lightgray", edgecolor="black")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Midpoint Rule")
    ax.legend(["f(x) = x*sin(x)"])
    ax.grid(True)
    plt.tight_layout()
    plt.show()
