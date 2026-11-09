# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt
from math import factorial


def taylor_series_approximation(f, df, a, x_range, n_terms):
    """计算函数给定项数的泰勒级数近似。
    
参数：
        f (callable): 待近似的函数。
        df (list of callable): 函数的导数列表。
        a (float): 展开点。
        x_range (numpy.ndarray): 用于绘图的 x 值范围。
        n_terms (int): 泰勒级数的项数。
    
返回：
        approx (numpy.ndarray): 函数的近似值。
    
"""
    approximations = np.zeros((n_terms, len(x_range)))
    for n in range(n_terms):
        approximations[n] = (df[n](a) / factorial(n)) * (x_range - a) ** n
    return approximations


def plot_taylor_series(f, df, a, x_range, n_terms):
    """绘制函数及其泰勒级数近似值。
    
参数：
        f (callable): 原始函数。
        df (list of callable): 函数的导数列表。
        a (float): 展开点。
        x_range (numpy.ndarray): 用于绘图的 x 值范围。
        n_terms (int): 泰勒级数的项数。
    
"""
    y_true = f(x_range)
    approximations = taylor_series_approximation(f, df, a, x_range, n_terms)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_range, y_true, label=f"Original function $f(x)$", color="black", linewidth=2
    )

    partial_sum = np.zeros_like(x_range)
    for n in range(n_terms):
        partial_sum += approximations[n]
        plt.plot(x_range, partial_sum, label=f"Taylor series (n={n})")

    plt.axvline(x=a, color="red", linestyle="--", label=f"Expansion point $a={a}$")
    plt.title("Taylor Series Approximation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 定义函数及其导数
    f = lambda x: np.exp(x)
    df = [
        lambda x: np.exp(x),  # f(x)
        lambda x: np.exp(x),  # f'(x)
        lambda x: np.exp(x),  # f''(x)
        lambda x: np.exp(x),  # f'''(x)
        lambda x: np.exp(x),  # f''''(x)
    ]  # 每个条目对应一个项（exp 的所有导数均为 exp）

    a = 0  # 展开点
    x_range = np.linspace(-2, 2, 400)  # x 值的范围
    n_terms = 5  # 泰勒级数的项数

    # 绘制泰勒级数近似
    plot_taylor_series(f, df, a, x_range, n_terms)
