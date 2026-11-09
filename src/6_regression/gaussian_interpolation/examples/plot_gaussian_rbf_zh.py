# 本文件为 plot_gaussian_rbf.py 的中文注释版本（代码逻辑与英文版完全一致）
"""绘制多个形状参数的高斯径向基函数插值。直接运行此文件以比较曲线及其插值节点。高斯中心差分多项式示例保留在 ``plot.py`` 中。



"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.gaussian_rbf import GaussianRBFInterpolator


def main():
    nodes = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 0.5, 0.0])
    queries = np.linspace(-0.5, 2.5, 400)

    for epsilon in (0.5, 1.0, 2.0):
        interpolator = GaussianRBFInterpolator(nodes, values, epsilon)
        plt.plot(queries, interpolator(queries), label=f"epsilon = {epsilon:g}")

    plt.plot(nodes, values, "ko", label="Data nodes")
    plt.xlabel("x")
    plt.ylabel("s(x)")
    plt.title("Gaussian radial basis function interpolation")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
