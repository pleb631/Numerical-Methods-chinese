# 本文件为 example.py 的中文注释版本（代码逻辑与英文版完全一致）
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from implementation.taylor_series import taylor_series


def func(x):
    return np.sin(x)


sin_derivatives = [
    np.sin,
    np.cos,
    lambda x: -np.sin(x),
    lambda x: -np.cos(x),
]  # sin(x) 的导数以 4 为周期循环。


def derivatives(k):
    """展开点处 sin(x) 的 k 阶导数值。"""
    return np.array([sin_derivatives[int(order) % 4](point) for order in k])


point = 0
n = 5  # 项数。
x = np.pi / 4

taylor_expansion = taylor_series(func, point, n, x, derivatives)
print("Taylor series expansion:")
print(taylor_expansion)
