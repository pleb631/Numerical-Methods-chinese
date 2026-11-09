# 本文件为 example.py 的中文注释版本（代码逻辑与英文版完全一致）
"""拟合一个三次多项式以拟合给定数据点的曲线：

A(-9, -2)
B(-5, 3)
C(-2.5, 0)
D(4, 5)
F(7, 11)

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from implementation.polynomial_regression import polynomial_regression


# 定义数据
x = np.array([-9, -5, -2.5, 4, 7])
y = np.array([-2, 3, 0, 5, 11])

# 绘制原始数据
plt.scatter(x, y, s=200, marker="P", color="black", label="Raw data")
plt.xlabel("x")
plt.ylabel("y")

# 创建定义域
x_domain = np.linspace(1.1 * min(x), 1.1 * max(x), 100)

# 创建模型并显示 djeada 实现的预测结果
coefficients = polynomial_regression(x, y, 3)
y_prediction = sum(
    coefficient * x_domain**power
    for power, coefficient in enumerate(coefficients)
)
plt.plot(x_domain, y_prediction, label="Polynomial regression implementation")

# 创建模型并显示 numpy 实现的预测结果
model = np.poly1d(np.polyfit(x, y, 3))
y_prediction_numpy = model(x_domain)
plt.plot(x_domain, y_prediction_numpy, "--", label="Polynomial regression numpy")

# 添加图例并显示绘图
plt.legend(loc="best")
plt.show()
