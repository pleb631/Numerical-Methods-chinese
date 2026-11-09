# 本文件为 curve_fitting.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 定义真实函数
def f(x):
    return 2.5 * x + 1.3

# 1. 使用上述函数生成点
np.random.seed(0) # 2. 为了可复现性
n_points = 10
X = np.linspace(0, 1, n_points)
y = f(X) + np.random.normal(0, 0.3, size=n_points) # 3. 添加一些噪声

# 4. 将线性函数拟合到数据
def fit_func(x, a, b):
    return a * x + b

params, _ = curve_fit(fit_func, X, y)

# 5. 为最佳拟合线生成 y 值
y_fit = params[0] * X + params[1]

plt.figure(figsize=(8, 6))

# 6. 绘制数据点
plt.scatter(X, y, label='Data points')

# 7. 绘制最佳拟合线
plt.plot(X, y_fit, 'r-', label='Best fit: a=%5.3f, b=%5.3f' % tuple(params))

# 8. 绘制垂直线以显示误差
for xi, yi, y_fit_i in zip(X, y, y_fit):
    plt.plot([xi, xi], [yi, y_fit_i], color='gray', linestyle='--')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Curve fitting')
plt.legend()
plt.show()
