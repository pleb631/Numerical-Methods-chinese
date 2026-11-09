# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def divided_differences(x, y):
    """计算牛顿插值的差商表。
    

    
    
    

    
    
"""
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table[0]


def newton_polynomial(x, coefficients, x_data):
    """在给定点处评估牛顿插值多项式。
    

    
    
    
    

    
    
"""
    n = len(coefficients)
    result = coefficients[0]
    product_term = 1.0

    for i in range(1, n):
        product_term *= x - x_data[i - 1]
        result += coefficients[i] * product_term

    return result


def plot_newton_interpolation(x_data, y_data):
    """绘制牛顿插值多项式与给定数据点。
    

    
    
    
"""
    # 计算差商。
    coefficients = divided_differences(x_data, y_data)

    # 生成用于平滑曲线的精细 x 值。
    x_fine = np.linspace(min(x_data), max(x_data), 500)
    y_fine = newton_polynomial(x_fine, coefficients, x_data)

    # 绘制数据点和插值多项式。
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, color="red", label="Data Points")
    plt.plot(x_fine, y_fine, label="Newton's Polynomial", color="blue")
    plt.title("Newton's Interpolation Polynomial")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


# 示例数据点。
x_data = np.array([1, 2, 3])
y_data = np.array([2, 3, 5])

# 绘制牛顿插值。
plot_newton_interpolation(x_data, y_data)
