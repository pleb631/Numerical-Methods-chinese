# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt

# 生成更多示例数据：输入 (X) 和带有噪声的输出 (Y)
np.random.seed(42)
x = np.linspace(1, 10, 20)  # 20 个均匀分布的特征值
y = 0.5 * x + 2 + np.random.normal(0, 0.5, size=len(x))  # 带噪声的线性关系

# 步骤 1：为 X 添加截距项（全 1 列）
X = np.vstack([np.ones(len(x)), x]).T  # 设计矩阵

# 步骤 2：计算正规方程：beta = (X^T X)^(-1) X^T Y
XT_X = X.T @ X  # X 的转置乘以 X
XT_Y = X.T @ y  # X 的转置乘以 Y
beta = np.linalg.inv(XT_X) @ XT_Y  # 求解 beta

# 步骤 3：定义回归线
x_fit = np.linspace(min(x), max(x), 100)  # 绘图平滑范围
y_fit = beta[0] + beta[1] * x_fit  # 拟合线

# 步骤 4：可视化
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="red", label="Data Points")  # 原始数据点
plt.plot(
    x_fit, y_fit, color="blue", label=f"Fitted Line: y = {beta[0]:.2f} + {beta[1]:.2f}x"
)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Least Squares Regression - Fitted Line")
plt.legend()
plt.grid(True)
plt.show()

# 显示回归系数
print(f"Regression Coefficients: Intercept = {beta[0]:.2f}, Slope = {beta[1]:.2f}")
