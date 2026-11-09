# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_integral(f, a, b, num_samples=1000000):
    """使用蒙特卡洛采样近似计算 f 从 a 到 b 的积分。
    
参数：
        f (callable): 待积分的函数。
        a (float): 下界。
        b (float): 上界。
        num_samples (int): 随机样本数量。
    
返回：
        float: 近似积分值。
    
"""
    samples = np.random.uniform(a, b, num_samples)
    evaluations = f(samples)
    return (b - a) * np.mean(evaluations)


if __name__ == "__main__":
    def f(x):
        return np.sin(x)

    a = 0
    b = np.pi
    num_samples = 1000

    np.random.seed(42)
    samples = np.random.uniform(a, b, num_samples)
    values = f(samples)

    result = monte_carlo_integral(f, a, b, num_samples=1000000)
    exact = 2.0
    print(f"Monte Carlo result: {result:.10f}")
    print(f"Exact value: {exact:.10f}")

    # 显示曲线下方采样点的图
    x = np.linspace(a, b, 500)
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "blue", linewidth=2, label="f(x) = sin(x)")
    ax.scatter(samples, values, s=1, color="red", alpha=0.3, label="Random samples")
    ax.fill_between(x, 0, y, alpha=0.1, color="blue")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Monte Carlo Integration Sampling")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
