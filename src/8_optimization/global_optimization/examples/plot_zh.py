# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt


def simulated_annealing(f, bounds, x0=None, T0=10.0, cooling_rate=0.95,
                        sigma=0.5, max_iterations=10000, seed=None):
    """使用模拟退火最小化函数。
    
参数：
        f (callable): 目标函数。
        bounds (list): 每个维度的 (min, max) 边界列表。
        x0 (numpy.ndarray): 初始点。
        T0 (float): 初始温度。
        cooling_rate (float): 每次迭代的温度冷却率。
        sigma (float): 随机扰动的标准差。
        max_iterations (int): 最大迭代次数。
        seed (int): 用于可重复性的随机种子。
    
返回：
        best_x (numpy.ndarray): 找到的最佳解。
        best_f (float): 找到的最佳目标值。
        history (list): 每次迭代的 (x, f_val) 列表。
    
"""
    rng = np.random.default_rng(seed)
    dim = len(bounds)

    if x0 is None:
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])

    x = x0.copy()
    f_x = f(x)
    best_x = x.copy()
    best_f = f_x
    history = [(x.copy(), f_x)]

    T = T0
    for _ in range(max_iterations):
        # 扰动并裁剪至边界
        x_new = x + rng.normal(0, sigma, size=dim)
        for d in range(dim):
            x_new[d] = np.clip(x_new[d], bounds[d][0], bounds[d][1])

        f_new = f(x_new)
        delta = f_new - f_x

        MIN_TEMPERATURE = 1e-300
        if delta < 0 or rng.random() < np.exp(-delta / max(T, MIN_TEMPERATURE)):
            x = x_new
            f_x = f_new
            if f_x < best_f:
                best_x = x.copy()
                best_f = f_x

        history.append((x.copy(), f_x))
        T *= cooling_rate

    return best_x, best_f, history


if __name__ == "__main__":
    # Rastrigin 函数（多模态）
    def rastrigin(x):
        A = 10
        return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    x0 = np.array([4.0, 4.0])
    best_x, best_f, history = simulated_annealing(
        rastrigin, bounds, x0=x0, T0=20.0, cooling_rate=0.999,
        sigma=0.3, max_iterations=20000, seed=42,
    )

    print(f"Best solution: ({best_x[0]:.6f}, {best_x[1]:.6f})")
    print(f"Best value:    {best_f:.6f}")

    # 绘制 Rastrigin 曲面及 SA 路径
    x_range = np.linspace(-5.12, 5.12, 300)
    y_range = np.linspace(-5.12, 5.12, 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 20 + X ** 2 + Y ** 2 - 10 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))

    fig, ax = plt.subplots(figsize=(9, 7))
    contour = ax.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="$f(x, y)$")

    # 绘制 SA 路径（为清晰起见进行子采样）
    positions = np.array([h[0] for h in history])
    step = max(1, len(positions) // 500)
    ax.plot(
        positions[::step, 0], positions[::step, 1],
        "w.-", markersize=1, linewidth=0.5, alpha=0.6, label="SA path",
    )
    ax.plot(x0[0], x0[1], "wo", markersize=10, label="Start")
    ax.plot(best_x[0], best_x[1], "r*", markersize=18, zorder=5, label="Best found")

    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_title("Simulated Annealing on Rastrigin Function", fontsize=16)
    ax.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    plt.show()
