# 本文件为 plot.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def linear_programming(c, A_ub, b_ub):
    """求解二维线性规划问题：最小化 c^T x，约束条件为 A_ub @ x <= b_ub 且 x >= 0。使用顶点枚举法解决二维问题。
    
    

    
    
    
    

    
    
    
"""
    m = A_ub.shape[0]
    # 构建所有约束线：A_ub @ x = b_ub 且 x_i = 0
    A_all = np.vstack([A_ub, np.eye(2)])
    b_all = np.concatenate([b_ub, np.zeros(2)])
    n_lines = len(b_all)

    vertices = []
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            A_pair = np.array([A_all[i], A_all[j]])
            b_pair = np.array([b_all[i], b_all[j]])
            try:
                v = np.linalg.solve(A_pair, b_pair)
            except np.linalg.LinAlgError:
                continue
            # 检查可行性
            if np.all(A_ub @ v <= b_ub + 1e-10) and np.all(v >= -1e-10):
                vertices.append(v)

    if not vertices:
        raise ValueError("No feasible region found.")

    vertices = np.array(vertices)
    objectives = vertices @ c
    idx = np.argmin(objectives)
    return vertices[idx], objectives[idx]


if __name__ == "__main__":
    # 1. 在约束条件 A_ub @ x <= b_ub 且 x >= 0 下最小化 c^T x
    c = np.array([-3, -5])  # 2. 最小化 -3x - 5y（即最大化 3x + 5y）
    A_ub = np.array([[1, 0], [0, 1], [1, 2]])
    b_ub = np.array([4, 6, 8])

    x_opt, obj = linear_programming(c, A_ub, b_ub)
    print(f"Optimal point: ({x_opt[0]:.4f}, {x_opt[1]:.4f})")
    print(f"Optimal value: {obj:.4f}")

    fig, ax = plt.subplots(figsize=(8, 7))

    # 绘制约束边界
    x_range = np.linspace(-0.5, 6, 400)
    ax.plot(x_range, np.full_like(x_range, 6), "r-", label="$y \\leq 6$")
    ax.axvline(x=4, color="g", linestyle="-", label="$x \\leq 4$")
    ax.plot(x_range, (8 - x_range) / 2, "b-", label="$x + 2y \\leq 8$")

    # 着色可行区域
    corners = [[0, 0], [4, 0], [4, 2], [0, 4]]
    polygon = Polygon(corners, alpha=0.2, color="skyblue", label="Feasible region")
    ax.add_patch(polygon)

    # 绘制最优解点
    ax.plot(x_opt[0], x_opt[1], "r*", markersize=20, zorder=5, label="Optimal point")

    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 7)
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_title("Constrained Optimization: Linear Programming", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
