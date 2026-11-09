# 本文件为 convex_optimization.py 的中文注释版本（代码逻辑与英文版完全一致）
# convex_optimization.py
import numpy as np
from typing import Callable, Optional


def _project_feasible(
    y: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    tol: float,
    max_sweeps: int = 5000,
) -> np.ndarray:
    """使用 Dykstra 算法计算 y 到集合 {A_ub x <= b_ub, A_eq x = b_eq} 的欧几里得投影。
    
    仅对各个集合进行循环投影只能找到交集的*一个*点，而非最近的点，因此需要 Dykstra 方法的修正项来进行精确投影。

    
    
    
"""
    sets = []
    if A_ub is not None and b_ub is not None:
        for a_i, b_i in zip(A_ub, b_ub):
            sets.append(("ub", a_i, b_i))
    if A_eq is not None and b_eq is not None:
        sets.append(("eq", A_eq, b_eq))

    x = y.copy()
    increments = [np.zeros_like(y) for _ in sets]

    for _ in range(max_sweeps):
        x_prev = x.copy()
        for k, (kind, a, b) in enumerate(sets):
            z = x + increments[k]
            if kind == "ub":
                violation = a @ z - b
                x = z - (violation / (a @ a)) * a if violation > 0 else z
            else:
                x = z - a.T @ np.linalg.solve(a @ a.T, a @ z - b)
            increments[k] = z - x

        feasible = True
        if A_ub is not None and b_ub is not None:
            feasible = bool(np.all(A_ub @ x - b_ub <= tol))
        if feasible and A_eq is not None and b_eq is not None:
            feasible = bool(np.linalg.norm(A_eq @ x - b_eq) <= tol)
        if feasible and np.linalg.norm(x - x_prev) < tol:
            break

    return x


def quadratic_programming(
    Q: np.ndarray,
    c: np.ndarray,
    A_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_iterations: int = 1000,
) -> np.ndarray:
    """使用投影梯度下降法求解凸二次规划问题。
    
每个梯度步骤后都跟随一次到可行多面体（使用 Dykstra 算法计算）的精确欧几里得投影。
    
    最小化：
    0.5 * x^T Q x + c^T x
    
    约束条件：
    A_ub @ x <= b_ub  (不等式约束)
        A_eq @ x == b_eq  (等式约束)
    
    参数：
    Q: 对称半正定矩阵 (n x n)。
        c: 线性成本向量 (n,)。
        A_ub: 不等式约束矩阵 (m x n)，可选。
        b_ub: 不等式约束向量 (m,)，可选。
        A_eq: 等式约束矩阵 (p x n)，可选。
        b_eq: 等式约束向量 (p,)，可选。
        x0: 初始可行点 (n,)，可选。默认为零。
        tol: 收敛容差。
        max_iterations: 最大迭代次数。
    
    返回：
    最优解 x*。
    
    异常：
    ValueError: 如果 Q 不是半正定。
        ValueError: 如果 Q 为 (数值上) 零，即问题为线性规划；请使用线性规划求解器。
        ValueError: 如果方法未收敛。
    
    
    
"""
    n = Q.shape[0]
    eigenvalues = np.linalg.eigvalsh(Q)
    if np.any(eigenvalues < -tol):
        raise ValueError("Q must be positive semidefinite for convex QP.")

    if x0 is None:
        x0 = np.zeros(n)
    x = x0.astype(float)

    # L = ||Q||_2 = lambda_max(Q) 是梯度的 Lipschitz 常数，
    # 因此步长 1/L 保证投影梯度单调递减。
    L = float(np.max(eigenvalues))
    if L <= tol:
        raise ValueError(
            "Q is numerically zero, so this is a linear program with no "
            "curvature to set a step size; use a linear programming solver."
        )
    alpha = 1.0 / L

    for iteration in range(max_iterations):
        grad = Q @ x + c
        x_new = x - alpha * grad

        if (A_ub is not None and b_ub is not None) or (
            A_eq is not None and b_eq is not None
        ):
            x_new = _project_feasible(x_new, A_ub, b_ub, A_eq, b_eq, tol)

        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new

    raise ValueError(
        "Quadratic programming did not converge within the maximum number of iterations."
    )


def is_convex_function(
    f: Callable[[np.ndarray], float],
    domain_points: np.ndarray,
    num_tests: int = 100,
    tol: float = 1e-6,
) -> bool:
    """数值测试函数在一系列样本点上是否凸。
    
对于凸性，f(theta*x + (1-theta)*y) <= theta*f(x) + (1-theta)*f(y)
    对所有 x, y 在定义域内且 theta 在 [0,1] 内成立。
    
参数：
        f: 待测试的函数。
        domain_points: 定义域内的点数组 (k x n)。
        num_tests: 执行的随机凸性测试次数。
        tol: 凸性不等式的容差。
    
返回：
        如果所有测试通过则为 True，否则为 False。
    
"""
    rng = np.random.default_rng(42)
    k = domain_points.shape[0]
    for _ in range(num_tests):
        i, j = rng.choice(k, size=2, replace=False)
        x, y = domain_points[i], domain_points[j]
        theta = rng.uniform(0, 1)
        lhs = f(theta * x + (1 - theta) * y)
        rhs = theta * f(x) + (1 - theta) * f(y)
        if lhs > rhs + tol:
            return False
    return True
