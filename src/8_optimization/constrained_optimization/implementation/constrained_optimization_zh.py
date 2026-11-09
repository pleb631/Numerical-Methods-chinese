# 本文件为 constrained_optimization.py 的中文注释版本（代码逻辑与英文版完全一致）
# constrained_optimization.py
import numpy as np
from typing import Callable, Optional, Tuple, List


def _pivot(tableau: np.ndarray, basis: List[int], row: int, col: int) -> None:
    """对表在 (row, col) 处进行高斯 - 若尔当主元化，更新基。"""
    tableau[row] /= tableau[row, col]
    for i in range(tableau.shape[0]):
        if i != row:
            tableau[i] -= tableau[i, col] * tableau[row]
    basis[row] = col


def _simplex_iterations(
    tableau: np.ndarray,
    basis: List[int],
    allowed_cols: int,
    tol: float,
    max_iterations: int,
) -> bool:
    """对包含已对基进行定价的缩减成本（位于最后一行）和右端项（位于最后一列）的单纯形表执行原始单纯形迭代。仅允许前``allowed_cols``列进入。使用丹齐格规则，在发生一系列退化迭代后回退到布兰德规则以防止循环。返回：若达到最优基则返回 True，若问题在改进方向上无界则返回 False。异常：ValueError：若最大迭代次数的迭代不足。
    
    
    

    
    

    
    
    

    
    
"""
    m = tableau.shape[0] - 1
    degenerate_run = 0

    for _ in range(max_iterations):
        reduced_costs = tableau[-1, :allowed_cols]
        if degenerate_run >= 10:
            candidates = np.flatnonzero(reduced_costs < -tol)
            pivot_col = int(candidates[0]) if candidates.size else -1
        else:
            pivot_col = int(np.argmin(reduced_costs)) if allowed_cols else -1
            if pivot_col >= 0 and reduced_costs[pivot_col] >= -tol:
                pivot_col = -1
        if pivot_col < 0:
            return True

        column = tableau[:m, pivot_col]
        rhs = tableau[:m, -1]
        rows = np.flatnonzero(column > tol)
        if rows.size == 0:
            return False

        ratios = rhs[rows] / column[rows]
        best = ratios.min()
        ties = rows[ratios <= best + tol]
        pivot_row = int(min(ties, key=lambda r: basis[r]))

        degenerate_run = degenerate_run + 1 if best <= tol else 0
        _pivot(tableau, basis, pivot_row, pivot_col)

    raise ValueError(
        "Simplex method did not converge within the maximum number of iterations."
    )


def linear_programming(
    c: np.ndarray,
    A_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iterations: int = 1000,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """使用两阶段单纯形法求解线性规划问题。
    
最小化：
        c^T x
    
约束条件：
        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        x >= 0
    
形如 a^T x >= b 的约束条件通过 -a^T x <= -b 提供。具有负右侧值的行由第一阶段处理，该阶段通过最小化人工变量的总和来寻找基本可行解（或证明不存在这样的解）。
    
    参数：
        c: 成本向量 (n,)。
        A_ub: 不等式约束矩阵 (m x n)，可选。
    b_ub: 不等式约束向量 (m,)，可选。可为负值。
        tol: 最优性和枢轴检查的容差。
        max_iterations: 每个阶段的单纯形枢轴最大次数。
        A_eq: 等式约束矩阵 (p x n)，可选。
        b_eq: 等式约束向量 (p,)，可选。
    
    返回：
        (optimal_x, optimal_value) 元组。
    
异常：
        ValueError: 如果约束数据形状不一致。
        ValueError: 如果问题不可行。
    ValueError: 如果问题无界。
        ValueError: 如果方法未收敛。
    
    
    
    
"""
    c = np.asarray(c, dtype=float)
    n = len(c)

    if (A_ub is None) != (b_ub is None) or (A_eq is None) != (b_eq is None):
        raise ValueError("A_ub/b_ub and A_eq/b_eq must be given in pairs.")
    A_ub = (
        np.zeros((0, n))
        if A_ub is None
        else np.atleast_2d(np.asarray(A_ub, dtype=float))
    )
    b_ub = np.zeros(0) if b_ub is None else np.asarray(b_ub, dtype=float)
    A_eq = (
        np.zeros((0, n))
        if A_eq is None
        else np.atleast_2d(np.asarray(A_eq, dtype=float))
    )
    b_eq = np.zeros(0) if b_eq is None else np.asarray(b_eq, dtype=float)
    if A_ub.shape != (len(b_ub), n) or A_eq.shape != (len(b_eq), n):
        raise ValueError("Constraint matrices and vectors have inconsistent shapes.")

    m_ub, m_eq = len(b_ub), len(b_eq)
    m = m_ub + m_eq

    # 标准形式 [A_ub I; A_eq 0] [x; s] = [b_ub; b_eq], x, s >= 0
    A = np.zeros((m, n + m_ub))
    A[:m_ub, :n] = A_ub
    A[:m_ub, n:] = np.eye(m_ub)
    A[m_ub:, :n] = A_eq
    b = np.concatenate([b_ub, b_eq])

    # 使所有右端项非负
    negative = b < 0
    A[negative] *= -1.0
    b = np.where(negative, -b, b)

    # 松弛列为 +1 的行可以以该松弛变量作为初始基变量；
    # 所有其他行（翻转的不等式、等式）需要引入人工变量。
    n_struct = n + m_ub
    basis: List[int] = [-1] * m
    for i in range(m_ub):
        if not negative[i]:
            basis[i] = n + i
    artificial_rows = [i for i in range(m) if basis[i] < 0]
    n_art = len(artificial_rows)

    # 表头列：结构（x, s）、人工变量、右端项。最后一行：成本。
    tableau = np.zeros((m + 1, n_struct + n_art + 1))
    tableau[:m, :n_struct] = A
    tableau[:m, -1] = b
    for k, i in enumerate(artificial_rows):
        tableau[i, n_struct + k] = 1.0
        basis[i] = n_struct + k

    def price_out(cost: np.ndarray) -> None:
        tableau[-1, :] = 0.0
        tableau[-1, : len(cost)] = cost
        for i, b_idx in enumerate(basis):
            if tableau[-1, b_idx] != 0.0:
                tableau[-1] -= tableau[-1, b_idx] * tableau[i]

    if n_art:
        # 第一阶段：最小化人工变量的总和
        cost1 = np.zeros(n_struct + n_art)
        cost1[n_struct:] = 1.0
        price_out(cost1)
        _simplex_iterations(tableau, basis, n_struct + n_art, tol, max_iterations)

        feas_tol = 1e-8 * (1.0 + np.max(np.abs(b)))
        if -tableau[-1, -1] > feas_tol:
            raise ValueError("The linear program is infeasible.")

        # 1. 将剩余（零水平）人工变量逐出基；一行
        # 2. 没有可用结构列的行是冗余的并被丢弃。
        keep_rows = []
        for i in range(m):
            if basis[i] >= n_struct:
                usable = np.flatnonzero(np.abs(tableau[i, :n_struct]) > 1e-9)
                if usable.size == 0:
                    continue
                _pivot(tableau, basis, i, int(usable[0]))
            keep_rows.append(i)
        tableau = np.vstack([tableau[keep_rows], tableau[-1:]])
        basis = [basis[i] for i in keep_rows]
        tableau = np.delete(tableau, np.s_[n_struct : n_struct + n_art], axis=1)

    # 第二阶段：原始成本
    cost2 = np.zeros(n_struct)
    cost2[:n] = c
    price_out(cost2)
    if not _simplex_iterations(tableau, basis, n_struct, tol, max_iterations):
        raise ValueError("The linear program is unbounded.")

    x = np.zeros(n_struct)
    for i, b_idx in enumerate(basis):
        x[b_idx] = tableau[i, -1]
    x = x[:n]
    return x, float(c @ x)


def _lagrangian_hessian_fd(
    grad_f: Callable[[np.ndarray], np.ndarray],
    grad_h: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    """在固定 lam 下，L(x, lam) = f + lam^T h 的中心差分海森矩阵。"""
    n = len(x)
    H = np.zeros((n, n))
    for j in range(n):
        step = 1e-6 * max(1.0, abs(x[j]))
        e = np.zeros(n)
        e[j] = step
        gp = grad_f(x + e) + grad_h(x + e).T @ lam
        gm = grad_f(x - e) + grad_h(x - e).T @ lam
        H[:, j] = (gp - gm) / (2.0 * step)
    return 0.5 * (H + H.T)


def lagrange_multiplier_minimize(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    h: Callable[[np.ndarray], np.ndarray],
    grad_h: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iterations: int = 1000,
    hess_lagrangian: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """[
    "使用拉格朗日乘子法结合牛顿法求解等式约束优化问题（应用于 KKT 系统，即序列二次规划步骤）。",
    "牛顿步使用拉格朗日函数 L(x, lam) = f(x) + lam^T h(x) 的海塞矩阵，因此在正则解附近收敛为二次收敛。对 KKT 残差范数进行回溯线搜索以全局化迭代。牛顿法寻找 KKT 点；它无法区分极小值与其他驻点。",
    "乘子通过最小二乘法拟合 x0 处的平稳性条件进行初始化。",
"最小化：",
    "    f(x)",
    "约束条件：",
    "    h(x) = 0",
    "参数：",
    "    f：目标函数。",
    "    grad_f：目标函数的梯度（返回 n 维向量）。",
    "    h：等式约束函数（返回 p 维向量）。",
"    grad_h：h 的雅可比矩阵（返回 p x n 矩阵）。",
    "    x0：x 的初始猜测值（n 维向量）。",
    "    tol：收敛容差。",
"    max_iterations：最大迭代次数。",
    "    hess_lagrangian：可选函数 (x, lam) -> (n x n) 拉格朗日函数的海塞矩阵，即 hess f(x) + sum_i lam_i hess h_i(x)。如果省略，则通过固定 lam 下拉格朗日梯度 grad_f + J_h^T lam 的中心差分进行近似。",
    "返回：",
"    (optimal_x, lagrange_multipliers) 元组。",
    "异常：",
    "    ValueError：如果方法未收敛。"
    ]
    
    
    
    
    
    
    
    
    

    
    

    
    
"""
    x = x0.astype(float)
    n = len(x)
    p = len(h(x))
    # 最小二乘乘子估计：argmin ||grad_f + J_h^T lam||
    lam = np.linalg.lstsq(grad_h(x).reshape(p, n).T, -grad_f(x), rcond=None)[0]

    for iteration in range(max_iterations):
        gf = grad_f(x)
        hx = h(x)
        Jh = grad_h(x)

        # KKT 残差：[grad_f + J_h^T lambda; h(x)]
        residual = np.concatenate([gf + Jh.T @ lam, hx])
        if np.linalg.norm(residual) < tol:
            return x, lam

        # 构建 KKT 系统
        # [H   J_h^T] [dx  ]   [-grad_f - J_h^T lam]
        # [J_h   0  ] [dlam] = [-h(x)              ]
        if hess_lagrangian is not None:
            H = np.asarray(hess_lagrangian(x, lam), dtype=float)
        else:
            H = _lagrangian_hessian_fd(grad_f, grad_h, x, lam)
        KKT = np.block([[H, Jh.T], [Jh, np.zeros((p, p))]])
        rhs = -residual

        try:
            delta = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            raise ValueError(f"KKT system is singular at iteration {iteration}.")

        dx = delta[:n]
        dlam = delta[n:]

        # 带步长缩减的线搜索
        alpha = 1.0
        for _ in range(50):
            x_new = x + alpha * dx
            lam_new = lam + alpha * dlam
            new_residual = np.concatenate(
                [grad_f(x_new) + grad_h(x_new).T @ lam_new, h(x_new)]
            )
            if np.linalg.norm(new_residual) < np.linalg.norm(residual):
                break
            alpha *= 0.5

        x = x + alpha * dx
        lam = lam + alpha * dlam

    raise ValueError(
        "Lagrange multiplier method did not converge within the maximum number of iterations."
    )
