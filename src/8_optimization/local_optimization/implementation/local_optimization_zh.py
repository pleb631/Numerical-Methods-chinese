# 本文件为 local_optimization.py 的中文注释版本（代码逻辑与英文版完全一致）
# local_optimization.py
import numpy as np
from typing import Callable, Optional, Tuple


def gradient_descent_optimize(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    learning_rate: float = 0.01,
    tol: float = 1e-6,
    max_iterations: int = 10000,
) -> np.ndarray:
    """无约束最小化的梯度下降法。
    
参数:
        f: 目标函数。
        grad_f: 目标函数的梯度。
        x0: 初始点。
        learning_rate: 步长。
        tol: 梯度范数的收敛容差。
        max_iterations: 最大迭代次数。
    
返回:
        近似极小值点 x*。
    
异常:
        ValueError: 如果方法未收敛。
    
"""
    x = x0.astype(float)
    for _ in range(max_iterations):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            return x
        x = x - learning_rate * g
    raise ValueError(
        "Gradient descent did not converge within the maximum number of iterations."
    )


def newtons_method_optimize(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iterations: int = 100,
) -> np.ndarray:
    """无约束最小化的牛顿法。
    
利用精确的海森矩阵信息，在解附近实现二次收敛。
    
    参数:
    f: 目标函数。
        grad_f: 目标的梯度。
        hess_f: 目标的海森矩阵。
        x0: 初始点。
        tol: 梯度范数的收敛容差。
        max_iterations: 最大迭代次数。
    
    返回:
    近似极小值点 x*。
    
    异常:
    ValueError: 如果在迭代过程中海森矩阵奇异。
        ValueError: 如果方法未收敛。
    
    
"""
    x = x0.astype(float)
    for iteration in range(max_iterations):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            return x
        H = hess_f(x)
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            raise ValueError(f"Hessian is singular at iteration {iteration}, x = {x}.")
        x = x + d
    raise ValueError(
        "Newton's method did not converge within the maximum number of iterations."
    )


def _wolfe_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    d: np.ndarray,
    fx: float,
    slope: float,
    c1: float,
    c2: float,
    alpha_init: float = 1.0,
    alpha_max: float = 1e6,
    max_iterations: int = 50,
) -> Optional[Tuple[float, float, np.ndarray]]:
    """用于满足强沃尔夫条件的线搜索（括弧阶段后接二分法缩放，参见 Nocedal & Wright，算法 3.5 和 3.6）。
    
        f(x + a d) <= f(x) + c1 a g^T d          (充分下降)
        |grad f(x + a d)^T d| <= c2 |g^T d|      (曲率)

    参数:
        f, grad_f: 目标函数及其梯度。
    x: 当前点。
        d: 下降方向。
        fx: f(x)。
        slope: 方向导数 g^T d（在 x 处必须为负）。
        c1, c2: 沃尔夫常数，满足 0 < c1 < c2 < 1。
        alpha_init: 首次试验步长。
        alpha_max: 步长的上界。
        max_iterations: 每个阶段的最大试验步数。
    
    返回:
        (alpha, f(x + alpha d), grad f(x + alpha d)) 或如果未找到可接受的步长则为 None。

    
    
    
"""

    def phi(a):
        return f(x + a * d)

    def dphi(a):
        return grad_f(x + a * d) @ d

    def zoom(lo, hi, phi_lo):
        for _ in range(max_iterations):
            a = 0.5 * (lo + hi)
            phi_a = phi(a)
            if phi_a > fx + c1 * a * slope or phi_a >= phi_lo:
                hi = a
            else:
                dphi_a = dphi(a)
                if abs(dphi_a) <= -c2 * slope:
                    return a
                if dphi_a * (hi - lo) >= 0:
                    hi = lo
                lo, phi_lo = a, phi_a
        return None

    alpha_prev, phi_prev = 0.0, fx
    alpha = alpha_init
    for i in range(max_iterations):
        phi_a = phi(alpha)
        if phi_a > fx + c1 * alpha * slope or (i > 0 and phi_a >= phi_prev):
            found = zoom(alpha_prev, alpha, phi_prev)
            break
        dphi_a = dphi(alpha)
        if abs(dphi_a) <= -c2 * slope:
            found = alpha
            break
        if dphi_a >= 0:
            found = zoom(alpha, alpha_prev, phi_a)
            break
        if alpha >= alpha_max:
            found = None
            break
        alpha_prev, phi_prev = alpha, phi_a
        alpha = min(2.0 * alpha, alpha_max)
    else:
        found = None

    if found is None:
        return None
    x_new = x + found * d
    return found, f(x_new), grad_f(x_new)


def bfgs(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iterations: int = 1000,
    c1: float = 1e-4,
    c2: float = 0.9,
    callback: Optional[Callable[[np.ndarray], None]] = None,
) -> np.ndarray:
    """无约束最小化的 BFGS 拟牛顿法。
    
利用梯度差构建逆海森矩阵的近似，在不计算精确海森矩阵的情况下实现超线性收敛。步长满足强沃尔夫条件，这保证了 y^T s > 0，从而确保更新保持近似矩阵正定。
    
    参数:
        f: 目标函数。
        grad_f: 目标的梯度。
        x0: 初始点。
    tol: 梯度范数的收敛容差。
        max_iterations: 最大迭代次数。
        c1: 沃尔夫条件的充分下降参数。
        c2: 沃尔夫条件的曲率条件参数
            (0 < c1 < c2 < 1)。
        callback: 可选函数，在每次迭代后调用。
    
    返回:
        近似极小值点 x*。
    
    异常:
    ValueError: 如果 c1 和 c2 不满足 0 < c1 < c2 < 1。
        ValueError: 如果沿最速下降步的线搜索失败。
        ValueError: 如果方法未收敛。

    
    
    
    
"""
    if not 0.0 < c1 < c2 < 1.0:
        raise ValueError("Wolfe constants must satisfy 0 < c1 < c2 < 1.")

    x = x0.astype(float)
    n = len(x)
    I = np.eye(n)
    H = I.copy()
    fx = f(x)
    g = grad_f(x)

    for _ in range(max_iterations):
        if np.linalg.norm(g) < tol:
            return x

        d = -H @ g
        slope = g @ d
        if slope >= 0:
            # H 数值上丢失正定性：重启。
            H = I.copy()
            d = -g
            slope = g @ d

        result = _wolfe_line_search(f, grad_f, x, d, fx, slope, c1, c2)
        if result is None and not np.array_equal(H, I):
            # 沿最速下降法重试，并使用新的海森矩阵模型。
            H = I.copy()
            d = -g
            slope = g @ d
            result = _wolfe_line_search(f, grad_f, x, d, fx, slope, c1, c2)
        if result is None:
            raise ValueError(
                "BFGS line search failed to find a step satisfying the Wolfe conditions."
            )
        alpha, fx, g_new = result

        s = alpha * d
        y = g_new - g
        x = x + s

        ys = y @ s
        if ys > 1e-10:
            rho = 1.0 / ys
            V1 = I - rho * np.outer(s, y)
            V2 = I - rho * np.outer(y, s)
            H = V1 @ H @ V2 + rho * np.outer(s, s)

        g = g_new
        if callback is not None:
            callback(x)

    raise ValueError("BFGS did not converge within the maximum number of iterations.")
