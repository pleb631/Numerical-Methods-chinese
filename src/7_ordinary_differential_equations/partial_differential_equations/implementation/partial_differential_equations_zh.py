# 本文件为 partial_differential_equations.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from typing import Callable, Tuple


def _time_grid(t_end: float, dt: float) -> np.ndarray:
    """时间网格为 0, dt, 2*dt, ...，最终精确结束于 t_end（最后一步缩短）。"""
    n_full = int(np.floor(t_end / dt + 1e-9))
    t = dt * np.arange(n_full + 1)
    if t_end - t[-1] > 1e-9 * dt:
        t = np.append(t, t_end)
    else:
        t[-1] = t_end
    return t


def heat_equation_explicit(
    u0: np.ndarray,
    x: np.ndarray,
    t_end: float,
    dt: float,
    alpha: float,
    bc_left: float = 0.0,
    bc_right: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """求解 u_t = alpha * u_xx，采用显式 FTCS 格式。

步长为 dt，但当 t_end / dt 不是整数时，最后一步会缩短，使得 t[-1] == t_end 精确成立。稳定性（alpha*dt/dx^2 <= 1/2）针对名义 dt 进行检查；较短的最终步同样稳定。狄利克雷边界 bc_left 和 bc_right。
时间一阶，空间二阶。

返回 (t, x, u)，其中 u[n, j] 近似 u(t[n], x[j])。



"""
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    if alpha <= 0:
        raise ValueError("Diffusion coefficient alpha must be positive.")
    if len(x) < 3:
        raise ValueError("Spatial grid must have at least 3 points.")
    if len(u0) != len(x):
        raise ValueError("Initial condition u0 must match the spatial grid size.")

    dx = x[1] - x[0]
    r = alpha * dt / dx**2

    if r > 0.5:
        raise ValueError(
            f"Stability condition violated: alpha*dt/dx^2 = {r} > 0.5. "
            "Reduce dt or increase dx."
        )

    t = _time_grid(t_end, dt)
    n_steps = len(t) - 1
    u = np.zeros((n_steps + 1, len(x)))
    u[0] = u0.copy()

    for n in range(n_steps):
        rn = alpha * (t[n + 1] - t[n]) / dx**2
        u[n + 1, 0] = bc_left
        u[n + 1, -1] = bc_right
        u[n + 1, 1:-1] = u[n, 1:-1] + rn * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])

    return t, x, u


def heat_equation_implicit(
    u0: np.ndarray,
    x: np.ndarray,
    t_end: float,
    dt: float,
    alpha: float,
    bc_left: float = 0.0,
    bc_right: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """求解 u_t = alpha * u_xx，采用隐式（向后欧拉）格式。

无条件稳定。步长为 dt，但当 t_end / dt 不是整数时，最后一步会缩短，使得 t[-1] == t_end 精确成立。
狄利克雷边界 bc_left 和 bc_right。时间一阶，空间二阶。

返回 (t, x, u)，其中 u[n, j] 近似 u(t[n], x[j])。


"""
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    if alpha <= 0:
        raise ValueError("Diffusion coefficient alpha must be positive.")
    if len(x) < 3:
        raise ValueError("Spatial grid must have at least 3 points.")
    if len(u0) != len(x):
        raise ValueError("Initial condition u0 must match the spatial grid size.")

    dx = x[1] - x[0]
    nx = len(x)

    t = _time_grid(t_end, dt)
    n_steps = len(t) - 1
    u = np.zeros((n_steps + 1, nx))
    u[0] = u0.copy()

    interior = nx - 2

    def build(r: float) -> np.ndarray:
        A = np.zeros((interior, interior))
        idx = np.arange(interior)
        A[idx, idx] = 1 + 2 * r
        A[idx[1:], idx[:-1]] = -r
        A[idx[:-1], idx[1:]] = -r
        return A

    cache: dict = {}
    for n in range(n_steps):
        h = t[n + 1] - t[n]
        key = round(h, 15)
        if key not in cache:
            cache[key] = build(alpha * h / dx**2)
        r = alpha * h / dx**2
        b = u[n, 1:-1].copy()
        b[0] += r * bc_left
        b[-1] += r * bc_right
        u[n + 1, 1:-1] = np.linalg.solve(cache[key], b)
        u[n + 1, 0] = bc_left
        u[n + 1, -1] = bc_right

    return t, x, u


def wave_equation_explicit(
    u0: np.ndarray,
    v0: np.ndarray,
    x: np.ndarray,
    t_end: float,
    dt: float,
    c: float,
    bc_left: float = 0.0,
    bc_right: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """求解 u_tt = c^2 * u_xx，采用显式 leapfrog 格式。

检查 CFL 条件 c*dt/dx <= 1 是否满足名义 dt。为了精确命中 t_end 同时保持均匀网格（leapfrog 需要恒定步长），使用 n = ceil(t_end / dt) 个大小为 dt_eff = t_end / n <= dt 的步长，使得 t[-1] == t_end 且 CFL 条件仍然成立。第一步使用初始速度 v0（泰勒展开）。时间二阶，空间二阶。狄利克雷边界 bc_left 和 bc_right。







"""
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    if c <= 0:
        raise ValueError("Wave speed c must be positive.")
    if len(x) < 3:
        raise ValueError("Spatial grid must have at least 3 points.")
    if len(u0) != len(x):
        raise ValueError("Initial condition u0 must match the spatial grid size.")
    if len(v0) != len(x):
        raise ValueError("Initial velocity v0 must match the spatial grid size.")

    dx = x[1] - x[0]
    r = c * dt / dx

    if r > 1.0:
        raise ValueError(
            f"CFL condition violated: c*dt/dx = {r} > 1.0. " "Reduce dt or increase dx."
        )

    n_steps = max(int(np.ceil(t_end / dt - 1e-9)), 1)
    t = np.linspace(0.0, t_end, n_steps + 1)
    dt = t_end / n_steps  # 返回 (t, x, u)，其中 u[n, j] 近似 u(t[n], x[j])。
    r2 = (c * dt / dx) ** 2
    u = np.zeros((n_steps + 1, len(x)))
    u[0] = u0.copy()

    u[1, 0] = bc_left
    u[1, -1] = bc_right
    for i in range(1, len(x) - 1):
        u[1, i] = (
            u[0, i] + dt * v0[i] + 0.5 * r2 * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1])
        )

    for n in range(1, n_steps):
        u[n + 1, 0] = bc_left
        u[n + 1, -1] = bc_right
        for i in range(1, len(x) - 1):
            u[n + 1, i] = (
                2 * u[n, i]
                - u[n - 1, i]
                + r2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
            )

    return t, x, u


def laplace_equation_2d(
    nx: int,
    ny: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bc: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nx < 3 or ny < 3:
        raise ValueError("Grid dimensions must be at least 3 in each direction.")
    if tol <= 0:
        raise ValueError("Tolerance must be positive.")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be positive.")

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = bc(X, Y).astype(float)

    # 加权五点模板；当 dx == dy 时退化为普通的 0.25 平均。
    wx = (y[1] - y[0]) ** 2
    wy = (x[1] - x[0]) ** 2
    denom = 2.0 * (wx + wy)

    for _ in range(max_iter):
        u_old = u.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = (
                    wx * (u[i + 1, j] + u[i - 1, j]) + wy * (u[i, j + 1] + u[i, j - 1])
                ) / denom

        if np.max(np.abs(u - u_old)) < tol:
            break

    return x, y, u
