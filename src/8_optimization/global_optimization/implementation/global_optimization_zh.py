# 本文件为 global_optimization.py 的中文注释版本（代码逻辑与英文版完全一致）
# global_optimization.py
import numpy as np
from typing import Callable, Optional, Tuple, List


def grid_search(
    f: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    n_points: int = 50,
) -> Tuple[np.ndarray, float]:
    """暴力网格搜索用于全局优化。
    在给定的边界范围内，在规则网格上的每一点评估 f，并返回函数值最小的点。
参数：
        f: 将 n 维向量映射为标量的目标函数。
        bounds: 每个维度的 (下界，上界) 边界列表。
    n_points: 每个维度的网格点数。
    返回：
        (best_x, best_f) 元组，其中 best_x 是近似全局最小值点，best_f 是对应的函数值。
    
    

    
    
    
"""
    grids = [np.linspace(lo, hi, n_points) for lo, hi in bounds]
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack(mesh, axis=-1).reshape(-1, len(bounds))

    best_x = points[0]
    best_f = f(points[0])
    for pt in points[1:]:
        val = f(pt)
        if val < best_f:
            best_f = val
            best_x = pt

    return best_x, best_f


def simulated_annealing(
    f: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    x0: Optional[np.ndarray] = None,
    T0: float = 10.0,
    cooling_rate: float = 0.95,
    sigma: float = 0.5,
    max_iterations: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """模拟退火用于全局优化。
    通过偶尔接受增加目标函数值的移动来利用概率接受准则以逃离局部极小值。
参数：
        f: 目标函数。
        bounds: 每个维度的 (下界，上界) 边界列表。
    x0: 初始点，可选。默认为边界中心。
        T0: 初始温度。
        cooling_rate: (0, 1) 范围内的乘法冷却因子。
        sigma: 扰动的标准差。
        max_iterations: 最大迭代次数。
        seed: 用于可重复性的随机种子。
    返回：
        (best_x, best_f) 元组。
    
    

    
    
"""
    rng = np.random.default_rng(seed)
    n = len(bounds)
    lowers = np.array([lo for lo, hi in bounds])
    uppers = np.array([hi for lo, hi in bounds])

    if x0 is None:
        x0 = (lowers + uppers) / 2.0

    x = x0.copy()
    fx = f(x)
    best_x = x.copy()
    best_f = fx
    T = T0

    for _ in range(max_iterations):
        candidate = x + sigma * rng.standard_normal(n)
        candidate = np.clip(candidate, lowers, uppers)
        fc = f(candidate)
        delta = fc - fx

        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-300)):
            x = candidate
            fx = fc

        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        T *= cooling_rate

    return best_x, best_f


def genetic_algorithm(
    f: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    pop_size: int = 50,
    n_generations: int = 200,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """遗传算法用于全局优化。
    使用受生物进化启发的选择、交叉和变异算子来进化候选解种群。
参数：
        f: 目标函数。
        bounds: 每个维度的 (下界，上界) 边界列表。
    pop_size: 种群大小。
        n_generations: 代数数量。
        mutation_rate: 每个基因发生变异的概率。
        crossover_rate: 两个亲本之间发生交叉的概率。
        seed: 用于可重复性的随机种子。
    返回：
        (best_x, best_f) 元组。
    
    

    
    
"""
    rng = np.random.default_rng(seed)
    n = len(bounds)
    lowers = np.array([lo for lo, hi in bounds])
    uppers = np.array([hi for lo, hi in bounds])

    population = rng.uniform(lowers, uppers, size=(pop_size, n))
    fitness = np.array([f(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]

    for _ in range(n_generations):
        # 锦标赛选择
        new_population = np.empty_like(population)
        for i in range(pop_size):
            a, b = rng.choice(pop_size, size=2, replace=False)
            winner = a if fitness[a] < fitness[b] else b
            new_population[i] = population[winner]

        # 交叉
        for i in range(0, pop_size - 1, 2):
            if rng.random() < crossover_rate:
                alpha = rng.uniform(0, 1, size=n)
                child1 = alpha * new_population[i] + (1 - alpha) * new_population[i + 1]
                child2 = (1 - alpha) * new_population[i] + alpha * new_population[i + 1]
                new_population[i] = child1
                new_population[i + 1] = child2

        # 变异
        for i in range(pop_size):
            mask = rng.random(n) < mutation_rate
            new_population[i, mask] += rng.standard_normal(np.sum(mask)) * (uppers[mask] - lowers[mask]) * 0.1
            new_population[i] = np.clip(new_population[i], lowers, uppers)

        population = new_population
        fitness = np.array([f(ind) for ind in population])

        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_x = population[gen_best_idx].copy()

    return best_x, best_f
