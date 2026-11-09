# 本文件为 regression_methods_comparison.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
from typing import Dict, List, Tuple, Callable


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(y_true) == 0:
        raise ValueError("Arrays must not be empty.")

    residuals = y_true - y_pred
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    n = len(y_true)

    rmse = float(np.sqrt(rss / n))
    mae = float(np.mean(np.abs(residuals)))
    if tss > 0:
        r_squared = 1.0 - rss / tss
    else:
        # 5. 常数 y_true：R^2 未定义；仅在精确拟合时报告 1.0。
        r_squared = 1.0 if rss == 0 else 0.0
    max_error = float(np.max(np.abs(residuals)))

    return {
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
        "max_error": max_error,
        "rss": rss,
    }


def compare_methods(
    x_data: np.ndarray,
    y_data: np.ndarray,
    methods: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """1. 根据方法在拟合数据上的误差对每种方法进行评分。

这些是*训练*指标：更灵活的模型总是能至少同样好地匹配训练点，因此按这些指标排名会偏向过拟合。
请使用 :func:`cross_validate_methods` 进行留一比较。

"""
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("X and Y vectors must have equal number of elements.")
    if x_data.shape[0] < 2:
        raise ValueError("At least 2 data points are required.")
    if len(methods) == 0:
        raise ValueError("At least one method must be provided.")

    results = {}
    for name, method in methods.items():
        y_pred = method(x_data, y_data)
        results[name] = compute_metrics(y_data, y_pred)

    return results


def rank_methods(
    results: Dict[str, Dict[str, float]],
    metric: str = "rmse",
) -> List[Tuple[str, float]]:
    """2. 按选定的指标对 ``(name, value)`` 对进行降序排序。

适用于 :func:`compare_methods`（训练误差）或 :func:`cross_validate_methods`（留一误差）的输出。

"""
    if len(results) == 0:
        raise ValueError("Results dictionary must not be empty.")
    if metric not in ("rmse", "mae", "r_squared", "max_error", "rss"):
        raise ValueError(f"Unknown metric: {metric}")

    items = [(name, metrics[metric]) for name, metrics in results.items()]

    if metric == "r_squared":
        items.sort(key=lambda x: x[1], reverse=True)
    else:
        items.sort(key=lambda x: x[1])

    return items


def polynomial_fit_predict(
    degree: int,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """3. 返回 ``fit_predict(x_train, y_train, x_eval)`` 以进行多项式拟合。"""
    if degree < 0:
        raise ValueError("Degree must be non-negative.")

    def fit_predict(
        x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray
    ) -> np.ndarray:
        coeffs = np.polyfit(x_train, y_train, degree)
        return np.polyval(coeffs, x_eval)

    return fit_predict


def cross_validate_methods(
    x_data: np.ndarray,
    y_data: np.ndarray,
    methods: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
    n_folds: int = 5,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """4. 基于留一预测的 K 折交叉验证指标。

与 :func:`compare_methods`（对拟合所用的每个点本身进行评分，因此更灵活的模型看起来总是至少同样好）不同，这里的每个点都是由从未见过该点的模型进行预测的。
``methods`` 将名称映射到 ``fit_predict(x_train, y_train, x_eval)``，该函数在训练部分进行拟合并在 ``x_eval`` 处返回预测值。
所有折的留一预测结果被汇总，并使用 :func:`compute_metrics` 进行评分；结果具有与 :func:`compare_methods` 相同的布局，因此可以传递给 :func:`rank_methods`。
``n_folds`` 被裁剪为点的数量（留一）。折是带种子的随机划分，对每种方法都相同。








"""
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("X and Y vectors must have equal number of elements.")
    if x_data.shape[0] < 3:
        raise ValueError("At least 3 data points are required for cross-validation.")
    if len(methods) == 0:
        raise ValueError("At least one method must be provided.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    n = x_data.shape[0]
    n_folds = min(n_folds, n)
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, n_folds)

    results = {}
    for name, fit_predict in methods.items():
        y_pred = np.empty(n, dtype=float)
        for held_out in folds:
            train = np.ones(n, dtype=bool)
            train[held_out] = False
            y_pred[held_out] = fit_predict(
                x_data[train], y_data[train], x_data[held_out]
            )
        results[name] = compute_metrics(y_data.astype(float), y_pred)

    return results
