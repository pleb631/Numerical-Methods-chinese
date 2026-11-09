# 本文件为 gaussian_rbf.py 的中文注释版本（代码逻辑与英文版完全一致）
"""全局高斯径向基函数 (RBF) 插值。

这与在 ``gaussian_interpolation.py`` 中实现的等间距中心差分插值不同。

"""

import numpy as np


class GaussianRBFInterpolator:
    """使用高斯径向基函数（Gaussian RBF）对一维数据进行插值。

核函数为 ``exp(-(epsilon * (x - center)) ** 2)``。权重通过求解一次稠密插值系统得到，并在查询时复用。

参数
----
x_data, y_data : array-like, shape (n,)
    不同节点处的有限数据；不要求有序或等间距。必须至少提供一个数据点。
epsilon : float, optional
    高斯逆宽度的正有限值；默认为 1。

备注
----
极小的 epsilon 或紧密分布的节点可能导致核矩阵病态。该方法直接求解线性系统，而非显式求逆矩阵；不对数据进行正则化。




"""

    def __init__(self, x_data, y_data, epsilon=1.0):
        nodes = np.asarray(x_data, dtype=float)
        values = np.asarray(y_data, dtype=float)
        if nodes.ndim != 1 or values.ndim != 1:
            raise ValueError("X and Y data must be one-dimensional.")
        if nodes.size == 0:
            raise ValueError("At least one data point is required.")
        if nodes.size != values.size:
            raise ValueError("X and Y data must have equal lengths.")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(values)):
            raise ValueError("X and Y data must contain only finite values.")
        if np.unique(nodes).size != nodes.size:
            raise ValueError("X data must contain distinct nodes.")

        shape = float(epsilon)
        if not np.isfinite(shape) or shape <= 0:
            raise ValueError("epsilon must be finite and positive.")

        self.nodes = nodes.copy()
        self.epsilon = shape
        differences = self.nodes[:, None] - self.nodes[None, :]
        kernel = np.exp(-np.square(self.epsilon * differences))
        self.weights = np.linalg.solve(kernel, values)

    def __call__(self, points):
        """评估标量或数组查询，并保持其形状。"""
        queries = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(queries)):
            raise ValueError("Query points must be finite.")
        basis = np.exp(-np.square(self.epsilon * (queries[..., None] - self.nodes)))
        result = basis @ self.weights
        return float(result) if queries.ndim == 0 else result


def gaussian_rbf_interpolation(x_data, y_data, point, epsilon=1.0):
    """单个查询的便捷接口；进行多个查询时请使用该类。"""
    return GaussianRBFInterpolator(x_data, y_data, epsilon)(point)
