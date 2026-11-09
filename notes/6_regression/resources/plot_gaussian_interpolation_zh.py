# 本文件为 plot_gaussian_interpolation.py 的中文注释版本（代码逻辑与英文版完全一致）
"""生成 gaussian_interpolation.md 中使用的高斯径向基函数图。"""
from pathlib import Path
import sys

import numpy as np

from svg_plot_utils import PALETTE, line_chart

# 与可运行的 RBF 示例及其测试共享实现。
MODULE_ROOT = Path(__file__).resolve().parents[3] / "src" / "6_regression" / "gaussian_interpolation"
sys.path.insert(0, str(MODULE_ROOT))
from implementation.gaussian_rbf import GaussianRBFInterpolator

OUT = Path(__file__).with_name("plots")


def gaussian_basis(queries, centers, epsilon):
    """评估无权重高斯基函数，每列对应一个中心。"""
    return np.exp(-(epsilon * (queries[:, None] - centers[None, :])) ** 2)


def main():
    centers = np.array([0., 1., 2., 3.])
    values = np.array([0., 1., .2, 1.2])
    # 包含中心，以便 SVG 中的每个单位峰值都有表示。
    x = np.unique(np.concatenate((np.linspace(-.5, 3.5, 801), centers)))
    epsilon = 1.2
    basis = gaussian_basis(x, centers, epsilon)
    peaks = [
        {"x": [c], "y": [1.], "color": PALETTE[j]}
        for j, c in enumerate(centers)
    ]
    line_chart(
        OUT / "gaussian_rbf_basis_functions.svg",
        "Gaussian RBFs at the data sites (epsilon = 1.2)", "x", "basis value",
        [{"x": x, "y": basis[:, j], "label": f"center {c:g}"}
         for j, c in enumerate(centers)],
        peaks + [{"x": centers, "y": np.zeros_like(centers),
                  "label": "data sites (not data values)", "color": "#111827"}],
        max_points=None, legend_below=True,
    )

    series = []
    for epsilon in [.5, 1., 2.]:
        interpolator = GaussianRBFInterpolator(centers, values, epsilon)
        matrix = gaussian_basis(centers, centers, epsilon)
        series.append({
            "x": x, "y": interpolator(x),
            "label": f"epsilon={epsilon:g}, cond={np.linalg.cond(matrix):.1e}",
        })
    line_chart(
        OUT / "gaussian_rbf_shape_parameter.svg",
        "Shape parameter changes curve and conditioning", "x", "s(x)",
        series, [{"x": centers, "y": values, "label": "data"}],
        max_points=None, legend_below=True,
    )


if __name__ == "__main__":
    main()
