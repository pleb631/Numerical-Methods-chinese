# 本文件为 test_lagrange_polynomial.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_lagrange_polynomial.py
import pytest
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from ..implementation.lagrange_polynomial import lagrange_polynomial


def _scipy_ref(x, y, point):
    """来自 scipy.interpolate.BarycentricInterpolator 的参考值。"""
    return float(BarycentricInterpolator(x, y)(point))


def test_lagrange_polynomial_basic():
    x = np.array([0, 1, 2], dtype=float)
    y = np.array([1, 3, 2], dtype=float)
    result = lagrange_polynomial(x, y, 1.5)
    assert np.isclose(result, 2.875, atol=1e-10)


def test_lagrange_polynomial_single_point():
    x = np.array([2.0])
    y = np.array([5.0])
    assert np.isclose(lagrange_polynomial(x, y, 2), 5.0, atol=1e-10)


def test_lagrange_polynomial_exact_match():
    """在数据点处求值必须返回数据值（Kronecker δ）。"""
    x = np.array([0, 1, 2, 3], dtype=float)
    y = np.array([0, 1, 8, 27], dtype=float)
    for xi, yi in zip(x, y):
        assert np.isclose(lagrange_polynomial(x, y, xi), yi, atol=1e-10)


def test_lagrange_polynomial_extrapolation():
    """线性数据 y = 2x：外推应保持精确。"""
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    assert np.isclose(lagrange_polynomial(x, y, 0), 0.0, atol=1e-10)
    assert np.isclose(lagrange_polynomial(x, y, 4), 8.0, atol=1e-10)


def test_lagrange_polynomial_non_equal_lengths():
    with pytest.raises(ValueError):
        lagrange_polynomial(np.array([0, 1, 2]), np.array([0, 1]), 1)


def test_lagrange_polynomial_insufficient_points():
    with pytest.raises(ValueError):
        lagrange_polynomial(np.array([]), np.array([]), 0)


def test_lagrange_polynomial_duplicate_x_values():
    with pytest.raises(ValueError):
        lagrange_polynomial(
            np.array([0, 1, 1, 2], dtype=float),
            np.array([0, 1, 1, 4], dtype=float),
            1,
        )


def test_lagrange_polynomial_negative_domain():
    """y = x² 在 {-2, -1, 0, 1} 上 — 多项式拟合完全精确。"""
    x = np.array([-2, -1, 0, 1], dtype=float)
    y = np.array([4, 1, 0, 1], dtype=float)
    assert np.isclose(lagrange_polynomial(x, y, -1.5), 2.25, atol=1e-10)


def test_lagrange_polynomial_float_precision():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])
    result = lagrange_polynomial(x, y, 1.999999)
    expected = _scipy_ref(x, y, 1.999999)
    assert np.isclose(result, expected, atol=1e-10)


def test_lagrange_polynomial_quadratic_exact():
    """y = x² 在 11 个点上的数据：通过它们的唯一次数 ≤ 10 的多项式即为 x²。"""
    x = np.linspace(0, 10, 11)
    y = x**2
    result = lagrange_polynomial(x, y, 7.3)
    assert np.isclose(result, 7.3**2, atol=1e-8)


def test_lagrange_polynomial_large_dataset_sin():
    x = np.linspace(-100, 100, 201)
    y = np.sin(x)
    result = lagrange_polynomial(x, y, 23.456)
    assert np.isclose(result, np.sin(23.456), atol=1e-3)


def test_lagrange_polynomial_multiple_queries():
    """y = x² 在 {0..4} 上：检查半整数点。"""
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = x**2
    points = [0.5, 1.5, 2.5, 3.5]
    expected = [p**2 for p in points]
    results = [lagrange_polynomial(x, y, p) for p in points]
    assert np.allclose(results, expected, atol=1e-10)


def test_lagrange_polynomial_notes_example():
    """示例：A(-1,1), B(2,3), C(3,5) → L(x) = x²/3 + x/3 + 1。"""
    x = np.array([-1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0])
    # 数据点精确
    for xi, yi in zip(x, y):
        assert np.isclose(lagrange_polynomial(x, y, xi), yi, atol=1e-10)
    # 查询点
    assert np.isclose(lagrange_polynomial(x, y, 0), 1.0, atol=1e-10)
    assert np.isclose(lagrange_polynomial(x, y, 1), 5 / 3, atol=1e-10)


def test_lagrange_polynomial_matches_scipy():
    """在多个数据集上将我们的实现与 scipy 进行比较。"""
    datasets = [
        (np.array([0, 1, 2], dtype=float), np.array([1, 3, 2], dtype=float)),
        (np.array([-1, 2, 3], dtype=float), np.array([1, 3, 5], dtype=float)),
        (
            np.array([-9, -5, -2.5, 4, 7], dtype=float),
            np.array([-2, 3, 0, 5, 11], dtype=float),
        ),
    ]
    for x, y in datasets:
        for pt in np.linspace(x.min(), x.max(), 10):
            ours = lagrange_polynomial(x, y, float(pt))
            ref = _scipy_ref(x, y, float(pt))
            assert np.isclose(
                ours, ref, atol=1e-10
            ), f"Mismatch at {pt}: impl={ours}, scipy={ref}"
