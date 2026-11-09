# 本文件为 test_cubic_spline.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_cubic_spline.py
import pytest
import numpy as np
from scipy.interpolate import CubicSpline
from ..implementation.cubic_spline import cubic_spline


def _scipy_natural(x, y, point):
    """来自 scipy 自然三次样条的参考值。"""
    return float(CubicSpline(x, y, bc_type="natural")(point))


def test_cubic_spline_basic():
    x = np.array([0, 1, 2, 3], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=float)
    spline = cubic_spline(x, y)
    result = spline(1.5)
    expected = _scipy_natural(x, y, 1.5)
    assert np.isclose(result, expected, atol=1e-10)


def test_cubic_spline_exact_at_knots():
    """样条必须在每个节点处重现数据值。"""
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = np.array([0, 1, 8, 27, 64], dtype=float)
    spline = cubic_spline(x, y)
    for xi, yi in zip(x, y):
        assert np.isclose(spline(xi), yi, atol=1e-10)


def test_cubic_spline_natural_boundary():
    """对于线性数据，自然样条精确重现该直线。"""
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    spline = cubic_spline(x, y)
    # 对于线性数据，所有 M_i 应为 0 → 样条 = 直线
    for pt in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        assert np.isclose(spline(pt), 2 * pt, atol=1e-10)


def test_cubic_spline_natural_boundary_second_derivative():
    """对于具有非零内部二阶导数的非线性数据，S''(x_0) = S''(x_n) = 0。"""
    x = np.array([0.0, 1.0, 2.5, 3.0, 5.0])
    y = np.array([1.0, -2.0, 4.0, 0.5, 3.0])
    spline = cubic_spline(x, y)
    h = 1e-4
    # 内部二阶导数（中心差分）明显不为零。
    s_mid = (spline(2.5 + h) - 2 * spline(2.5) + spline(2.5 - h)) / h**2
    assert abs(s_mid) > 1.0
    # 单侧二阶差分等于端点处的 S'' 加上 S''' * h
    # 精确为 0（每段均为三次），因此对于自然样条曲线，它们必须约为 0。
    s_left = (spline(x[0] + 2 * h) - 2 * spline(x[0] + h) + spline(x[0])) / h**2
    s_right = (spline(x[-1]) - 2 * spline(x[-1] - h) + spline(x[-1] - 2 * h)) / h**2
    assert abs(s_left) < 1e-2
    assert abs(s_right) < 1e-2


def test_cubic_spline_out_of_bounds_low():
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    spline = cubic_spline(x, y)
    with pytest.raises(ValueError):
        spline(0)


def test_cubic_spline_out_of_bounds_high():
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    spline = cubic_spline(x, y)
    with pytest.raises(ValueError):
        spline(4)


def test_cubic_spline_non_equal_lengths():
    with pytest.raises(ValueError):
        cubic_spline(np.array([0, 1, 2]), np.array([0, 1]))


def test_cubic_spline_insufficient_points():
    with pytest.raises(ValueError):
        cubic_spline(np.array([0, 1]), np.array([0, 1]))


def test_cubic_spline_duplicate_x_values():
    with pytest.raises(ValueError):
        cubic_spline(np.array([0, 1, 1, 2]), np.array([0, 1, 1, 4]))


def test_cubic_spline_negative_domain():
    """y = x² on {-2,-1,0,1} — compare against scipy."""
    x = np.array([-2, -1, 0, 1], dtype=float)
    y = x**2
    spline = cubic_spline(x, y)
    for pt in [-1.5, -0.5, 0.5]:
        result = spline(pt)
        expected = _scipy_natural(x, y, pt)
        assert np.isclose(result, expected, atol=1e-10)


def test_cubic_spline_float_precision():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])
    spline = cubic_spline(x, y)
    result = spline(1.999999)
    expected = _scipy_natural(x, y, 1.999999)
    assert np.isclose(result, expected, atol=1e-8)


def test_cubic_spline_quadratic_data():
    """11 points on y=x²: spline approximates well (not exact — cubic, not quadratic)."""
    x = np.linspace(0, 10, 11)
    y = x**2
    spline = cubic_spline(x, y)
    result = spline(7.3)
    expected = _scipy_natural(x, y, 7.3)
    assert np.isclose(result, expected, atol=1e-10)


def test_cubic_spline_sin_data():
    x = np.linspace(-100, 100, 201)
    y = np.sin(x)
    spline = cubic_spline(x, y)
    result = spline(23.456)
    expected = _scipy_natural(x, y, 23.456)
    assert np.isclose(result, expected, atol=1e-10)


def test_cubic_spline_multiple_queries():
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = x**2
    spline = cubic_spline(x, y)
    for pt in [0.5, 1.5, 2.5, 3.5]:
        result = spline(pt)
        expected = _scipy_natural(x, y, pt)
        assert np.isclose(result, expected, atol=1e-10)


def test_cubic_spline_notes_example():
    """Notes worked example: (0,0), (1,0.5), (2,0) → S(0.5) = 0.34375."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.5, 0.0])
    spline = cubic_spline(x, y)
    # 来自笔记的精确值
    assert np.isclose(spline(0.5), 0.34375, atol=1e-10)
    assert np.isclose(spline(0.0), 0.0, atol=1e-10)
    assert np.isclose(spline(1.0), 0.5, atol=1e-10)
    assert np.isclose(spline(2.0), 0.0, atol=1e-10)
    # 也与 scipy 匹配
    for pt in [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]:
        assert np.isclose(spline(pt), _scipy_natural(x, y, pt), atol=1e-10)


def test_cubic_spline_matches_scipy_scattered():
    """更大的数据集已与 scipy 进行了验证。"""
    np.random.seed(0)
    x = np.sort(np.random.uniform(0, 10, 15))
    y = np.sin(x) * np.cos(x / 2)
    spline = cubic_spline(x, y)
    ref = CubicSpline(x, y, bc_type="natural")
    pts = np.linspace(x[0], x[-1], 50)
    for pt in pts:
        assert np.isclose(
            spline(float(pt)), float(ref(pt)), atol=1e-10
        ), f"Mismatch at {pt}"
