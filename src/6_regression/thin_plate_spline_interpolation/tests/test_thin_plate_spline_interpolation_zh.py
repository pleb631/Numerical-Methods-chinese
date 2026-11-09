# 本文件为 test_thin_plate_spline_interpolation.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_thin_plate_spline_interpolation.py
import pytest
import numpy as np
from scipy.interpolate import RBFInterpolator
from ..implementation.thin_plate_spline_interpolation import (
    thin_plate_spline_interpolation,
)


def _scipy_tps(x, y, z, point):
    """使用 scipy.interpolate.RBFInterpolator 引用薄板样条（TPS）。"""
    coords = np.column_stack([x, y])
    rbf = RBFInterpolator(coords, z, kernel="thin_plate_spline", degree=1)
    return float(rbf(np.array([point]))[0])


def test_tps_basic():
    """非共线三角形。"""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 1.0, 0.0])
    pt = (1.0, 0.5)
    result = thin_plate_spline_interpolation(x, y, z, pt)
    expected = _scipy_tps(x, y, z, pt)
    assert np.isclose(result, expected, atol=1e-10)


def test_tps_single_point():
    x = np.array([2])
    y = np.array([3])
    z = np.array([5])
    with pytest.raises(ValueError):
        thin_plate_spline_interpolation(x, y, z, (2, 3))


def test_tps_exact_match():
    """在数据点处评估必须返回该数据点的值。"""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 4.0, 9.0])
    result = thin_plate_spline_interpolation(x, y, z, (1.0, 0.0))
    assert np.isclose(result, 1.0, atol=1e-8)


def test_tps_extrapolation():
    """TPS 可以在数据点的凸包之外进行外推。"""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])  # 平面数据：z = x + y
    pt = (2.0, 2.0)
    result = thin_plate_spline_interpolation(x, y, z, pt)
    assert np.isclose(result, 4.0, atol=1e-8)


def test_tps_non_equal_lengths():
    x = np.array([0, 1, 2])
    y = np.array([0, 1])
    z = np.array([0, 1, 4])
    with pytest.raises(ValueError):
        thin_plate_spline_interpolation(x, y, z, (1, 1))


def test_tps_insufficient_points():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        thin_plate_spline_interpolation(x, y, z, (0.5, 0.5))


def test_tps_duplicate_points():
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 1.0, 2.0])
    z = np.array([0.0, 1.0, 1.0, 4.0])
    with pytest.raises(ValueError):
        thin_plate_spline_interpolation(x, y, z, (1, 1))


def test_tps_negative_coordinates():
    """具有负坐标的二维散乱数据。"""
    x = np.array([-2.0, -1.0, 0.0, 1.0, -1.0])
    y = np.array([0.0, -1.0, 0.0, 1.0, 1.0])
    z = x**2 + y**2
    pt = (-1.5, 0.5)
    result = thin_plate_spline_interpolation(x, y, z, pt)
    expected = _scipy_tps(x, y, z, pt)
    assert np.isclose(result, expected, atol=1e-8)


def test_tps_float_precision():
    """在非常接近数据点处进行评估。"""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])  # z = x + y
    pt = (0.999999, 0.999999)
    result = thin_plate_spline_interpolation(x, y, z, pt)
    assert np.isclose(result, 2.0, atol=1e-4)


def test_tps_grid_data():
    """散乱网格点（非共线）。"""
    gx = np.array([0.0, 2.0, 4.0, 6.0])
    gy = np.array([0.0, 3.0, 6.0])
    X, Y = np.meshgrid(gx, gy)
    x = X.ravel()
    y = Y.ravel()
    z = x + 2 * y  # 平面数据
    pt = (3.0, 4.5)
    result = thin_plate_spline_interpolation(x, y, z, pt)
    assert np.isclose(result, 3.0 + 2 * 4.5, atol=1e-6)


def test_tps_exact_match_all_corners():
    """验证插值精确重现所有数据点。"""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([1.0, 0.0, 0.0, 1.0])
    for xi, yi, zi in zip(x, y, z):
        result = thin_plate_spline_interpolation(x, y, z, (xi, yi))
        assert np.isclose(
            result, zi, atol=1e-8
        ), f"Mismatch at ({xi},{yi}): got {result}, expected {zi}"


def test_tps_planar_data_exact():
    """平面数据 z = x + y 被精确重现（仿射部分）。"""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])
    for pt in [(0.5, 0.5), (0.25, 0.75), (0.3, 0.7)]:
        result = thin_plate_spline_interpolation(x, y, z, pt)
        assert np.isclose(
            result, pt[0] + pt[1], atol=1e-8
        ), f"Planar mismatch at {pt}: got {result}, expected {pt[0]+pt[1]}"


def test_tps_saddle_data_midpoint():
    """非平面马鞍状数据。"""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([1.0, 0.0, 0.0, 1.0])
    # 中点值 = 0.5（由对称性决定）
    assert np.isclose(
        thin_plate_spline_interpolation(x, y, z, (0.5, 0.5)), 0.5, atol=1e-8
    )
    # 其他点与 scipy 进行验证
    for pt in [(0.25, 0.75), (0.3, 0.1)]:
        result = thin_plate_spline_interpolation(x, y, z, pt)
        expected = _scipy_tps(x, y, z, pt)
        assert np.isclose(result, expected, atol=1e-8)


def test_tps_matches_scipy_scattered():
    """更大的散乱数据集与 scipy 进行验证。"""
    np.random.seed(42)
    N = 10
    x = np.random.rand(N) * 4
    y = np.random.rand(N) * 4
    z = np.sin(x) * np.cos(y)
    for pt in [(1.0, 1.0), (2.0, 3.0), (0.5, 2.5)]:
        result = thin_plate_spline_interpolation(x, y, z, pt)
        expected = _scipy_tps(x, y, z, pt)
        assert np.isclose(
            result, expected, atol=1e-8
        ), f"Mismatch at {pt}: got {result}, expected {expected}"


def test_tps_collinear_points_raise_value_error():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    z = np.array([0.0, 1.0, 4.0, 9.0])
    with pytest.raises(ValueError, match="collinear"):
        thin_plate_spline_interpolation(x, y, z, (1.5, 4.0))


def test_tps_collinear_axis_aligned_and_scaled():
    z = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="collinear"):
        thin_plate_spline_interpolation(
            np.array([0.0, 1.0, 2.0]), np.zeros(3), z, (0.5, 0.0)
        )
    # 共线性是相对于尺度检测的，适用于极小和极大的坐标。
    for scale in (1e-8, 1e8):
        t = np.array([0.0, 1.0, 2.0, 5.0]) * scale
        with pytest.raises(ValueError, match="collinear"):
            thin_plate_spline_interpolation(t, -3.0 * t, np.arange(4.0), (scale, 0.0))


def test_tps_no_implicit_meshgrid_mode():
    """x 和 y 必须与 z_data 长度匹配；网格状输入不会被扩展。"""
    gx = np.array([0.0, 1.0, 2.0])
    gy = np.array([0.0, 1.0])
    z = np.arange(6.0)
    with pytest.raises(ValueError):
        thin_plate_spline_interpolation(gx, gy, z, (0.5, 0.5))
