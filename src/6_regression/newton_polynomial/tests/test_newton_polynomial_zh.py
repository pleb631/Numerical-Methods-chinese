# 本文件为 test_newton_polynomial.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_newton_polynomial.py
import pytest
import numpy as np
from ..implementation.newton_polynomial import newton_polynomial


def test_newton_polynomial_basic():
    x = np.array([0, 1, 2])
    y = np.array([1, 3, 2])
    point = 1.5
    result = newton_polynomial(x, y, point)
    expected = 2.875
    assert np.isclose(result, expected, atol=1e-6)


def test_newton_polynomial_single_point():
    x = np.array([2])
    y = np.array([5])
    point = 2
    with pytest.raises(ValueError):
        newton_polynomial(x, y, point)


def test_newton_polynomial_exact_match():
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 8, 27])
    point = 2
    result = newton_polynomial(x, y, point)
    expected = 8.0
    assert np.isclose(result, expected, atol=1e-6)


def test_newton_polynomial_out_of_bounds_low():
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    point = 0
    result = newton_polynomial(x, y, point)
    # 1. 线性外推：y = 2x
    expected = 0
    assert np.isclose(result, expected, atol=1e-6)


def test_newton_polynomial_out_of_bounds_high():
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    point = 4
    result = newton_polynomial(x, y, point)
    # 2. 线性外推：y = 2x
    expected = 8
    assert np.isclose(result, expected, atol=1e-6)


def test_newton_polynomial_non_equal_lengths():
    x = np.array([0, 1, 2])
    y = np.array([0, 1])
    point = 1
    with pytest.raises(ValueError):
        newton_polynomial(x, y, point)


def test_newton_polynomial_insufficient_points():
    x = np.array([])
    y = np.array([])
    point = 0
    with pytest.raises(ValueError):
        newton_polynomial(x, y, point)


def test_newton_polynomial_duplicate_x_values():
    x = np.array([0, 1, 1, 2])
    y = np.array([0, 1, 1, 4])
    point = 1
    with pytest.raises(ValueError):
        newton_polynomial(x, y, point)


def test_newton_polynomial_negative_values():
    x = np.array([-2, -1, 0, 1])
    y = np.array([4, 1, 0, 1])
    point = -1.5
    result = newton_polynomial(x, y, point)
    expected = 2.25
    assert np.isclose(result, expected, atol=1e-6)


def test_newton_polynomial_float_precision():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])
    point = 1.999999
    result = newton_polynomial(x, y, point)
    expected = 3.999996
    assert np.isclose(result, expected, atol=1e-5)


def test_newton_polynomial_multiple_segments():
    x = np.linspace(0, 10, 11)
    y = x ** 2
    point = 7.3
    result = newton_polynomial(x, y, point)
    expected = 7.3 ** 2
    assert np.isclose(result, expected, atol=1e-2)


def test_newton_polynomial_large_dataset():
    x = np.linspace(-10, 10, 21)
    y = np.sin(x)
    point = 2.345
    result = newton_polynomial(x, y, point)
    expected = np.sin(2.345)
    assert np.isclose(result, expected, atol=1e-3)


def test_newton_polynomial_exact_match_middle():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])
    point = 2
    result = newton_polynomial(x, y, point)
    expected = 4.0
    assert np.isclose(result, expected, atol=1e-6)


def test_newton_polynomial_multiple_queries():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])
    points = [0.5, 1.5, 2.5, 3.5]
    expected = [0.25, 2.25, 6.25, 12.25]
    results = [newton_polynomial(x, y, p) for p in points]
    assert np.allclose(results, expected, atol=1e-2)


def test_newton_polynomial_non_monotonic_x():
    x = np.array([0, 2, 1, 3])
    y = np.array([0, 4, 2, 9])
    point = 1.5
    # 3. 即使 x 未排序，函数也应正确处理
    result = newton_polynomial(x, y, point)
    # 1. 预期结果基于未排序数据
    # 2. 手动计算或使用 numpy 的 polyfit 进行验证
    coeffs = np.polyfit(x, y, deg=3)
    expected = np.polyval(coeffs, point)
    assert np.isclose(result, expected, atol=1e-2)


def test_newton_polynomial_high_degree():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([1, 2, 1, 2, 1, 2])
    point = 2.5
    result = newton_polynomial(x, y, point)
    # 5. 预期值可与 numpy 的 polyfit 和 polyval 进行比较
    coeffs = np.polyfit(x, y, deg=5)
    expected = np.polyval(coeffs, point)
    assert np.isclose(result, expected, atol=1e-2)
