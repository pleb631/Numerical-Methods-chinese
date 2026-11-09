# 本文件为 test_polynomial_regression.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_polynomial_regression.py
import pytest
import numpy as np
from ..implementation.polynomial_regression import polynomial_regression


def test_polynomial_regression_linear():
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x + 1
    degree = 1
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([1, 2])
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_quadratic():
    x = np.array([-2, -1, 0, 1, 2])
    y = x**2 + 2 * x + 1
    degree = 2
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([1, 2, 1])
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_cubic():
    x = np.array([-1, 0, 1, 2])
    y = -(x**3) + 3 * x**2 + x + 5
    degree = 3
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([5, 1, 3, -1])
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_exact_match():
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    degree = 1
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([1, 1])  # y = x +1
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_overdetermined():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 3, 7, 13, 21])
    degree = 2
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([1, 1, 1])  # y = 1 + x + x^2
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_exactly_determined_line():
    # 两个点且次数为 1：n == degree + 1，因此拟合结果进行插值。
    x = np.array([0, 1])
    y = np.array([1, 3])
    degree = 1
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([1, 2])  # y = 1 + 2x
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_insufficient_points():
    x = np.array([0, 1])
    y = np.array([1, 3])
    degree = 2
    with pytest.raises(ValueError):
        polynomial_regression(x, y, degree)


def test_polynomial_regression_non_equal_lengths():
    x = np.array([0, 1, 2])
    y = np.array([1, 3])
    degree = 1
    with pytest.raises(ValueError):
        polynomial_regression(x, y, degree)


def test_polynomial_regression_negative_degree():
    x = np.array([0, 1, 2])
    y = np.array([1, 3, 5])
    degree = -1
    with pytest.raises(ValueError):
        polynomial_regression(x, y, degree)


def test_polynomial_regression_duplicate_x_values():
    x = np.array([1, 1, 2, 3])
    y = np.array([2, 2, 4, 6])
    degree = 1
    # 多项式回归可以处理重复的 x 值，前提是 y 值一致。
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([0, 2])  # y = 2x
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_noisy_data():
    np.random.seed(0)
    x = np.linspace(0, 10, 50)
    y = 3 + 2 * x + x**2 + np.random.normal(0, 1, x.shape[0])
    degree = 2
    coeffs = polynomial_regression(x, y, degree)
    # 拟合结果必须是 polyfit 计算出的最小二乘解。
    assert np.allclose(coeffs, np.polyfit(x, y, 2)[::-1], rtol=1e-6, atol=1e-8)
    # 它必须保持在真实曲线的两个噪声标准差范围内。
    true_curve = 3 + 2 * x + x**2
    fitted = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2
    assert np.max(np.abs(fitted - true_curve)) < 2 * 1.0


def test_polynomial_regression_high_degree():
    x = np.linspace(-1, 1, 10)
    y = 1 - 2 * x + 3 * x**2 - 4 * x**3 + 5 * x**4
    degree = 4
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([1, -2, 3, -4, 5])
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_large_dataset():
    np.random.seed(1)
    x = np.linspace(0, 100, 1001)
    y = 5 + 3 * x + 2 * x**2 + np.random.normal(0, 10, x.shape[0])
    degree = 2
    coeffs = polynomial_regression(x, y, degree)
    assert np.allclose(coeffs, np.polyfit(x, y, 2)[::-1], rtol=1e-6, atol=1e-8)
    true_curve = 5 + 3 * x + 2 * x**2
    fitted = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2
    assert np.max(np.abs(fitted - true_curve)) < 2 * 10.0


def test_polynomial_regression_single_variable():
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 4, 11, 22])
    degree = 2
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([1, 1, 2])  # y = 1 + x + 2x^2
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_exact_polynomial():
    def poly(x):
        return 4 - 3 * x + 2 * x**2 - x**3

    x = np.array([-2, -1, 0, 1, 2])
    y = poly(x)
    degree = 3
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([4, -3, 2, -1])
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_extrapolation():
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 4, 11, 22])
    degree = 2
    coeffs = polynomial_regression(x, y, degree)
    point = 4  # 在拟合范围 [0, 3] 之外。
    result = np.dot(coeffs, np.array([1, point, point**2]))
    expected = 1 + point + 2 * point**2  # 基于 y = 1 + x + 2x^2。
    assert np.isclose(result, expected, atol=1e-8)


def test_polynomial_regression_integer_coefficients():
    x = np.array([1, 2, 3, 4])
    y = np.array([8, 15, 26, 41])
    degree = 2
    coeffs = polynomial_regression(x, y, degree)
    expected = np.array([5, 1, 2])  # y = 5 + x + 2x^2
    assert np.allclose(coeffs, expected, atol=1e-6)


def test_polynomial_regression_small_scale_x():
    # 小量级的 x 不应因绝对容差而被拒绝为“奇异”。
    x = np.linspace(0, 0.01, 20)
    y = 1 + 2 * x + 3 * x**2 + 4 * x**3
    coeffs = polynomial_regression(x, y, 3)
    assert np.allclose(coeffs, [1, 2, 3, 4], rtol=1e-3)


def test_polynomial_regression_integer_input_no_overflow():
    # 整数 x 的幂次在正规方程中会导致 int64 溢出。
    rng = np.random.default_rng(0)
    x = np.arange(0, 5000, 50)
    y = 1 + x + 1e-3 * x**2 + rng.normal(0, 1, len(x))
    coeffs = polynomial_regression(x, y, 3)
    expected = np.polyfit(x.astype(float), y, 3)[::-1]
    assert np.allclose(coeffs, expected, rtol=1e-4, atol=1e-8)
