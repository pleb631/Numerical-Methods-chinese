# 本文件为 test_secant_method.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_secant_method.py
import pytest
import numpy as np
from ..implementation.secant_method import secant_method


def test_secant_linear():
    f = lambda x: 2 * x - 4
    root = secant_method(f, 0, 5)
    expected = 2.0
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_quadratic():
    f = lambda x: x**2 - 4
    root = secant_method(f, 1, 3)
    expected = 2.0
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_sin():
    f = np.sin
    root = secant_method(f, 3, 4)
    expected = np.pi
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_no_root():
    # exp(x) 没有实根；迭代值仅向负无穷漂移
    # 永不低于 tol 的步骤，因此迭代上限决定了其停止。
    f = lambda x: np.exp(x)
    with pytest.raises(ValueError, match="did not converge"):
        secant_method(f, 0, 1, max_iterations=50)


def test_secant_zero_denominator():
    # 常数函数给出 f(x1) == f(x0)，因此割线是水平的。
    f = lambda x: 3.0
    with pytest.raises(ValueError, match="Denominator is zero"):
        secant_method(f, 0.0, 1.0)


def test_secant_multiple_roots():
    f = lambda x: x**3 - x
    root = secant_method(f, 0.5, 2.0)
    expected = 1.0
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_tolerance():
    f = lambda x: x**3 - 6 * x**2 + 11 * x - 6
    root = secant_method(f, 2.5, 3.5, tol=1e-10)
    expected = 3.0
    assert np.isclose(root, expected, atol=1e-10)


def test_secant_max_iterations():
    f = lambda x: 1 - x
    # 第一步恰好落在根上，但步长（99）是
    # 仍 >= tol，因此一次迭代不足以*确认*收敛。
    with pytest.raises(ValueError, match="did not converge"):
        secant_method(f, -100, 100, max_iterations=1)
    # 第二次迭代检测到零步长并返回根。
    assert secant_method(f, -100, 100, max_iterations=2) == 1.0


def test_secant_exact_root():
    f = lambda x: x - 5
    root = secant_method(f, 0, 10)
    expected = 5.0
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_close_to_root():
    f = lambda x: x**2 - 2
    root = secant_method(f, 1.4, 1.5)
    expected = np.sqrt(2)
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_negative_root():
    f = lambda x: x + 3
    root = secant_method(f, 0, 1)
    expected = -3.0
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_high_precision():
    f = lambda x: x**3 - 6 * x**2 + 11 * x - 6
    root = secant_method(f, 3.5, 4.0, tol=1e-12)
    expected = 3.0
    assert np.isclose(root, expected, atol=1e-12)


def test_secant_initial_guess():
    f = lambda x: x**2 - 1
    root = secant_method(f, 0.5, 2.0)
    expected = 1.0
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_fractional_minimum():
    f = lambda x: x - 2.5
    root = secant_method(f, 1.0, 3.0)
    expected = 2.5
    assert np.isclose(root, expected, atol=1e-8)


def test_secant_function_with_noise():
    np.random.seed(0)
    # 斜率为 1 的函数上 1e-6 的噪声将根限制在约 1e-6。
    f = lambda x: x - 3 + np.random.normal(0, 1e-6)
    root = secant_method(f, 2.0, 4.0)
    expected = 3.0
    assert np.isclose(root, expected, atol=1e-5)


def test_secant_small_scale_function():
    # f 的差分极小 (< 1e-8)，但问题是完全良定义的。
    f = lambda x: 1e-10 * (x - 2.0)
    root = secant_method(f, 0.0, 5.0)
    expected = 2.0
    assert np.isclose(root, expected, atol=1e-6)
