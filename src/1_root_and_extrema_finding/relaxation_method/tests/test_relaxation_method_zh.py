# 本文件为 test_relaxation_method.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_relaxation_method.py
import pytest
import numpy as np
from ..implementation.relaxation_method import relaxation_method


def test_relaxation_constant():
    f = lambda x: 5.0
    initial_guess = 0.0
    root = relaxation_method(f, initial_guess)
    expected = 5.0
    assert np.isclose(root, expected, atol=1e-6)


def test_relaxation_linear():
    f = lambda x: 2.0 * x + 1.0
    initial_guess = 0.0
    root = relaxation_method(f, initial_guess)
    # 不动点：x = 2x + 1 => x = -1
    expected = -1.0
    assert np.isclose(root, expected, atol=1e-6)


def test_relaxation_quadratic():
    f = lambda x: np.sqrt(4 + x)
    initial_guess = 0.0
    root = relaxation_method(f, initial_guess)
    # 不动点：x = sqrt(4 + x) => x ≈ 2.5615528128
    expected = 2.5615528128
    assert np.isclose(root, expected, atol=1e-6)


def test_relaxation_cos():
    f = lambda x: np.cos(x)
    root = relaxation_method(f, 1.0, tol=1e-12)
    # cos 的不动点（多蒂数）
    assert np.isclose(root, 0.7390851332151607, atol=1e-10)


def test_relaxation_sin_slow_but_converges():
    # 1. g'(0) = 1，因此收敛到不动点 0 是次线性的。步长
    # 2. 标准 |x_new - x| < tol 因此会在 x 距离不动点
    # 3. 小于 tol 之前停止：误差约为 (6 * tol)**(1/3)。
    f = lambda x: np.sin(x)
    root = relaxation_method(f, 2.0, tol=1e-6)
    assert 0.0 < root < 0.05


def test_relaxation_non_convergent_raises():
    f = lambda x: 1 - x
    # g'(x) = -1：普通迭代在 1000 和 -999 之间永远振荡。
    with pytest.raises(ValueError, match="did not converge"):
        relaxation_method(f, 1000.0, tol=1e-6, max_iterations=100)


def test_relaxation_max_iterations_exhausted_raises():
    f = lambda x: 0.5 * x + 1
    with pytest.raises(ValueError, match="did not converge"):
        relaxation_method(f, 0.0, omega=0.5, tol=1e-12, max_iterations=3)


def test_relaxation_exact_fixed_point():
    f = lambda x: 3.0
    initial_guess = 10.0
    root = relaxation_method(f, initial_guess)
    expected = 3.0
    assert np.isclose(root, expected, atol=1e-6)


def test_relaxation_close_to_fixed_point():
    f = lambda x: x - 0.1 * (x**3 - x - 2)
    initial_guess = 1.5
    root = relaxation_method(f, initial_guess, omega=0.5, tol=1e-10)
    expected = 1.5213797068045676  # x^3 - x - 2 = 0 的近似根
    assert np.isclose(root, expected, atol=1e-6)


def test_relaxation_negative_fixed_point():
    f = lambda x: -2.0
    initial_guess = 1.0
    root = relaxation_method(f, initial_guess)
    expected = -2.0
    assert np.isclose(root, expected, atol=1e-6)


def test_relaxation_high_precision():
    f = lambda x: x / 2 + 1
    initial_guess = 0.0
    root = relaxation_method(f, initial_guess, tol=1e-12)
    # 不动点：x = x/2 + 1 => x = 2
    expected = 2.0
    assert np.isclose(root, expected, atol=1e-12)


def test_relaxation_initial_guess():
    f = lambda x: x**2 - 2
    initial_guess = 1.5
    root = relaxation_method(f, initial_guess)
    # 不动点：x = 2 和 x = -1
    assert not np.isclose(root, 0.0, atol=1e-6)
    assert np.isclose(root, 2.0, atol=1e-6) or np.isclose(root, -1.0, atol=1e-6)


def test_relaxation_fractional_fixed_point():
    f = lambda x: 1.5
    initial_guess = 0.0
    root = relaxation_method(f, initial_guess)
    expected = 1.5
    assert np.isclose(root, expected, atol=1e-6)


def test_relaxation_large_number_iterations():
    f = lambda x: x
    initial_guess = 100.0
    root = relaxation_method(f, initial_guess, tol=1e-12, max_iterations=100000)
    expected = 100.0
    assert np.isclose(root, expected, atol=1e-12)


def test_relaxation_omega_adjustment():
    f = lambda x: 0.5 * x + 1
    initial_guess = 0.0
    root = relaxation_method(f, initial_guess)
    # 不动点：x = 0.5x + 1 => x = 2
    expected = 2.0
    assert np.isclose(root, expected, atol=1e-6)


def test_relaxation_user_supplied_omega_is_used():
    # 1. g(x) = -2x + 3 具有不动点 1，且 g'(x) = -2。
    # 2. 当 omega = 1/3 时，松弛映射 x -> 1 精确成立，因此一个固定的 omega
    # 3. 会立即收敛；而自适应选择（omega = 0.5）则不会。
    f = lambda x: -2.0 * x + 3.0
    root = relaxation_method(f, 5.0, omega=1.0 / 3.0, tol=1e-12, max_iterations=3)
    assert np.isclose(root, 1.0, atol=1e-9)
