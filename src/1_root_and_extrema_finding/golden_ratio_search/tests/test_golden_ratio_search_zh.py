# 本文件为 test_golden_ratio_search.py 的中文注释版本（代码逻辑与英文版完全一致）
import pytest
import numpy as np
from ..implementation.golden_ratio_search import golden_ratio_search


def test_golden_ratio_search_linear():
    # 1. |2x - 4| 是单峰的，最小值在 x = 2
    f = lambda x: abs(2 * x - 4)
    a, b = 0, 5
    root = golden_ratio_search(f, a, b)
    expected = 2.0
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_quadratic():
    f = lambda x: (x - 3) ** 2
    a, b = 0, 6
    root = golden_ratio_search(f, a, b)
    expected = 3.0
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_sin():
    # 2. cos 在 [3, 4] 区间内仅在 pi 处有一个最小值
    f = lambda x: np.cos(x)
    a, b = 3, 4
    root = golden_ratio_search(f, a, b)
    expected = 3.141592653589793
    assert np.isclose(root, expected, atol=1e-6)


def test_golden_ratio_search_multiple_minima():
    # 3. 最小值位于 x = 1 和 x = 3；括号 [2, 4] 隔离了 x = 3
    f = lambda x: (x - 1) ** 2 * (x - 3) ** 2
    a, b = 2, 4
    root = golden_ratio_search(f, a, b)
    expected = 3.0
    assert np.isclose(root, expected, atol=1e-6)


def test_golden_ratio_search_minimum_value_not_zero():
    # 4. 即使最小值 -5 在 x = 1 处取得（f 在 1 处无根），优化器仍选择 x = 1
    f = lambda x: (x - 1) ** 2 - 5
    a, b = -3, 4
    root = golden_ratio_search(f, a, b)
    expected = 1.0
    assert np.isclose(root, expected, atol=1e-6)


def test_golden_ratio_search_tolerance():
    f = lambda x: (x - 2) ** 2
    a, b = 0, 4
    root = golden_ratio_search(f, a, b, tol=1e-10)
    expected = 2.0
    assert np.isclose(root, expected, atol=1e-10)


def test_golden_ratio_search_max_iterations():
    f = lambda x: (x - 2) ** 2
    a, b = 0, 4
    with pytest.raises(ValueError):
        golden_ratio_search(f, a, b, max_iterations=0)


def test_golden_ratio_search_exact_minimum():
    f = lambda x: (x - 5) ** 2
    a, b = 3, 7
    root = golden_ratio_search(f, a, b)
    expected = 5.0
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_close_to_minimum():
    f = lambda x: x**2 - 2 * x + 1
    a, b = 0, 3
    root = golden_ratio_search(f, a, b)
    expected = 1.0
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_sin_minimum():
    # 5. sin 在 [pi, 2*pi] 区间内是单峰的，最小值在 3*pi/2
    f = lambda x: np.sin(x)
    a, b = np.pi, 2 * np.pi
    root = golden_ratio_search(f, a, b)
    expected = 1.5 * np.pi
    assert np.isclose(root, expected, atol=1e-6)


def test_golden_ratio_search_negative_interval():
    f = lambda x: (x + 3) ** 2
    a, b = -5, -1
    root = golden_ratio_search(f, a, b)
    expected = -3.0
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_large_interval():
    f = lambda x: (x - 100) ** 2
    a, b = 90, 110
    root = golden_ratio_search(f, a, b)
    expected = 100.0
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_flat_function():
    f = lambda x: 0.0
    a, b = -10, 10
    root = golden_ratio_search(f, a, b)
    expected = 0.0
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_one_evaluation_per_iteration():
    calls = []

    def f(x):
        calls.append(x)
        return (x - 3) ** 2

    root = golden_ratio_search(f, 0, 7)
    assert np.isclose(root, 3.0, atol=1e-6)
    # 2 次初始评估 + 每次迭代 1 次；~log_phi(7 / 1e-8) = 43 次迭代。
    # 每次迭代重新评估两个内部点需要约 86 次调用。
    assert len(calls) <= 46


def test_golden_ratio_search_integer_bounds():
    f = lambda x: (x - 4) ** 2
    a, b = 2, 6
    root = golden_ratio_search(f, a, b)
    expected = 4.0
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_fractional_minimum():
    f = lambda x: (x - 2.5) ** 2
    a, b = 0, 5
    root = golden_ratio_search(f, a, b)
    expected = 2.5
    assert np.isclose(root, expected, atol=1e-8)


def test_golden_ratio_search_function_with_noise():
    np.random.seed(0)
    f = lambda x: (x - 3) ** 2 + np.random.normal(0, 1e-6)
    a, b = 0, 6
    root = golden_ratio_search(f, a, b)
    expected = 3.0
    # 7. f 中 1e-6 的噪声会使比较在 (x - 3)**2 ~ 1e-6 时失效，即 |x - 3| ~ 1e-3
    assert np.isclose(root, expected, atol=1e-2)
