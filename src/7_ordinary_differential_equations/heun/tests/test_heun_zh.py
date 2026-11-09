# 本文件为 test_heun.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_heun_solver.py
import pytest
import numpy as np
from ..implementation.heun import heun_method


def test_heun_method_linear_ode():
    f = lambda t, y: y
    t0 = 0.0
    y0 = np.array([1.0])
    t_end = 1.0
    h = 0.1
    t, y = heun_method(f, t0, y0, t_end, h)
    # y' = y：海恩法给出的结果为 (1 + h + h^2 / 2)^n
    assert y.shape == (11,)
    assert np.allclose(y, (1 + h + h * h / 2) ** np.arange(11), rtol=0, atol=1e-12)
    # 二阶误差：在 t = 1 处观察到 4.2e-3（约为 h^2 * e / 6）
    assert np.allclose(y, np.exp(t), rtol=0, atol=4.3e-3)


def test_heun_method_constant_ode():
    f = lambda t, y: np.array([2.0])
    t0 = 0.0
    y0 = np.array([0.0])
    t_end = 1.0
    h = 0.2
    t, y = heun_method(f, t0, y0, t_end, h)
    expected = 2.0 * t
    assert np.allclose(y, expected, rtol=0, atol=1e-12)


def test_heun_method_system_ode():
    def f(t, y):
        dy1 = y[1]
        dy2 = -y[0]
        return np.array([dy1, dy2])

    t0 = 0.0
    y0 = np.array([0.0, 1.0])
    t_end = 2 * np.pi
    h = 0.1
    t, y = heun_method(f, t0, y0, t_end, h)
    assert y.shape == (len(t), 2)
    # 1. w = y2 + i*y1 满足 w' = i*w，因此海恩法给出 w_n = (1 + i*h - h^2/2)^n
    # 2. (已在完整步骤中验证；最后一步已缩短)
    w = (1 + 1j * h - h * h / 2) ** np.arange(len(t) - 1)
    assert np.allclose(y[:-1, 0], w.imag, rtol=0, atol=1e-12)
    assert np.allclose(y[:-1, 1], w.real, rtol=0, atol=1e-12)
    # 二阶误差：观察到 1.0e-2 / 8.2e-3
    assert np.allclose(y[:, 0], np.sin(t), rtol=0, atol=1.1e-2)
    assert np.allclose(y[:, 1], np.cos(t), rtol=0, atol=9e-3)


def test_heun_method_zero_step_size():
    f = lambda t, y: y
    t0 = 0.0
    y0 = np.array([1.0])
    t_end = 1.0
    h = 0.0
    with pytest.raises(ValueError):
        heun_method(f, t0, y0, t_end, h)


def test_heun_method_negative_step_size():
    f = lambda t, y: y
    t0 = 0.0
    y0 = np.array([1.0])
    t_end = 1.0
    h = -0.1
    with pytest.raises(ValueError):
        heun_method(f, t0, y0, t_end, h)


def test_heun_method_t_end_less_than_t0():
    f = lambda t, y: y
    t0 = 1.0
    y0 = np.array([1.0])
    t_end = 0.0
    h = 0.1
    with pytest.raises(ValueError):
        heun_method(f, t0, y0, t_end, h)


def test_heun_method_single_step():
    f = lambda t, y: np.array([3.0])
    t0 = 0.0
    y0 = np.array([2.0])
    t_end = 0.1
    h = 0.1
    t, y = heun_method(f, t0, y0, t_end, h)
    expected_y = np.array([2.0, 2.0 + (3.0 + 3.0) / 2 * 0.1])
    assert np.allclose(y, expected_y, rtol=0, atol=1e-12)


def test_heun_method_multiple_steps():
    f = lambda t, y: np.array([t])
    t0 = 0.0
    y0 = np.array([0.0])
    t_end = 1.0
    h = 0.25
    t, y = heun_method(f, t0, y0, t_end, h)
    expected_y = np.array(
        [
            0.0,
            0.0 + (0.0 + 0.25) / 2 * 0.25,
            (0.0625) + (0.25 + 0.5) / 2 * 0.25,
            (0.15625) + (0.5 + 0.75) / 2 * 0.25,
            (0.3125) + (0.75 + 1.0) / 2 * 0.25,
        ]
    )
    # 梯形法则精确积分 y' = t：y = t^2 / 2
    assert np.allclose(y, t**2 / 2, rtol=0, atol=1e-12)
    assert np.allclose(y, [0.0, 0.03125, 0.125, 0.28125, 0.5], rtol=0, atol=1e-12)


def test_heun_method_high_precision():
    f = lambda t, y: y
    t0 = 0.0
    y0 = np.array([1.0])
    t_end = 1.0
    h = 0.01
    t, y = heun_method(f, t0, y0, t_end, h)
    expected = np.exp(t)
    # 二阶误差：观察到 4.5e-5，比 h = 0.1 时小 100 倍
    assert np.allclose(y, expected, rtol=0, atol=5e-5)


def test_heun_method_large_steps():
    f = lambda t, y: y
    t0 = 0.0
    y0 = np.array([1.0])
    t_end = 1.0
    h = 0.5
    t, y = heun_method(f, t0, y0, t_end, h)
    # R(0.5) = 1 + 0.5 + 0.125 = 1.625, so y = [1, 1.625, 1.625^2]
    assert np.allclose(y, [1.0, 1.625, 1.625**2], rtol=0, atol=1e-12)
    assert np.allclose(y, np.exp(t), rtol=0, atol=7.8e-2)


def _linear_system_exact(t):
    # y' = [[1, 1], [1, -1]] y，y(0) = [1, 0]；A^2 = 2I，因此 exp(At) 有闭式解
    s2 = np.sqrt(2.0)
    return np.cosh(s2 * t) + np.sinh(s2 * t) / s2, np.sinh(s2 * t) / s2


def test_heun_method_vector_valued_ode():
    def f(t, y):
        return np.array([y[0] + y[1], y[0] - y[1]])

    t0 = 0.0
    y0 = np.array([1.0, 0.0])
    t_end = 1.0
    h = 0.1
    t, y = heun_method(f, t0, y0, t_end, h)
    assert y.shape == (11, 2)
    expected_y1, expected_y2 = _linear_system_exact(t)

    # 二阶误差：观察到 1.5e-2 / 6.6e-3
    assert np.allclose(
        y[:, 0], expected_y1, rtol=0, atol=1.5e-2
    ), "y1 does not match expected values"
    assert np.allclose(
        y[:, 1], expected_y2, rtol=0, atol=7e-3
    ), "y2 does not match expected values"


def test_heun_method_non_linear_ode():
    def f(t, y):
        return np.array([y[0] ** 2])

    t0 = 0.0
    y0 = np.array([1.0])
    t_end = 0.5
    h = 0.1
    t, y = heun_method(f, t0, y0, t_end, h)
    expected = 1 / (1 - t)
    # 二阶误差：在 t = 1 附近的爆破处观察到 1.7e-2
    assert np.allclose(y, expected, rtol=0, atol=1.7e-2)


def test_heun_method_high_dimension():
    def f(t, y):
        return np.array([y[1], -y[0]])

    t0 = 0.0
    y0 = np.array([0.0, 1.0])
    t_end = 2 * np.pi
    h = 0.01
    t, y = heun_method(f, t0, y0, t_end, h)
    # 二阶误差：比 h = 0.1 时小 100 倍（1.0e-2 / 8.2e-3）
    assert np.allclose(y[:, 0], np.sin(t), rtol=0, atol=1.1e-4)
    assert np.allclose(y[:, 1], np.cos(t), rtol=0, atol=9e-5)


def test_heun_method_non_convergent():
    def f(t, y):
        return np.array([1 - y[0]])

    t0 = 0.0
    y0 = np.array([0.0])
    t_end = 1.0
    h = 0.1
    # 海恩法是显式的，且应对此简单 ODE 收敛
    t, y = heun_method(f, t0, y0, t_end, h)
    expected = 1 - np.exp(-t)
    # 二阶误差：观察到 6.6e-4
    assert np.allclose(y, expected, rtol=0, atol=7e-4)


def test_heun_method_last_step_is_shortened():
    f = lambda t, y: np.array([1.0])
    t, y = heun_method(f, 0.0, np.array([0.0]), 1.0, 0.3)
    assert t[-1] == 1.0
    assert np.allclose(t, [0.0, 0.3, 0.6, 0.9, 1.0])
    assert np.allclose(np.diff(t)[:-1], 0.3)
    # y' = 1 被精确积分，因此返回网格上的 y 必须等于 t
    assert np.allclose(y, t, rtol=0, atol=1e-12)


def test_heun_method_last_step_value():
    # 先进行一步 0.5 的完整步长，再进行一步 0.25 的步长，每一步均乘以 1 + h + h^2/2
    f = lambda t, y: y
    t, y = heun_method(f, 0.0, np.array([1.0]), 0.75, 0.5)
    assert np.allclose(t, [0.0, 0.5, 0.75])
    expected = [1.0, 1.625, 1.625 * (1 + 0.25 + 0.25**2 / 2)]
    assert np.allclose(y, expected, rtol=0, atol=1e-12)


def test_heun_method_t_end_equals_t0():
    f = lambda t, y: y
    t, y = heun_method(f, 1.0, np.array([2.0]), 1.0, 0.1)
    assert np.allclose(t, [1.0])
    assert y.shape == (1,)


def test_heun_method_output_shapes():
    t, y = heun_method(lambda t, y: -y, 0.0, np.array([1.0]), 1.0, 0.25)
    assert y.shape == (5,)
    f_vec = lambda t, y: np.array([y[1], -y[0]])
    t, y = heun_method(f_vec, 0.0, np.array([0.0, 1.0]), 1.0, 0.25)
    assert y.shape == (5, 2)


def test_heun_method_second_order_convergence():
    f = lambda t, y: y
    errs = []
    for h in (0.1, 0.05):
        t, y = heun_method(f, 0.0, np.array([1.0]), 1.0, h)
        errs.append(abs(y[-1] - np.e))
    assert np.log2(errs[0] / errs[1]) == pytest.approx(2.0, abs=0.2)


def test_heun_method_step_count_with_float_ratio():
    # 0.07 / 0.01 在浮点数中等于 7.000000000000001；步数仍应为 7
    f = lambda t, y: np.array([0.0])
    t, y = heun_method(f, 0.0, np.array([0.0]), 0.07, 0.01)
    assert len(t) == 8
    assert t[-1] == 0.07
