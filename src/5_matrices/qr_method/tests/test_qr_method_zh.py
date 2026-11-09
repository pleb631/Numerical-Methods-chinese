# 本文件为 test_qr_method.py 的中文注释版本（代码逻辑与英文版完全一致）
# test_qr_method.py
import pytest
import numpy as np
from ..implementation.qr_method import (
    qr_decomposition,
    qr_algorithm,
    qr_algorithm_with_shifts,
)


def test_qr_decomposition_identity():
    A = np.eye(3)
    Q, R = qr_decomposition(A)
    assert np.allclose(Q @ R, A)
    assert np.allclose(Q, np.eye(3))
    assert np.allclose(R, np.eye(3))


def test_qr_decomposition_random():
    np.random.seed(0)
    A = np.random.rand(4, 4)
    Q, R = qr_decomposition(A)
    assert np.allclose(Q @ R, A, atol=1e-8)
    assert np.allclose(Q @ Q.T, np.eye(4), atol=1e-8)


def test_qr_algorithm_diagonal():
    A = np.diag([1, 2, 3])
    eigenvalues = qr_algorithm(A)
    expected = np.array([1, 2, 3])
    assert np.allclose(eigenvalues, expected, atol=1e-8)


def test_qr_algorithm_symmetric():
    A = np.array([[2, 1], [1, 2]], dtype=float)
    eigenvalues = qr_algorithm(A)
    expected = np.linalg.eigvalsh(A)
    assert np.allclose(np.sort(eigenvalues), np.sort(expected), atol=1e-8)


def test_qr_algorithm_non_symmetric():
    A = np.array([[0, 1], [-2, -3]], dtype=float)
    eigenvalues = qr_algorithm(A)
    expected = np.linalg.eigvals(A)
    assert np.allclose(np.sort(eigenvalues), np.sort(expected), atol=1e-6)


def test_qr_algorithm_with_shifts_diagonal():
    A = np.diag([4, 5, 6])
    eigenvalues = qr_algorithm_with_shifts(A)
    expected = np.array([4, 5, 6])
    assert np.allclose(eigenvalues, expected, atol=1e-8)


def test_qr_algorithm_with_shifts_symmetric():
    A = np.array([[3, 2], [2, 3]], dtype=float)
    eigenvalues = qr_algorithm_with_shifts(A)
    expected = np.linalg.eigvalsh(A)
    assert np.allclose(np.sort(eigenvalues), np.sort(expected), atol=1e-8)


def test_qr_algorithm_with_shifts_non_symmetric():
    A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
    eigenvalues = qr_algorithm_with_shifts(A)
    expected = np.linalg.eigvals(A)
    assert np.allclose(np.sort(eigenvalues), np.sort(expected), atol=1e-6)


def test_qr_algorithm_max_iterations():
    A = np.array([[1, 1], [0, 1]], dtype=float)
    with pytest.raises(ValueError):  # 1. 正确的错误类型
        qr_algorithm(A, max_iterations=1)


def test_qr_algorithm_with_shifts_max_iterations():
    A = np.array([[1, 1], [0, 1]], dtype=float)
    with pytest.raises(ValueError):  # 2. 正确的错误类型
        qr_algorithm_with_shifts(A, max_iterations=1)


def test_qr_algorithm_zero_matrix():
    A = np.zeros((3, 3))
    eigenvalues = qr_algorithm(A)
    expected = np.zeros(3)
    assert np.allclose(eigenvalues, expected, atol=1e-8)


def test_qr_algorithm_with_shifts_zero_matrix():
    A = np.zeros((4, 4))
    eigenvalues = qr_algorithm_with_shifts(A)
    expected = np.zeros(4)
    assert np.allclose(eigenvalues, expected, atol=1e-8)


def test_qr_algorithm_single_element():
    A = np.array([[5]], dtype=float)
    eigenvalues = qr_algorithm(A)
    expected = np.array([5])
    assert np.allclose(eigenvalues, expected, atol=1e-8)


def test_qr_algorithm_with_shifts_single_element():
    A = np.array([[7]], dtype=float)
    eigenvalues = qr_algorithm_with_shifts(A)
    expected = np.array([7])
    assert np.allclose(eigenvalues, expected, atol=1e-8)


def test_qr_algorithm_non_square():
    A = np.random.rand(3, 4)
    with pytest.raises(ValueError):
        qr_algorithm(A)


def test_qr_algorithm_with_shifts_non_square():
    A = np.random.rand(4, 5)
    with pytest.raises(ValueError):
        qr_algorithm_with_shifts(A)


@pytest.mark.parametrize("method", [qr_algorithm, qr_algorithm_with_shifts])
def test_qr_algorithm_complex_pair_not_at_top(method):
    # 3. 实特征值 3 占主导地位，因此复数对最终位于末尾的 2x2 块中
    A = np.array([[3, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    eigenvalues = method(A)
    expected = np.linalg.eigvals(A)
    assert np.allclose(np.sort_complex(eigenvalues), np.sort_complex(expected))


@pytest.mark.parametrize("method", [qr_algorithm, qr_algorithm_with_shifts])
def test_qr_algorithm_two_complex_pairs(method):
    A = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -2], [0, 0, 2, 0]], dtype=float
    )
    eigenvalues = method(A)
    expected = np.linalg.eigvals(A)
    assert np.allclose(np.sort_complex(eigenvalues), np.sort_complex(expected))


def test_qr_algorithm_with_shifts_wilkinson_shift_choice():
    # 具有特征值 1.1 和 2.9 的三角块：移位即为该值
    # 最接近角点入口（而非幅值较大的那个）
    from ..implementation.qr_method import _wilkinson_shift

    block = np.array([[2.9, 0.1], [0.0, 1.1]])
    assert np.isclose(_wilkinson_shift(block), 1.1)
    block = np.array([[1.1, 0.1], [0.0, 2.9]])
    assert np.isclose(_wilkinson_shift(block), 2.9)
    # 5. 复数对：没有实 Wilkinson 移位
    assert _wilkinson_shift(np.array([[0.0, -1.0], [1.0, 0.0]])) is None


@pytest.mark.parametrize("seed", range(25))
def test_qr_algorithm_with_shifts_random_nonsymmetric(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 13))
    A = rng.standard_normal((n, n))
    eigenvalues = qr_algorithm_with_shifts(A)
    expected = np.linalg.eigvals(A)
    assert np.allclose(
        np.sort_complex(eigenvalues), np.sort_complex(expected), atol=1e-6
    )


@pytest.mark.parametrize("seed", range(10))
def test_qr_algorithm_with_shifts_random_symmetric(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(6, 13))
    B = rng.standard_normal((n, n))
    A = B + B.T
    eigenvalues = qr_algorithm_with_shifts(A)
    assert np.allclose(eigenvalues, np.linalg.eigvalsh(A), atol=1e-6)
