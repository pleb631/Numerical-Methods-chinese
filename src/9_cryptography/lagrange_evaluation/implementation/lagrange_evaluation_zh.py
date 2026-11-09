# 本文件为 lagrange_evaluation.py 的中文注释版本（代码逻辑与英文版完全一致）
import secrets
from typing import List, Optional, Tuple, Union

import numpy as np


def lagrange_basis(
    i: int,
    x_values: Union[List[float], np.ndarray],
    x: Union[float, int],
    modulus: Optional[int] = None,
) -> Union[float, int]:
    """计算第 i 个拉格朗日基多项式在 x 处的值。
    
第 i 个基多项式定义为：
        L_i(x) = 对所有 j != i 的 (x - x_j) / (x_i - x_j) 的乘积
    
如果提供了模数，所有算术运算均在有限域 Z/模数中进行（使用模逆元，用于 Shamir 秘密共享）。
    
    参数：
    i: 基多项式的索引。
        x_values: 数据点的 x 坐标数组。
        x: 用于评估基多项式的点。
        modulus: 用于有限域算术的可选素数模数。
    
    返回：
    第 i 个拉格朗日基多项式在 x 处的值。
    
    异常：
    ValueError: 如果 i 超出范围或 x_values 包含重复条目。
    
    
"""
    x_values = np.asarray(x_values)

    if i < 0 or i >= len(x_values):
        raise ValueError(
            f"Index i={i} is out of range for {len(x_values)} data points."
        )

    if len(x_values) != len(set(x_values.tolist())):
        raise ValueError("x_values must contain unique values.")

    if modulus is not None and len({int(v) % modulus for v in x_values}) != len(
        x_values
    ):
        raise ValueError("x_values must be distinct modulo the modulus.")

    result: Union[float, int] = 1

    for j in range(len(x_values)):
        if j == i:
            continue
        if modulus is not None:
            numerator = x - int(x_values[j])
            denominator = int(x_values[i]) - int(x_values[j])
            numerator = numerator % modulus
            denominator = denominator % modulus
            result = (
                result * numerator * pow(denominator, modulus - 2, modulus)
            ) % modulus
        else:
            numerator = x - x_values[j]
            denominator = x_values[i] - x_values[j]
            result = result * numerator / denominator

    return result


def lagrange_interpolate(
    x_data: Union[List[float], np.ndarray],
    y_data: Union[List[float], np.ndarray],
    x: Union[float, int],
    modulus: Optional[int] = None,
) -> Union[float, int]:
    """在点 x 处评估拉格朗日插值多项式。
    
构造一个次数不超过 n-1 的唯一多项式，使其通过所有 n 个数据点，并在 x 处评估该多项式。当提供模数时，计算在有限域 Z/模数中进行，这是 Shamir 秘密共享方案中用于重构秘密的核心操作。
    
    参数：
        x_data: 数据点的 x 坐标数组。
        y_data: 数据点的 y 坐标数组。
        x: 用于评估插值多项式的点。
    modulus: 用于有限域算术的可选素数模数。
    
    返回：
        插值多项式在 x 处的值。
    
    异常：
    ValueError: 如果 x_data 和 y_data 长度不同、为空或包含重复的 x 值。
    
    

    
    
    
"""
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    if len(x_data) != len(y_data):
        raise ValueError(
            f"x_data (length {len(x_data)}) and y_data (length {len(y_data)}) "
            "must have the same length."
        )

    if len(x_data) == 0:
        raise ValueError("At least one data point is required.")

    if len(x_data) != len(set(x_data.tolist())):
        raise ValueError("x_data must contain unique values.")

    if modulus is not None and len({int(v) % modulus for v in x_data}) != len(x_data):
        raise ValueError("x_data must be distinct modulo the modulus.")

    result: Union[float, int] = 0

    for i in range(len(x_data)):
        basis = lagrange_basis(i, x_data, x, modulus)
        yi = int(y_data[i]) if modulus is not None else y_data[i]

        if modulus is not None:
            result = (result + yi * basis) % modulus
        else:
            result = result + yi * basis

    return result


def generate_shares(
    secret: int,
    num_shares: int,
    threshold: int,
    prime: int,
) -> List[Tuple[int, int]]:
    """为 Shamir 秘密共享生成份额。
    
构造一个次数为 (阈值 - 1) 的随机多项式，其常数项等于秘密。每个份额是一个点 (x, f(x))，在有限域 Z/素数上进行评估。
    
    参数：
        secret: 要共享的整数秘密（必须 >= 0）。
    num_shares: 要生成的份额总数。
        threshold: 重构秘密所需的最小份额数。
        prime: 严格大于秘密的素数。
    
    返回：
        代表份额的 (x, y) 元组列表。

    异常：
        ValueError: 如果参数无效。

    
    
"""
    if secret < 0:
        raise ValueError("Secret must be a non-negative integer.")

    if threshold < 1:
        raise ValueError("Threshold must be at least 1.")

    if threshold > num_shares:
        raise ValueError(
            f"Threshold ({threshold}) must be <= num_shares ({num_shares})."
        )

    if num_shares >= prime:
        raise ValueError(
            f"num_shares ({num_shares}) must be less than prime ({prime}); "
            "otherwise share x-coordinates repeat modulo the prime or hit x = 0."
        )

    if prime <= secret:
        raise ValueError(
            f"Prime ({prime}) must be strictly greater than the secret ({secret})."
        )

    # 构建一个次数为 (阈值 - 1) 的随机多项式，其常数项等于秘密
    coefficients = [secret] + [secrets.randbelow(prime) for _ in range(threshold - 1)]

    shares: List[Tuple[int, int]] = []
    for i in range(1, num_shares + 1):
        y = 0
        for power, coeff in enumerate(coefficients):
            y = (y + coeff * pow(i, power, prime)) % prime
        shares.append((i, y))

    return shares


def reconstruct_secret(
    shares: List[Tuple[int, int]],
    prime: int,
) -> int:
    """使用在 x=0 处的拉格朗日插值从份额重构秘密。
    
给定由 ``generate_shares`` 生成的至少 *threshold* 个份额，此函数通过在有限域 Z/素数上在 x = 0 处评估拉格朗日插值多项式来恢复原始秘密。
    
    参数：
        shares: (x, y) 元组列表（至少 *threshold* 个份额）。
    prime: 份额生成期间使用的素数模数。
    
    返回：
        重构的秘密（整数）。

    异常：
        ValueError: 如果份额列表为空。

    
    
"""
    if not shares:
        raise ValueError("At least one share is required.")

    x_data = [s[0] for s in shares]
    y_data = [s[1] for s in shares]

    return int(lagrange_interpolate(x_data, y_data, 0, modulus=prime))
