# 本文件为 checksum.py 的中文注释版本（代码逻辑与英文版完全一致）
"""校验和算法

常见校验和算法的纯 Python 实现：
Luhn、CRC-32、Fletcher-16 和 Adler-32。

"""

from typing import List

MOD_ADLER: int = 65521  # 小于 2^16 的最大质数


# ---------------------------------------------------------------------------
# Luhn 算法
# ---------------------------------------------------------------------------


def luhn_checksum(number_str: str) -> int:
    """计算数字字符串的 Luhn 校验位。

校验位的选择使得将其追加到 *number_str* 后生成的字符串能通过 Luhn 验证。

参数：
    number_str：由 ASCII 数字组成的非空字符串。

返回：
    校验位（0–9 范围内的整数）。

异常：
    TypeError：如果 *number_str* 不是字符串。
    ValueError：如果 *number_str* 为空或包含非数字字符。

"""

    if not isinstance(number_str, str):
        raise TypeError(f"Expected a string, got {type(number_str).__name__}.")
    if len(number_str) == 0:
        raise ValueError("Input string must not be empty.")
    if not number_str.isdigit():
        raise ValueError(
            f"Input string must contain only digits, got '{number_str}'."
        )

    digits: List[int] = [int(ch) for ch in number_str]

    # 将右侧每第二个数字（即载荷数字）翻倍。
    # 由于我们正在计算校验位，现有字符串中的每个数字都从最右侧的一个开始处理。
    # 
    total: int = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return (10 - (total % 10)) % 10


def luhn_validate(number_str: str) -> bool:
    """使用 Luhn 算法验证数字字符串。

*number_str* 的最后一位被视为校验位。

参数：
    number_str：至少包含两个数字的字符串。

返回：
    如果字符串通过 Luhn 校验则返回 ``True``，否则返回 ``False``。

异常：
    TypeError：如果 *number_str* 不是字符串。
    ValueError：如果 *number_str* 少于两个字符或包含非数字字符。

"""

    if not isinstance(number_str, str):
        raise TypeError(f"Expected a string, got {type(number_str).__name__}.")
    if len(number_str) < 2:
        raise ValueError("Input string must have at least two digits.")
    if not number_str.isdigit():
        raise ValueError(
            f"Input string must contain only digits, got '{number_str}'."
        )

    digits: List[int] = [int(ch) for ch in number_str]

    total: int = 0
    # 从最右边的数字开始处理；校验位（从右数第 0 位）
    # 不翻倍，下一个翻倍，交替进行。
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return total % 10 == 0


# ---------------------------------------------------------------------------
# CRC-32
# ---------------------------------------------------------------------------


def crc32(data: bytes, polynomial: int = 0xEDB88320) -> int:
    """计算 *data* 的 CRC-32 校验和。

使用带有反射多项式的标准逐位算法。

参数：
    data：要计算校验和的字节对象。
    polynomial：反射的 CRC-32 多项式。默认值
        ``0xEDB88320`` 对应于以太网、gzip、PNG 等使用的标准 CRC-32。

返回：
    作为无符号 32 位整数的 CRC-32 值。

异常：
    TypeError：如果 *data* 不是 ``bytes`` 对象。

"""

    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}.")

    # 构建一个 256 条目的查找表以提高速度。
    table: List[int] = []
    for byte in range(256):
        crc: int = byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ polynomial
            else:
                crc >>= 1
        table.append(crc)

    crc = 0xFFFFFFFF
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)

    return crc ^ 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Fletcher-16
# ---------------------------------------------------------------------------


def fletcher16(data: bytes) -> int:
    """计算 *data* 的 Fletcher-16 校验和。

参数：
    data：要计算校验和的字节对象。

返回：
    作为 16 位无符号整数的 Fletcher-16 值。

异常：
    TypeError：如果 *data* 不是 ``bytes`` 对象。
"""

    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}.")

    sum1: int = 0
    sum2: int = 0

    for byte in data:
        sum1 = (sum1 + byte) % 255
        sum2 = (sum2 + sum1) % 255

    return (sum2 << 8) | sum1


# ---------------------------------------------------------------------------
# Adler-32
# ---------------------------------------------------------------------------


def adler32(data: bytes) -> int:
    """计算 *data* 的 Adler-32 校验和。

参数：
    data：要计算校验和的字节对象。

返回：
    作为无符号 32 位整数的 Adler-32 值。

异常：
    TypeError：如果 *data* 不是 ``bytes`` 对象。
"""

    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}.")

    a: int = 1
    b: int = 0

    for byte in data:
        a = (a + byte) % MOD_ADLER
        b = (b + a) % MOD_ADLER

    return (b << 16) | a
