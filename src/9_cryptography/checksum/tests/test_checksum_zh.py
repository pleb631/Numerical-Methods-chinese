# 本文件为 test_checksum.py 的中文注释版本（代码逻辑与英文版完全一致）
"""校验和算法测试——Luhn、CRC-32、Fletcher-16 和 Adler-32。

"""

import pytest

from ..implementation.checksum import (
    luhn_checksum,
    luhn_validate,
    crc32,
    fletcher16,
    adler32,
)


# ---------------------------------------------------------------------------
# luhn_checksum 测试
# ---------------------------------------------------------------------------


def test_luhn_checksum_single_digit():
    # 校验位应为 0（0 + 0 = 0 mod 10）。
    assert luhn_checksum("0") == 0


def test_luhn_checksum_known_card_number():
    # {'1': 'Visa 测试号码：4539578763621486'}
    # {'2': '负载为 "453957876362148"；校验位应为 6。'}
    assert luhn_checksum("453957876362148") == 6


def test_luhn_checksum_another_card_number():
    # MasterCard 测试号码：5500000000000004
    # 负载"550000000000000"；校验位应为 4。
    assert luhn_checksum("550000000000000") == 4


def test_luhn_checksum_amex():
    # 1. Amex 测试号：378282246310005
    # 2. 负载"37828224631000"；校验位应为 5。
    assert luhn_checksum("37828224631000") == 5


def test_luhn_checksum_type_error():
    with pytest.raises(TypeError, match="Expected a string"):
        luhn_checksum(12345)  # []


def test_luhn_checksum_empty_string():
    with pytest.raises(ValueError, match="must not be empty"):
        luhn_checksum("")


def test_luhn_checksum_non_digit():
    with pytest.raises(ValueError, match="only digits"):
        luhn_checksum("123a56")


# ---------------------------------------------------------------------------
# luhn_validate 测试
# ---------------------------------------------------------------------------


def test_luhn_validate_valid_visa():
    assert luhn_validate("4539578763621486") is True


def test_luhn_validate_valid_mastercard():
    assert luhn_validate("5500000000000004") is True


def test_luhn_validate_valid_amex():
    assert luhn_validate("378282246310005") is True


def test_luhn_validate_invalid_number():
    assert luhn_validate("4539578763621480") is False


def test_luhn_validate_another_invalid():
    assert luhn_validate("1234567890") is False


def test_luhn_validate_roundtrip():
    payload = "7992739871"
    check = luhn_checksum(payload)
    assert luhn_validate(payload + str(check)) is True


def test_luhn_validate_type_error():
    with pytest.raises(TypeError, match="Expected a string"):
        luhn_validate(12345)  # []


def test_luhn_validate_too_short():
    with pytest.raises(ValueError, match="at least two digits"):
        luhn_validate("5")


def test_luhn_validate_non_digit():
    with pytest.raises(ValueError, match="only digits"):
        luhn_validate("12ab")


# ---------------------------------------------------------------------------
# crc32 测试
# ---------------------------------------------------------------------------


def test_crc32_standard_check_value():
    # ASCII 字符串 "123456789" 的 CRC-32 校验和为 0xCBF43926。
    assert crc32(b"123456789") == 0xCBF43926


def test_crc32_empty():
    # 空数据的 CRC-32 校验和为 0x00000000。
    assert crc32(b"") == 0x00000000


def test_crc32_single_byte():
    # ["b"a"的 CRC-32 校验和为 0xE8B7BE43（与 zlib.crc32 相同）。"]
    assert crc32(b"a") == 0xE8B7BE43


def test_crc32_hello_world():
    # 已知字符串 b"Hello, World!"的 CRC-32 校验和
    assert crc32(b"Hello, World!") == 0xEC4AC3D0


def test_crc32_type_error():
    with pytest.raises(TypeError, match="Expected bytes"):
        crc32("not bytes")  # []


# ---------------------------------------------------------------------------
# fletcher16 测试
# ---------------------------------------------------------------------------


def test_fletcher16_simple():
    # 1. 手动计算的 Fletcher-16 校验和 [0x01, 0x02]：
    # 2. sum1: 0 -> 1 -> 3   (mod 255)
    # 3. sum2: 0 -> 1 -> 4   (mod 255)
    # 4. result: (4 << 8) | 3 = 0x0403 = 1027
    assert fletcher16(b"\x01\x02") == 0x0403


def test_fletcher16_empty():
    assert fletcher16(b"") == 0x0000


def test_fletcher16_single_byte():
    # 对于字节 0x01：sum1 = 1，sum2 = 1 → (1 << 8) | 1 = 0x0101
    assert fletcher16(b"\x01") == 0x0101


def test_fletcher16_abc():
    # abcde（字节 97、98、99、100、101）
    # 已知的 Fletcher-16 测试向量（Wikipedia）：0xC8F0
    assert fletcher16(b"abcde") == 0xC8F0


def test_fletcher16_type_error():
    with pytest.raises(TypeError, match="Expected bytes"):
        fletcher16("string")  # []


# ---------------------------------------------------------------------------
# adler32 测试
# ---------------------------------------------------------------------------


def test_adler32_wikipedia():
    # 已知测试向量：adler32(b"Wikipedia") == 0x11E60398
    assert adler32(b"Wikipedia") == 0x11E60398


def test_adler32_empty():
    # 空数据的 Adler-32：a=1, b=0 → (0 << 16) | 1 = 1
    assert adler32(b"") == 0x00000001


def test_adler32_single_a():
    # b"a" = [97]: a = (1 + 97) % 65521 = 98, b = (0 + 98) % 65521 = 98
    # result = (98 << 16) | 98 = 0x00620062
    assert adler32(b"a") == 0x00620062


def test_adler32_abc():
    # 已知：adler32(b"abc") = 0x024d0127
    assert adler32(b"abc") == 0x024D0127


def test_adler32_type_error():
    with pytest.raises(TypeError, match="Expected bytes"):
        adler32(123)  # type: ignore[arg-type]
