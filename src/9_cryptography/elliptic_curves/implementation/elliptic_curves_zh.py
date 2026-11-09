# 本文件为 elliptic_curves.py 的中文注释版本（代码逻辑与英文版完全一致）
"""素有限域上的椭圆曲线运算

提供形如 y² ≡ x³ + ax + b (mod p) 的椭圆曲线的核心运算，其中 *p* 是大于 3 的素数（短魏尔斯特拉斯形式仅在特征不为 2 和 3 时有效），以及一个简单的 ECDH 密钥交换辅助函数。

群运算函数接受曲线系数 *a*，但不接受 *b*（公式中不需要它）。传递可选关键字 ``b`` 可在计算前对曲线和输入点进行验证。

点表示为 ``(x, y)`` 元组。*无穷远点*（单位元）表示为 ``None``。








"""

from typing import Optional, Tuple

# 点要么是 (x, y) 元组，要么是 None（无穷远点）。
Point = Optional[Tuple[int, int]]


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------


def _validate_field(p: int) -> None:
    """除非 *p* 是短 Weierstrass 形式的有效特征（p > 3），否则抛出异常。假设 *p* 为素数，不进行测试。

异常：
    ValueError：如果 *p* 不大于 3。

"""
    if p <= 3:
        raise ValueError(f"Prime p must be > 3, got {p}.")


def _validate_inputs(
    points: Tuple[Point, ...], a: int, b: Optional[int], p: int
) -> None:
    """验证字段，并在提供 *b* 时验证曲线和点。"""
    _validate_field(p)
    if b is not None:
        _validate_curve(a, b, p)
        for point in points:
            _validate_point_on_curve(point, a, b, p)


def _validate_curve(a: int, b: int, p: int) -> None:
    """验证曲线参数是否定义了一个非奇异曲线。

    参数：
        a: 曲线的系数 *a*。
        b: 曲线的系数 *b*。
        p: 有限域的素数模数。

    异常：
        ValueError: 如果 *p* 不大于 3 或曲线是奇异的。
"""
    _validate_field(p)

    discriminant = (4 * pow(a, 3) + 27 * pow(b, 2)) % p
    if discriminant == 0:
        raise ValueError(
            f"Singular curve: 4a³ + 27b² ≡ 0 (mod {p}). " "Choose different a, b, or p."
        )


def _validate_point_on_curve(point: Point, a: int, b: int, p: int) -> None:
    """如果 *point* 不在曲线上则抛出异常。

    参数：
        point：要验证的点（始终接受 ``None``）。
        a：曲线的系数 *a*。
        b：曲线的系数 *b*。
        p：有限域的素数模数。

    异常：
        ValueError：如果点不在曲线上。
"""
    if point is not None and not is_on_curve(point, a, b, p):
        raise ValueError(f"Point {point} is not on the curve.")


# ------------------------------------------------------------------
# 公共 API
# ------------------------------------------------------------------


def is_on_curve(point: Point, a: int, b: int, p: int) -> bool:
    """检查点 *point* 是否在曲线 y² ≡ x³ + ax + b (mod p) 上。

无穷远点（``None``）始终被认为位于每条曲线上。

参数:
    point: 表示无穷远点的 ``(x, y)`` 元组或 ``None``。
    a: 曲线方程的系数 *a*。
    b: 曲线方程的系数 *b*。
    p: 定义有限域的素数模数。

返回:
    如果点满足曲线方程，则返回 ``True``，否则返回 ``False``。


"""
    if point is None:
        return True

    x, y = point
    lhs = pow(y, 2, p)
    rhs = (pow(x, 3, p) + a * x + b) % p
    return lhs == rhs


def point_add(P: Point, Q: Point, a: int, p: int, b: Optional[int] = None) -> Point:
    """[
    "在椭圆曲线上添加两个点。",
    "处理以下特殊情况：",
    "* 任一操作数为 ``None``（无穷远点/单位元）。",
    "* ``P == Q`` —— 委托给 :func:`point_double`。",
    "* ``P`` 和 ``Q`` 具有相同的 *x* 但不同的 *y*（逆点）—— 返回 ``None``。",
    "参数：",
    "    P: 第一个点（或 ``None``）。",
    "    Q: 第二个点（或 ``None``）。",
    "    a: 曲线方程的系数 *a*。",
    "    p: 定义有限域的素数模数。",
    "    b: 可选系数 *b*；如果给出，将验证曲线和两个点。",
    "返回：",
    "    曲线上的和 ``P + Q``，或无穷远点的 ``None``。",
    "异常：",
    "    ValueError: 如果 ``p <= 3``，或（当给出 *b* 时）曲线是奇异的或点不在曲线上。"
]







"""
    _validate_inputs((P, Q), a, b, p)
    if P is None:
        return Q
    if Q is None:
        return P

    x_p, y_p = P
    x_q, y_q = Q

    if x_p % p == x_q % p:
        if y_p % p == y_q % p:
            return point_double(P, a, p)
        # P 和 Q 互为逆矩阵
        return None

    # 割线斜率
    lam = ((y_q - y_p) * pow(x_q - x_p, -1, p)) % p

    x_r = (lam * lam - x_p - x_q) % p
    y_r = (lam * (x_p - x_r) - y_p) % p
    return (x_r, y_r)


def point_double(P: Point, a: int, p: int, b: Optional[int] = None) -> Point:
    """对椭圆曲线上的点进行倍点运算。

如果 *P* 是无穷远点或 *y = 0*，则结果为
``None``。

参数：
    P: 要倍点的点（或 ``None``）。
    a: 曲线方程中的系数 *a*。
    p: 定义有限域的素模数。
    b: 可选系数 *b*；如果给出，将验证曲线和点。

返回：
    曲线上的点 ``2P``，或对于无穷远点返回 ``None``。

异常：
    ValueError: 如果 ``p <= 3``，或（当给出 *b* 时）曲线是奇异的或点不在曲线上。



"""
    _validate_inputs((P,), a, b, p)
    if P is None:
        return None

    x_p, y_p = P

    if y_p % p == 0:
        return None

    lam = ((3 * x_p * x_p + a) * pow(2 * y_p, -1, p)) % p

    x_r = (lam * lam - 2 * x_p) % p
    y_r = (lam * (x_p - x_r) - y_p) % p
    return (x_r, y_r)


def scalar_multiply(k: int, P: Point, a: int, p: int, b: Optional[int] = None) -> Point:
    """["使用双加算法计算 ``k * P``。

    参数：
        k: 非负整数标量。
        P: 曲线上的点（或 ``None``）。
        a: 曲线方程的系数 *a*。
        p: 定义有限域的素数模数。
        b: 可选系数 *b*；如果给出，将验证曲线和点。

    返回：
        点 ``k * P``，或无穷远点 ``None``。

    异常：
        ValueError: 如果 *k* 为负数。
        ValueError: 如果 ``p <= 3``，或（当给出 *b* 时）曲线奇异或点不在曲线上。"]


"""
    if k < 0:
        raise ValueError(f"Scalar k must be non-negative, got {k}.")
    _validate_inputs((P,), a, b, p)

    if k == 0 or P is None:
        return None

    result: Point = None
    addend: Point = P

    while k > 0:
        if k & 1:
            result = point_add(result, addend, a, p)
        addend = point_double(addend, a, p)
        k >>= 1

    return result


def ecdh_shared_secret(
    private_key: int,
    public_key: Point,
    a: int,
    p: int,
    b: Optional[int] = None,
) -> Point:
    """计算 ECDH 共享密钥。

共享密钥是点 ``private_key * public_key``。

参数：
    private_key: 本方的私标量（正整数）。
    public_key: 曲线上对方的公钥点。
    a: 曲线方程的系数 *a*。
    p: 定义有限域的素数模数。
    b: 可选系数 *b*；如果提供，将验证曲线和公钥（此操作可拒绝无效曲线攻击）。

返回：
    曲线上的共享密钥点。

异常：
    ValueError: 如果 *private_key* 不是正整数。
    ValueError: 如果 ``p <= 3``，或者（当提供 *b* 时）曲线是奇异的或公钥不在曲线上。


"""
    if private_key <= 0:
        raise ValueError(f"Private key must be a positive integer, got {private_key}.")
    return scalar_multiply(private_key, public_key, a, p, b)
