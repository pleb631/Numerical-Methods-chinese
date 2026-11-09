# 本文件为 example.py 的中文注释版本（代码逻辑与英文版完全一致）
"""Shamir 秘密共享 — 示例

演示如何使用有限域上的拉格朗日求值将秘密整数分割为多个份额，
然后从这些份额的子集中重构秘密。


"""

from ..implementation.lagrange_evaluation import (
    generate_shares,
    reconstruct_secret,
)


def main() -> None:
    secret = 42
    prime = 97  # 必须大于秘密
    num_shares = 5
    threshold = 3  # 重构所需的最小份额数

    print(f"Original secret : {secret}")
    print(f"Prime modulus   : {prime}")
    print(f"Total shares    : {num_shares}")
    print(f"Threshold       : {threshold}")
    print()

    shares = generate_shares(secret, num_shares, threshold, prime)
    print("Generated shares:")
    for x, y in shares:
        print(f"  Share (x={x}, y={y})")
    print()

    # 使用恰好阈数值量的份额进行重构
    subset = shares[:threshold]
    recovered = reconstruct_secret(subset, prime)
    print(f"Reconstructed secret from {threshold} shares: {recovered}")
    assert recovered == secret, "Reconstruction failed!"

    # 使用所有份额进行重构
    recovered_all = reconstruct_secret(shares, prime)
    print(f"Reconstructed secret from all {num_shares} shares: {recovered_all}")
    assert recovered_all == secret, "Reconstruction failed!"

    print("\nSuccess — the secret was correctly recovered.")


if __name__ == "__main__":
    main()
