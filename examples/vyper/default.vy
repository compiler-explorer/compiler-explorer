@external
def power(base: int128, exponent: uint128) -> int128:
    result: int128 = 1
    temp_base: int128 = base
    temp_exponent: uint128 = exponent

    # Exponentiation by squaring
    for i: uint256 in range(256):
        if temp_exponent == 0:
            break
        if temp_exponent % 2 == 1:
            result *= temp_base
        temp_base *= temp_base
        temp_exponent //= 2

    return result
