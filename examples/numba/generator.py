import numba


@numba.njit(locals={"x": numba.uint64})
def xorshift64(x):
    while True:
        x ^= x >> 13
        x ^= x << 7
        x ^= x >> 17
        yield x


rng = xorshift64(1)

if __name__ == "__main__":
    for _ in range(16):
        print(f"{next(rng):016x}")
