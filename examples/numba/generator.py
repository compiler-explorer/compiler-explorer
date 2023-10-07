import numba


@numba.njit(locals={"x": numba.uint32})
def xorshift32(x):
    while True:
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        yield x


rng = xorshift32(4)
for i in range(10):
    print(f"{next(rng):08x}")
