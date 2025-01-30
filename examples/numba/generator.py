import itertools

import numba


@numba.njit(locals={"x": numba.uint64})
def xorshift(x):
    while True:
        x ^= x >> 13
        x ^= x << 7
        x ^= x >> 17
        yield x


RNG = xorshift(1)

if __name__ == "__main__":
    for x in itertools.islice(RNG, 16):
        print(f"{x:016x}")
