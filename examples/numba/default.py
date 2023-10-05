import numba


@numba.njit("int32(int32)")
def square(num):
    return num * num
