import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B):
    N = 64
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)
    with T.Kernel(1, threads=64):
        for i in T.Parallel(N):
            C[i] = A[i] + B[i]
    return C
