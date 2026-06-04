import cutlass
import cutlass.cute as cute


@cute.kernel
def hello_world_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello world from CuTe DSL")


@cute.jit
def hello_world():
    hello_world_kernel().launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


cute.compile(hello_world)
