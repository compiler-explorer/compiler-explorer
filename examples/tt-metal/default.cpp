// TT-Metalium kernel example
// Compile with the Tenstorrent LLVM fork targeting RISC-V

#include <cstdint>

extern "C" {
    __attribute__((noinline))
    int add_kernel(int a, int b) {
        return a + b;
    }

    __attribute__((noinline))
    void vec_add(int* __restrict__ dst,
                 const int* __restrict__ src_a,
                 const int* __restrict__ src_b,
                 int n) {
        for (int i = 0; i < n; i++) {
            dst[i] = src_a[i] + src_b[i];
        }
    }

    __attribute__((noinline))
    int dot_product(const int* a, const int* b, int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
