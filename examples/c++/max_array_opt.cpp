// Compile with -O3 -march=native to see autovectorization
void maxArray(double* __restrict x, double* __restrict y) {
    // Alignment hints supported on GCC 4.7+ and any compiler
    // supporting the appropriate builtin (clang 3.6+).
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#if __GNUC__ > 4 \
        || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7) \
        || __has_builtin(__builtin_assume_aligned)
    x = static_cast<double*>(__builtin_assume_aligned(x, 64));
    y = static_cast<double*>(__builtin_assume_aligned(y, 64));
#endif
    for (int i = 0; i < 65536; i++) {
        x[i] = ((y[i] > x[i]) ? y[i] : x[i]);
    }
}
