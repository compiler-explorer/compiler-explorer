// Compile with O3 and gcc4.7 and -march=corei7-avx for Sandy Bridge
void maxArray(double* __restrict x, double* __restrict y) {
#if __GNUC_MINOR__ >= 7  // 4.7+
    x = static_cast<double*>(__builtin_assume_aligned(x, 64));
    y = static_cast<double*>(__builtin_assume_aligned(y, 64));
#endif
    for (int i = 0; i < 65536; i++) {
        x[i] = ((y[i] > x[i]) ? y[i] : x[i]);
    }
}
