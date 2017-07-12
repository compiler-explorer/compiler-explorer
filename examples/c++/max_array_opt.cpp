// Compile with -O3 -march=native to see autovectorization
typedef double *__attribute__((aligned(64))) aligned_double;

void maxArray(aligned_double __restrict x, aligned_double __restrict y) {
    for (int i = 0; i < 65536; i++) {
        x[i] = ((y[i] > x[i]) ? y[i] : x[i]);
    }
}
