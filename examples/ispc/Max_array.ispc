// Compile with --opt=force-aligned-memory to improve vectorization
// by assuming the input arrays are aligned. SSE, AVX and AVX-512
// targets will assume 16, 32 or 64 byte alignment respectively.

void maxArray(uniform double x[], uniform double y[]) {
    foreach (i = 0 ... 65536) {
        x[i] = max(x[i], y[i]);
    }
}
