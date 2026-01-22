void maxArray(double* x, const double* y) {
    for (int i = 0; i < 65536; i++) {
        if (y[i] > x[i]) x[i] = y[i];
    }
}
