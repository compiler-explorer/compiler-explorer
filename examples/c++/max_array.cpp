void maxArray(double* x, double* y) {
    for (int i = 0; i < 65536; i++) {
        if (y[i] > x[i]) x[i] = y[i];
    }
}
