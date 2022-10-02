void maxArray(double* x, double* y) {
    int i;

    for (i = 0; i < 65536; i++) {
        if (y[i] > x[i]) x[i] = y[i];
    }
}
