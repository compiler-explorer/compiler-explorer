void maxArray(uniform double x[], uniform double y[]) {
    foreach (i = 0 ... 65536) {
        x[i] = max(x[i], y[i]);
    }
}
