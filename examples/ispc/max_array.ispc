void maxArray(uniform double x[], uniform double y[]) {
    foreach (i = 0 ... 65535) {
        x[i] = max(x[i], y[i]);
    }
}
