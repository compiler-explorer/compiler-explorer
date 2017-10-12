uniform int sumArray(uniform int input[], uniform int length) {
    varying int sum = 0;
    foreach (i = 0 ... length) {
        sum += input[i];
    }
    return reduce_add(sum);
}
