// Hint: Try to compile with -O3 -release -boundscheck=off -mcpu=native
int testFunction(int[] input) {
    int sum = 0;
    foreach (elem; input) {
        sum += elem;
    }
    return sum;
}