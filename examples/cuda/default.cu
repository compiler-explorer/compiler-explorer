// Type your code here, or load an example.
__global__ void square(int *array, int n) {
    int tid = blockIdx.x;
    if (tid < n)
        array[tid] = array[tid] * array[tid];
}
