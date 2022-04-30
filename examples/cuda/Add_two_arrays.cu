__global__ void elementwise_add(const int * array1,
    const int * array2, int * result, int size) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    while (idx < size) {
        result[idx] = array1[idx] + array2[idx];
        idx += stride;
    }
}
