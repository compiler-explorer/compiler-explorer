__global__ void sum_array(const int * array, int * total, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int input_idx = idx;

    __shared__ int partial_res[256];

    int partial_sum = 0;
    while (input_idx < n) {
        partial_sum += array[input_idx];
        input_idx += stride;
    }

    partial_res[threadIdx.x] = partial_sum;

    __syncthreads();

    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            partial_res[threadIdx.x] += partial_res[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(total, partial_res[0]);
    }
}
