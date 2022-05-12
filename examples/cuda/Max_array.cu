#define MAX(x, y)((x > y) ? x : y)

__global__ void find_max(const int * array, int * max, int * mutex, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int input_idx = idx;

    __shared__ int partial_res[256];

    int local_max = array[0];
    while (input_idx < n) {
        local_max = MAX(local_max, array[input_idx]);
        input_idx += stride;
    }

    partial_res[threadIdx.x] = local_max;

    __syncthreads();

    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            partial_res[threadIdx.x] = MAX(partial_res[threadIdx.x], partial_res[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        while (atomicCAS(mutex, 0, 1) != 0);
        * max = MAX( * max, partial_res[0]);
        atomicExch(mutex, 0);
    }
}
