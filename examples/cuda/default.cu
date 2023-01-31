#include <stdio.h>

void __device__ foo_impl(double *x_in, double *x_out) {
  x_out[0] = x_in[0] * x_in[0];
}

typedef void (*f_ptr)(double *, double *);

extern void __device__ __enzyme_autodiff(f_ptr, int, double *, double *, int,
                                         double *, double *);

void __global__ foo(double *x_in, double *x_out) { foo_impl(x_in, x_out); }

int __device__ enzyme_dup;
int __device__ enzyme_out;
int __device__ enzyme_const;

void __global__ foo_grad(double *x, double *d_x, double *y, double *d_y) {

  __enzyme_autodiff(foo_impl, enzyme_dup, x, d_x, enzyme_dup, y, d_y);
}

int main() {

  double *x, *d_x, *y, *d_y; // device pointers

  cudaMalloc(&x, sizeof(*x));
  cudaMalloc(&d_x, sizeof(*d_x));
  cudaMalloc(&y, sizeof(*y));
  cudaMalloc(&d_y, sizeof(*d_y));

  double host_x = 1.4;
  double host_d_x = 0.0;
  double host_y;
  double host_d_y = 1.0;

  cudaMemcpy(x, &host_x, sizeof(*x), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, &host_d_x, sizeof(*d_x), cudaMemcpyHostToDevice);
  cudaMemcpy(y, &host_y, sizeof(*y), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, &host_d_y, sizeof(*d_y), cudaMemcpyHostToDevice);

  // foo<<<1,1>>>(x, y); fwd-pass only
  foo_grad<<<1, 1>>>(x, d_x, y, d_y); // fwd and bkwd pass

  cudaDeviceSynchronize(); // synchroniz

  cudaMemcpy(&host_x, x, sizeof(*x), cudaMemcpyDeviceToHost);
  cudaMemcpy(&host_d_x, d_x, sizeof(*d_x), cudaMemcpyDeviceToHost);
  cudaMemcpy(&host_y, y, sizeof(*y), cudaMemcpyDeviceToHost);
  cudaMemcpy(&host_d_y, d_y, sizeof(*d_y), cudaMemcpyDeviceToHost);

  printf("%f %f\n", host_x, host_y);
  printf("%f %f\n", host_d_x, host_d_y);
}
