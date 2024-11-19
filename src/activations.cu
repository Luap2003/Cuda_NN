// activations.cu
#include "../include/activations.h"
#include <math.h>

__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void sigmoid_kernel(float *input, float *output, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    output[idx] = sigmoid(input[idx]);
  }
}

__global__ void sigmoid_derivative_kernel(float *d_output_grad, float *output,
                                          float *delta, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    delta[idx] = d_output_grad[idx] * output[idx] * (1.0f - output[idx]);
  }
}

__device__ float relu(float x) { return x > 0 ? x : 0; }

__global__ void relu_kernel(float *input, float *output, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    output[idx] = relu(input[idx]);
  }
}

__global__ void relu_derivative_kernel(float *d_output_grad, float *output,
                                       float *delta, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    delta[idx] = d_output_grad[idx] * (output[idx] > 0 ? 1.0f : 0.0f);
  }
}

__device__ float linear(float x) { return x; }

__global__ void linear_kernel(float *input, float *output, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    output[idx] = linear(input[idx]);
  }
}

__global__ void linear_derivative_kernel(float *d_output_grad, float *delta,
                                         int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    delta[idx] = d_output_grad[idx];
  }
}
