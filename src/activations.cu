// activations.cu
#include "../include/activations.h"
#include <math.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void sigmoid_kernel(float *input, float *output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sigmoid(input[idx]);
    }
}

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__global__ void relu_kernel(float *input, float *output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        output[idx] = relu(input[idx]);
    }
}

__device__ float linear(float x) {
    return x;
}

__global__ void linear_kernel(float *input, float *output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        output[idx] = linear(input[idx]);
    }
}

// Implement other activation functions similarly
