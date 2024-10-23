// activations.h

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include <cuda_runtime.h>

__device__ float sigmoid(float x);
__global__ void sigmoid_kernel(float *input, float *output, int n);

__device__ float relu(float x);
__global__ void relu_kernel(float *input, float *output, int n);

__device__ float linear(float x);
__global__ void linear_kernel(float *input, float *output, int n);
#endif // ACTIVATIONS_H
