// activations.h

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include <cuda_runtime.h>
typedef enum {
    ACTIVATION_LINEAR,
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU
    // Add more activations as needed
} ActivationType;

__device__ float sigmoid(float x);
__global__ void sigmoid_kernel(float *input, float *output, int n);
__global__ void sigmoid_derivative_kernel(float *d_output_grad, float *output, float *delta, int size);


__device__ float relu(float x);
__global__ void relu_kernel(float *input, float *output, int n);
__global__ void relu_derivative_kernel(float *d_output_grad, float *output, float *delta, int size);

__device__ float linear(float x);
__global__ void linear_kernel(float *input, float *output, int n);
__global__ void linear_derivative_kernel(float *d_output_grad, float *output, float *delta, int size);
#endif // ACTIVATIONS_H
