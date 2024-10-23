// activations.h

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include <cuda_runtime.h>

__device__ float sigmoid(float x);
__global__ void sigmoid_kernel(float *input, float *output, int n);

#endif // ACTIVATIONS_H
