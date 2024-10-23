// activations.h
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

__device__ float sigmoid(float x);
__global__ void sigmoid_kernel(float *input, float *output, int n);

// Other activation functions

#endif // ACTIVATIONS_H
