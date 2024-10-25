// lossFunction.cu
#include "../include/lossFunction.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_loss_kernel(float *d_predictions, float *d_labels, float *d_loss, int size, LossFunction loss_function) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        float loss = 0.0f;
        if (loss_function == LOSS_MSE) {
            float diff = d_predictions[idx] - d_labels[idx];
            loss = diff * diff;
        }
        else if (loss_function == LOSS_CROSSENTROPY) {
            // To avoid log(0), add a small epsilon
            const float epsilon = 1e-12f;
            float pred = fmaxf(d_predictions[idx], epsilon);
            float label = d_labels[idx];
            loss = - (label * logf(pred) + (1.0f - label) * logf(1.0f - pred));
        }
        // You can add more loss functions here
        atomicAdd(d_loss, loss);
    }
}

float compute_loss(float *d_predictions, float *d_labels, int size, LossFunction loss_function) {
    float h_loss = 0.0f;
    float *d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_loss_kernel<<<blocks, threads>>>(d_predictions, d_labels, d_loss, size, loss_function);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss / size;
}
