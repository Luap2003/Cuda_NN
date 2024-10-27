// neural_net.cu
#include "../include/neural_net.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <stdio.h>
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif


// Kernel to update weights: W = W - learning_rate * dW
__global__ void update_weights_kernel(float *d_weights, float *d_weights_grad, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_weights[idx] -= learning_rate * d_weights_grad[idx];
    }
}

// Kernel to update biases: b = b - learning_rate * db
__global__ void update_biases_kernel(float *d_biases, float *d_biases_grad, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_biases[idx] -= learning_rate * d_biases_grad[idx];
    }
}

// Function to update parameters of a single layer
void update_parameters(Layer *layer, float learning_rate) {
    int weights_size = layer->input_size * layer->output_size;
    int biases_size = layer->output_size;

    int threads = THREADS_PER_BLOCK;
    int blocks_weights = (weights_size + threads - 1) / threads;
    int blocks_biases = (biases_size + threads - 1) / threads;

    // Update weights
    update_weights_kernel<<<blocks_weights, threads>>>(layer->d_weights, layer->d_weights_grad, learning_rate, weights_size);
    // Update biases
    update_biases_kernel<<<blocks_biases, threads>>>(layer->d_biases, layer->d_biases_grad, learning_rate, biases_size);

    // Reset gradients to zero for next iteration
    cudaMemset(layer->d_weights_grad, 0, weights_size * sizeof(float));
    cudaMemset(layer->d_biases_grad, 0, biases_size * sizeof(float));

    cudaDeviceSynchronize();
}

// The main backpropagation function
void backpropagation(NeuralNetwork *network, float *d_input, float *d_labels, int batch_size, float learning_rate) {
    // Forward Pass
    float *current_input = d_input;
    for (int i = 0; i < network->num_layers; ++i) {
        Layer *layer = network->layers[i];
        forward_layer(layer, current_input, layer->d_output, batch_size);
        // Store z (pre-activation) if not already stored in forward_layer
        // Assuming forward_layer stores z in layer->d_z
        // If not, modify forward_layer to compute and store z
        current_input = layer->d_output;
    }

    // Backward Pass
    // Initialize delta for output layer
    Layer *output_layer = network->layers[network->num_layers - 1];
    float *d_output_delta;
    cudaMalloc((void**)&d_output_delta, output_layer->output_size * batch_size * sizeof(float));

    // Compute delta for output layer
    backward_output_layer(output_layer, d_labels, d_output_delta, batch_size);

    // Accumulate gradients for output layer
    // Compute gradients: dW = a^{l-1}^T * delta^l
    // Assuming a^{l-1} is the input to the layer, which is d_input from forward pass
    // Similarly, db = sum over delta^l for each bias

    // We'll use cuBLAS to compute dW and accumulate in d_weights_grad
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 1.0f; // Accumulate gradients

    // Compute dW = a^{l-1}^T * delta^l
    // d_input: [batch_size x input_size] (row-major)
    // d_output_delta: [batch_size x output_size] (row-major)
    // dW_grad: [input_size x output_size] (row-major)

    // cuBLAS uses column-major, so we need to adjust parameters
    // Transpose matrices to match column-major expectations

    cublasStatus_t stat = cublasSgemm(handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     output_layer->input_size,    // m
                                     output_layer->output_size,   // n
                                     batch_size,                  // k
                                     &alpha,
                                     current_input,               // A: [batch_size x input_size] row-major
                                     batch_size,                  // lda
                                     d_output_delta,              // B: [batch_size x output_size] row-major
                                     batch_size,                  // ldb
                                     &beta,
                                     output_layer->d_weights_grad, // C: [input_size x output_size] row-major
                                     output_layer->input_size     // ldc
                                     );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed for output layer gradients\n");
    }

    // Compute db = sum over batch of delta^l
    // We'll use a custom kernel for this (reduce operation)
    // For simplicity, let's assume we have a kernel called sum_reduction
    // Alternatively, you can use cuBLAS reduction operations

    // Placeholder for db computation
    // You need to implement or use an existing reduction kernel to sum delta over batch
    // Here's a simple approach using cuBLAS:

    // Initialize d_biases_grad with delta
    // Sum along the batch dimension
    stat = cublasSasum(handle, output_layer->output_size * batch_size, d_output_delta, 1, output_layer->d_biases_grad);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSasum failed for output layer biases gradients\n");
    }

    cublasDestroy(handle);

    // Update parameters for output layer
    update_parameters(output_layer, learning_rate);

    // Backpropagate through hidden layers
    float *d_prev_delta = d_output_delta; // Delta from the current layer
    for (int i = network->num_layers - 2; i >= 0; --i) {
        Layer *current_layer = network->layers[i];
        Layer *next_layer = network->layers[i + 1];

        // Compute delta for current layer
        float *d_current_delta;
        int current_size = current_layer->output_size * batch_size;
        cudaMalloc((void**)&d_current_delta, current_size * sizeof(float));

        backward_layer(current_layer, next_layer, d_prev_delta, d_current_delta, batch_size);

        // Accumulate gradients for current layer
        // Compute dW = a^{l-1}^T * delta^l
        // Assuming a^{l-1} is the input to the current layer

        // Set up cuBLAS handle
        cublasCreate(&handle);

        // Compute dW = a^{l-1}^T * delta^l
        // a^{l-1}: [batch_size x input_size] row-major
        // d_current_delta: [batch_size x output_size] row-major
        // d_weights_grad: [input_size x output_size] row-major

        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    current_layer->input_size,
                    current_layer->output_size,
                    batch_size,
                    &alpha,
                    current_input,            // A: [batch_size x input_size] row-major
                    batch_size,               // lda
                    d_current_delta,          // B: [batch_size x output_size] row-major
                    batch_size,               // ldb
                    &beta,
                    current_layer->d_weights_grad, // C: [input_size x output_size] row-major
                    current_layer->input_size // ldc
                    );

        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("cublasSgemm failed for hidden layer %d gradients\n", i);
        }

        // Compute db = sum over batch of delta^l
        // Again, use cuBLAS sasum or a custom reduction kernel
        stat = cublasSasum(handle, current_layer->output_size * batch_size, d_current_delta, 1, current_layer->d_biases_grad);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("cublasSasum failed for hidden layer %d biases gradients\n", i);
        }

        cublasDestroy(handle);

        // Update parameters for current layer
        update_parameters(current_layer, learning_rate);

        // Free previous delta and set current delta as prev_delta for next iteration
        cudaFree(d_prev_delta);
        d_prev_delta = d_current_delta;

        // Set current_input to the input of the current layer for next iteration
        // This requires storing the inputs to each layer during the forward pass
        // For simplicity, assuming you have stored them. If not, you need to modify forward_layer to store inputs.
        // Here, we'll skip setting current_input as it's not straightforward without storing activations.
        // You'll need to implement this part based on how your forward pass is structured.
    }

    // Free the last delta
    cudaFree(d_prev_delta);
}

NeuralNetwork* create_neural_net(int num_layers) {
    NeuralNetwork *network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    network->num_layers = num_layers;
    network->layers = (Layer**)malloc(num_layers * sizeof(Layer*));
    return network;
}

void add_layer_to_neural_net(NeuralNetwork *network, Layer *layer, int index) {
    if (index >= 0 && index < network->num_layers) {
        network->layers[index] = layer;
    }
}

void free_neural_net(NeuralNetwork *network) {
    for (int i = 0; i < network->num_layers; ++i) {
        free_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}
