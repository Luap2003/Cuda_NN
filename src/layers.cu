// layers.cu
#include "../include/layers.h"
#include "../include/activations.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h> 

Layer* create_dense_layer(int input_size, int output_size, const char *activation) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));

    // Set layer properties
    layer->type = strdup("dense");  // Duplicate string to avoid pointer issues
    layer->activation = strdup(activation);
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Allocate host memory for weights and biases
    layer->weights = (float*)malloc(input_size * output_size * sizeof(float));
    layer->biases = (float*)malloc(output_size * sizeof(float));

    // Xavier Initialization
    float limit = sqrtf(6.0f / (input_size + output_size));
    for (int i = 0; i < input_size * output_size; ++i) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
    }
    for (int i = 0; i < output_size; ++i) {
        layer->biases[i] = 0.0f;  // Initialize biases to zero
    }

    // Allocate device memory
    cudaMalloc((void**)&(layer->d_weights), input_size * output_size * sizeof(float));
    cudaMalloc((void**)&(layer->d_biases), output_size * sizeof(float));

    // Copy weights and biases to device
    cudaMemcpy(layer->d_weights, layer->weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(layer->d_biases, layer->biases, output_size * sizeof(float), cudaMemcpyHostToDevice);

    return layer;
}

__global__ void add_bias(float *d_output, float *d_biases, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_size;
    if (idx < total_elements) {
        int bias_idx = idx % output_size;
        d_output[idx] += d_biases[bias_idx];
    }
}

void forward_layer(Layer *layer, float *d_input, float *d_output, int batch_size) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform matrix multiplication: d_output = alpha * d_input * d_weights^T + beta * d_output
    cublasStatus_t stat = cublasSgemm(handle,CUBLAS_OP_T, CUBLAS_OP_N, 
                layer->output_size,
                batch_size,
                layer->input_size,
                &alpha,
                layer->d_weights, layer->input_size,      // d_weights^T has dimensions [n x k]
                d_input, layer->input_size,               // d_input has dimensions [m x k]
                &beta,
                d_output, layer->output_size);            // d_output has dimensions [n x m]

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed\n");
    }

    int threads_per_block = THREADS_PER_BLOCK;
    int total_elements = batch_size * layer->output_size;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    add_bias<<<num_blocks, threads_per_block>>>(d_output, layer->d_biases, batch_size, layer->output_size);
    cudaDeviceSynchronize();

    if (strcmp(layer->activation, "sigmoid") == 0) {
        // Call sigmoid activation function
        sigmoid_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output, total_elements);
    } else if (strcmp(layer->activation, "relu") == 0) {
        // Call relu activation function
        relu_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output, total_elements);
    } else if (strcmp(layer->activation, "linear") == 0) {
        // Call linear activation function
        linear_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output, total_elements);
    } else { 
        // Add other activation functions here using else if
        printf("Activation function not implemented\n");
    }
    cudaDeviceSynchronize();
    cublasDestroy(handle);

}


void free_layer(Layer *layer) {
    // Free device memory
    cudaFree(layer->d_weights);
    cudaFree(layer->d_biases);

    // Free host memory
    free(layer->weights);
    free(layer->biases);
    free(layer);
}

void print_layer(Layer *layer) {
    printf("Type: %s\n", layer->type);
    printf("Input Size: %d\n", layer->input_size);
    printf("Output Size: %d\n", layer->output_size);
    printf("Activation: %s\n", layer->activation);

    // Print a snippet of weights and biases
    int num_weights_to_print = 5;
    printf("Weights (first %d values):\n", num_weights_to_print);
    for (int i = 0; i < num_weights_to_print && i < layer->input_size * layer->output_size; ++i) {
        printf("%f ", layer->weights[i]);
    }
    printf("\nBiases (first %d values):\n", num_weights_to_print);
    for (int i = 0; i < num_weights_to_print && i < layer->output_size; ++i) {
        printf("%f ", layer->biases[i]);
    }
    printf("\n");
}