// layers.cu
#include "layers.h"
#include "activations.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

Layer* create_dense_layer(int input_size, int output_size) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));

    // Set layer type
    layer->type = "dense";
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Allocate host memory for weights and biases
    layer->weights = (float*)malloc(input_size * output_size * sizeof(float));
    layer->biases = (float*)malloc(output_size * sizeof(float));

    // Initialize weights and biases (e.g., Xavier initialization)

    // Allocate device memory
    cudaMalloc((void**)&(layer->d_weights), input_size * output_size * sizeof(float));
    cudaMalloc((void**)&(layer->d_biases), output_size * sizeof(float));

    // Copy weights and biases to device
    cudaMemcpy(layer->d_weights, layer->weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(layer->d_biases, layer->biases, output_size * sizeof(float), cudaMemcpyHostToDevice);

    return layer;
}

// Implement forward, backward, and update functions using CUDA kernels

void free_layer(Layer *layer) {
    // Free device memory
    cudaFree(layer->d_weights);
    cudaFree(layer->d_biases);

    // Free host memory
    free(layer->weights);
    free(layer->biases);
    free(layer);
}
