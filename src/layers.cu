// layers.cu
#include "../include/layers.h"
#include "../include/activations.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h> 

Layer* create_dense_layer(int input_size, int output_size, ActivationType activation) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));

    // Set layer properties
    layer->type = strdup("dense");  // Duplicate string to avoid pointer issues
    layer->activation = activation;
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
    cudaMalloc((void**)&(layer->d_weights_grad), input_size * output_size * sizeof(float));
    cudaMalloc((void**)&(layer->d_biases_grad), output_size * sizeof(float));
    cudaMalloc((void**)&(layer->d_output), output_size * sizeof(float)); // Adjust size based on batch
    cudaMalloc((void**)&(layer->d_z), output_size * sizeof(float));      // Adjust size based on batch

    // Initialize gradients to zero
    cudaMemset(layer->d_weights_grad, 0, input_size * output_size * sizeof(float));
    cudaMemset(layer->d_biases_grad, 0, output_size * sizeof(float));

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

    if (layer->activation == ACTIVATION_SIGMOID) {
        // Call sigmoid activation function
        sigmoid_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output, total_elements);
    } else if (layer->activation == ACTIVATION_RELU) {
        // Call relu activation function
        relu_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output, total_elements);
    } else if (layer->activation == ACTIVATION_LINEAR) {
        // Call linear activation function
        linear_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output, total_elements);
    } else { 
        // Add other activation functions here using else if
        printf("Activation function not implemented\n");
    }
    cudaDeviceSynchronize();
    cublasDestroy(handle);

}

__device__ int my_strcmp(const char *str1, const char *str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(unsigned char *)str1 - *(unsigned char *)str2;
}

__global__ void compute_output_delta(float *d_output_delta, float *d_output, float *d_z, float *d_labels, int size, ActivationType activation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Compute (a^L - y)
        float error = d_output[idx] - d_labels[idx];

        // Compute σ'(z^L)
        float sigma_prime;
        if (activation == ACTIVATION_SIGMOID) {
            float sigmoid = 1.0f / (1.0f + expf(-d_z[idx]));
            sigma_prime = sigmoid * (1.0f - sigmoid);
        } else if (activation == ACTIVATION_RELU) {
            sigma_prime = d_z[idx] > 0 ? 1.0f : 0.0f;
        } else if (activation == ACTIVATION_LINEAR) {
            sigma_prime = 1.0f;
        } else {
            sigma_prime = 1.0f; // Default to linear if unknown
        }

        // Compute δ^L = (a^L - y) * σ'(z^L)
        d_output_delta[idx] = error * sigma_prime;
    }
}

void backward_output_layer(Layer *layer, float *d_labels, float *d_output_delta, int batch_size) {
    int size = batch_size * layer->output_size;
    int threads = THREADS_PER_BLOCK;
    int blocks = (size + threads - 1) / threads;

    // Launch kernel to compute δ^L
    compute_output_delta<<<blocks, threads>>>(d_output_delta, layer->d_output, layer->d_z, d_labels, size, layer->activation);
    cudaDeviceSynchronize();
}

__global__ void compute_hidden_delta(float *d_current_delta, float *d_z, ActivationType activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Compute σ'(z^l)
        float sigma_prime;
        if (activation == ACTIVATION_SIGMOID) {
            float sigmoid = 1.0f / (1.0f + expf(-d_z[idx]));
            sigma_prime = sigmoid * (1.0f - sigmoid);
        } else if (activation == ACTIVATION_RELU) {
            sigma_prime = d_z[idx] > 0 ? 1.0f : 0.0f;
        } else if (activation == ACTIVATION_LINEAR) {
            sigma_prime = 1.0f;
        } else {
            // Add other activation derivatives as needed
            sigma_prime = 1.0f; // Default to linear if unknown
        }

        // Compute δ^l = (W^{l+1}^T * δ^{l+1}) * σ'(z^l)
        d_current_delta[idx] *= sigma_prime;
    }
}

void backward_layer(Layer *current_layer, Layer *next_layer, float *d_output_delta, float *d_input_grad, int batch_size) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Dimensions
    int m = current_layer->output_size; // Number of neurons in current layer
    int n = next_layer->output_size;    // Number of neurons in next layer
    int k = batch_size;

    // Allocate memory for delta of the current layer
    float *d_current_delta;
    cudaMalloc(&d_current_delta, batch_size * current_layer->output_size * sizeof(float));
    cudaMemset(d_current_delta, 0, batch_size * current_layer->output_size * sizeof(float));

    // Compute (W^{l+1})^T * δ^{l+1}
    // Using cublasSgemm: C = alpha * A * B + beta * C
    // A: d_weights of next layer [output_size_next x input_size_next]
    // B: d_output_delta of next layer [batch_size x output_size_next]
    // We need to compute A^T [input_size_next x output_size_next] * B^T [output_size_next x batch_size] = [input_size_next x batch_size]
    // Transpose operations accordingly
    cublasStatus_t stat = cublasSgemm(handle,
                                     CUBLAS_OP_T, CUBLAS_OP_T,
                                     current_layer->output_size, batch_size, next_layer->output_size,
                                     &alpha,
                                     next_layer->d_weights, next_layer->input_size,      // A^T
                                     d_output_delta, next_layer->output_size,            // B^T
                                     &beta,
                                     d_current_delta, current_layer->output_size);      // C

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm for hidden layer delta failed\n");
    }

    // Element-wise multiply with σ'(z^l)
    int size = batch_size * current_layer->output_size;
    int threads = THREADS_PER_BLOCK;
    int blocks = (size + threads - 1) / threads;

    // Launch kernel to compute δ^l = (W^{l+1}^T * δ^{l+1}) * σ'(z^l)
    compute_hidden_delta<<<blocks, threads>>>(d_current_delta, current_layer->d_z, current_layer->activation, size);
    cudaDeviceSynchronize();

    // Now, d_current_delta contains δ^l, which can be used to compute gradients
    // Proceed to compute gradients...

    // Free temporary memory
    cudaFree(d_current_delta);

    // Destroy CUBLAS handle
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