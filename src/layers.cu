// layers.cu
#include "../include/layers.h"
#include "../include/activations.h"
#include "../include/lossFunction.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h> 
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

// Macro to check CUDA errors
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Function to initialize weights with small random values
__global__ void init_weights_kernel(float *w, int size, int fan_in, int fan_out, unsigned int seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        // Initialize CURAND
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Calculate Glorot Uniform limit
        float limit = sqrtf(6.0f / (float)(fan_in + fan_out));

        // Generate a random number in [0, 1)
        float rand_uniform = curand_uniform(&state); // [0, 1)

        // Scale to [-limit, limit)
        w[idx] = rand_uniform * 2.0f * limit - limit;
    }
}

void layer_init(Layer *layer, int m, int n_in, int n_out, ActivationType aktfunc) {
    layer->m = m;
    layer->n_in = n_in;
    layer->n_out = n_out;
    layer->aktfunc = aktfunc;

    size_t w_size = n_out * n_in * sizeof(float);
    size_t b_size = n_out * sizeof(float);
    size_t A_size = n_out * m * sizeof(float);
    size_t Z_size = n_out * m * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&layer->w_d, w_size);
    cudaMalloc((void**)&layer->b_d, b_size);
    cudaMalloc((void**)&layer->A_d, A_size);
    cudaMalloc((void**)&layer->Z_d, Z_size);
    cudaMalloc((void**)&layer->dZ_d, Z_size);
    cudaMalloc((void**)&layer->dW_d, w_size);
    cudaMalloc((void**)&layer->db_d, b_size);
    cudaCheckError();

    // Initialize weights using Glorot Uniform
    int threads = 256;
    int blocks = (n_out * n_in + threads - 1) / threads;
    unsigned int seed = time(NULL); // For randomness; consider using a fixed seed for reproducibility
    init_weights_kernel<<<blocks, threads>>>(layer->w_d, n_out * n_in, n_in, n_out, seed);
    cudaCheckError();

    // Initialize biases to zero
    cudaMemset(layer->b_d, 0, b_size);
    cudaCheckError();
}


// Kernel to add bias
__global__ void add_bias_kernel(float *Z, const float *b, int n_out, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_out * m) {
        int row = idx % n_out;
        Z[idx] += b[row];
    }
}

// Kernel for activation function
__global__ void activation_forward_kernel(const float *Z, float *A, int size, ActivationType aktfunc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        switch (aktfunc) {
            case ACTIVATION_RELU:
                A[idx] = fmaxf(0.0f, z);
                break;
            case ACTIVATION_SIGMOID:
                A[idx] = 1.0f / (1.0f + expf(-z));
                break;
            default:
                A[idx] = z; // Linear activation
                break;
        }
    }
}

void layer_forward(Layer *layer, float *A_prev_d, cublasHandle_t handle) {

    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute Z_d = w_d * A_prev_d
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,       // Transpose w_d
        CUBLAS_OP_N,       // Transpose A_prev_d
        layer->n_out,                 // Number of rows in Z_d
        layer->m,                 // Number of columns in Z_d
        layer->n_in,                 // Shared dimension
        &alpha,
        layer->w_d,        // Pointer to w_d
        layer->n_out,                 // Leading dimension of w_d
        A_prev_d,          // Pointer to A_prev_d
        layer->n_in,                 // Leading dimension of A_prev_d
        &beta,
        layer->Z_d,        // Pointer to Z_d
        layer->n_out                  // Leading dimension of Z_d
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error in cublasSgemm: %d\n", status);
        // Handle the error or exit
        exit(EXIT_FAILURE);
    }
        
    int threads = 256;
    int total_elements = layer->n_out * layer->m;
    int blocks = (total_elements + threads - 1) / threads;

    add_bias_kernel<<<blocks, threads>>>(layer->Z_d, layer->b_d, layer->n_out, layer->m);
    cudaCheckError();
    
    // Apply activation function
    threads = 256;
    blocks = (total_elements + threads - 1) / threads;
    activation_forward_kernel<<<blocks, threads>>>(layer->Z_d, layer->A_d, total_elements, layer->aktfunc);
    
    cudaCheckError();
}

// Kernel to compute dZ = A - Y
__global__ void compute_dZ(float *dZ, const float *A, const float *Y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = A[idx] - Y[idx];
    }
}

// Kernel to compute db = (1/m) * sum(dZ) (column-wise sum)
__global__ void compute_db(float *db, const float *dZ, int n_out, int m) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < n_out) {
        float sum = 0.0f;
        for (int col = 0; col < m; ++col) {
            sum += dZ[row + col * n_out];
        }
        db[row] = sum / m;
    }
}

// Helper function to compute dW and db
void compute_gradients(Layer *layer, float *A_prev_d, cublasHandle_t handle) {
    int n_out = layer->n_out;
    int n_in = layer->n_in;
    int m = layer->m;

    float alpha = 1.0f / m;
    float beta = 0.0f;

    // Compute dW = (1/m) * dZ * A_prev^T
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,       // dZ
        CUBLAS_OP_T,       // A_prev_d
        n_out,             // Number of rows in dZ
        n_in,              // Number of columns in A_prev^T
        m,                 // Shared dimension
        &alpha,
        layer->dZ_d,       // Pointer to dZ
        n_out,             // Leading dimension of dZ
        A_prev_d,          // Pointer to A_prev_d
        n_in,              // Leading dimension of A_prev_d
        &beta,
        layer->dW_d,       // Pointer to dW
        n_out              // Leading dimension of dW
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM error: %d\n", status);
        return;
    }

    // Compute db = (1/m) * sum(dZ) (column-wise sum)
    int block_size = 256;
    int grid_size = (n_out + block_size - 1) / block_size;
    compute_db<<<grid_size, block_size>>>(layer->db_d, layer->dZ_d, n_out, m);
    //cudaDeviceSynchronize();
    cudaCheckError();
}

// Kernel for activation function derivative
__global__ void deriv_akt_kernel(const float *Z, float *dZ, int size, ActivationType aktfunc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        switch (aktfunc) {
            case ACTIVATION_RELU:
                dZ[idx] = dZ[idx] * (z > 0);
                break;
            default:
                dZ[idx] = z; // Linear activation
                break;
        }
    }
}

void backward_output_layer(Layer *layer, float *Y, float *A_prev_d, cublasHandle_t handle) {
    int dZ_size = layer->n_out * layer->m;
    int block_size = 256;
    int grid_size = (dZ_size + block_size - 1) / block_size;

    // Compute dZ = A - Y
    compute_dZ<<<grid_size, block_size>>>(layer->dZ_d, layer->A_d, Y, dZ_size);
    //cudaDeviceSynchronize();
    cudaCheckError();
    // Compute dW and db
    compute_gradients(layer, A_prev_d, handle);
}

void backward_layer(Layer *layer, float *W_next_d, float *dZ_next_d, float *A_prev_d, int n_out_next, cublasHandle_t handle) {
    int m = layer->n_out;
    int n = layer->m;
    int k = n_out_next;

    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute dZ = W_next^T * dZ_next
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T,      // W_next^T
        CUBLAS_OP_N,      // dZ_next
        m,                // Number of rows in W_next^T
        n,                // Number of columns in dZ_next
        k,                // Shared dimension
        &alpha,
        W_next_d,         // Pointer to W_next_d
        k,                // Leading dimension of W_next_d
        dZ_next_d,        // Pointer to dZ_next_d
        k,                // Leading dimension of dZ_next_d
        &beta,
        layer->dZ_d,      // Pointer to dZ
        m                 // Leading dimension of dZ
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM error: %d\n", status);
        return;
    }

    // Apply activation function derivative
    int total_elements = m * n;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    deriv_akt_kernel<<<blocks, threads>>>(layer->Z_d, layer->dZ_d, total_elements, layer->aktfunc);
    //cudaDeviceSynchronize();
    cudaCheckError();
    // Compute dW and db
    compute_gradients(layer, A_prev_d, handle);
}
// Combined kernel to update weights and biases
__global__ void update_params(
    float *w,    // Pointer to weights on device
    const float *dW,  // Pointer to weight gradients on device
    float *b,    // Pointer to biases on device
    const float *db,  // Pointer to bias gradients on device
    float learning_rate,
    int w_size,
    int b_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // First, update weights if idx falls within the weight range
    if (idx < w_size) {
        w[idx] -= learning_rate * dW[idx];
    }

    // Next, update biases if idx falls within the bias range
    // We shift idx by w_size to map to the bias section
    int b_idx = idx - w_size;
    if (b_idx >= 0 && b_idx < b_size) {
        b[b_idx] -= learning_rate * db[b_idx];
    }
}

void update(Layer *layer, float learning_rate) {
    int block_size = 256;

    // Calculate sizes
    int w_size = layer->n_out * layer->n_in;  // Total number of weights
    int b_size = layer->n_out;                // Total number of biases
    
    // The total size is the sum of weights and biases
    int total_size = w_size + b_size;

    // Configure the grid
    int grid_size = (total_size + block_size - 1) / block_size;

    // Launch the combined kernel
    update_params<<<grid_size, block_size>>>(
        layer->w_d,
        layer->dW_d,
        layer->b_d,
        layer->db_d,
        learning_rate,
        w_size,
        b_size
    );
    //cudaDeviceSynchronize();
    cudaCheckError();
}


void free_layer(Layer *layer) {
    cudaFree(layer->w_d);
    cudaFree(layer->b_d);
    cudaFree(layer->A_d);
    cudaFree(layer->Z_d);
    cudaFree(layer->dZ_d);
    cudaFree(layer->dW_d);
    cudaFree(layer->db_d);
}
/*
void print_layer(Layer *layer) {
    printf("Type: %s\n", layer->type);
    printf("Input Size: %d\n", layer->input_size);
    printf("Output Size: %d\n", layer->output_size);
    const char* activation_str;
    switch (layer->activation) {
        case ACTIVATION_SIGMOID:
            activation_str = "sigmoid";
            break;
        case ACTIVATION_RELU:
            activation_str = "relu";
            break;
        case ACTIVATION_LINEAR:
            activation_str = "linear";
            break;
        default:
            activation_str = "unknown or not implemented (maybe forgot to add it to print_layer)";
            break;
    }
    printf("Activation: %s\n", activation_str);

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

*/