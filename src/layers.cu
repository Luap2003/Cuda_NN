// layers.cu
#include "../include/activations.h"
#include "../include/layers.h"
#include "../include/lossFunction.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Layer *create_dense_layer(int input_size, int output_size,
                          ActivationType activation) {
  Layer *layer = (Layer *)malloc(sizeof(Layer));

  // Set layer properties
  layer->type = strdup("dense"); // Duplicate string to avoid pointer issues
  layer->activation = activation;
  layer->input_size = input_size;
  layer->output_size = output_size;

  // Allocate host memory for weights and biases
  layer->weights = (float *)malloc(input_size * output_size * sizeof(float));
  layer->biases = (float *)malloc(output_size * sizeof(float));

  // Xavier Initialization
  float limit = sqrtf(6.0f / (input_size + output_size));
  for (int i = 0; i < input_size * output_size; ++i) {
    layer->weights[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
  }
  for (int i = 0; i < output_size; ++i) {
    layer->biases[i] = 0.0f; // Initialize biases to zero
  }

  // Allocate device memory
  cudaMalloc((void **)&(layer->d_weights),
             input_size * output_size * sizeof(float));
  cudaMalloc((void **)&(layer->d_biases), output_size * sizeof(float));
  cudaMalloc((void **)&(layer->d_weights_grad),
             input_size * output_size * sizeof(float));
  cudaMalloc((void **)&(layer->d_biases_grad), output_size * sizeof(float));
  cudaMalloc((void **)&(layer->d_output),
             output_size * sizeof(float)); // Adjust size based on batch
  cudaMalloc((void **)&(layer->d_z),
             output_size * sizeof(float)); // Adjust size based on batch

  // Initialize gradients to zero
  cudaMemset(layer->d_weights_grad, 0,
             input_size * output_size * sizeof(float));
  cudaMemset(layer->d_biases_grad, 0, output_size * sizeof(float));

  // Copy weights and biases to device
  cudaMemcpy(layer->d_weights, layer->weights,
             input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(layer->d_biases, layer->biases, output_size * sizeof(float),
             cudaMemcpyHostToDevice);

  return layer;
}

__global__ void add_bias(float *d_output, float *d_biases, int batch_size,
                         int output_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * output_size;
  if (idx < total_elements) {
    int bias_idx = idx % output_size;
    d_output[idx] += d_biases[bias_idx];
  }
}

void forward_layer(Layer *layer, float *d_input, float *d_output,
                   int batch_size) {

  if (layer->d_input == NULL || layer->batch_size != batch_size) {
    if (layer->d_input != NULL) {
      cudaFree(layer->d_input);
    }
    cudaMalloc(&(layer->d_input),
               batch_size * layer->input_size * sizeof(float));
    layer->batch_size = batch_size;
  }

  // Copy d_input to layer->d_input
  cudaMemcpy(layer->d_input, d_input,
             batch_size * layer->input_size * sizeof(float),
             cudaMemcpyDeviceToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  // Compute z = W * a +b
  // Perform matrix multiplication: d_output = alpha * d_input * d_weights^T +
  // beta * d_output
  cublasStatus_t stat = cublasSgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, layer->output_size, batch_size,
      layer->input_size, &alpha, layer->d_weights,
      layer->input_size,          // d_weights^T has dimensions [n x k]
      d_input, layer->input_size, // d_input has dimensions [m x k]
      &beta, d_output, layer->output_size); // d_output has dimensions [n x m]

  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm failed\n");
  }

  int threads_per_block = THREADS_PER_BLOCK;
  int total_elements = batch_size * layer->output_size;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  add_bias<<<num_blocks, threads_per_block>>>(d_output, layer->d_biases,
                                              batch_size, layer->output_size);
  cudaDeviceSynchronize();

  if (layer->activation == ACTIVATION_SIGMOID) {
    // Call sigmoid activation function
    sigmoid_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output,
                                                      total_elements);
  } else if (layer->activation == ACTIVATION_RELU) {
    // Call relu activation function
    relu_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output,
                                                   total_elements);
  } else if (layer->activation == ACTIVATION_LINEAR) {
    // Call linear activation function
    linear_kernel<<<num_blocks, threads_per_block>>>(d_output, d_output,
                                                     total_elements);
  } else {
    // Add other activation functions here using else if
    printf("Activation function not implemented\n");
  }
  cudaDeviceSynchronize();
  cublasDestroy(handle);
}

__device__ float compute_activation_derivative(float z,
                                               ActivationType activation) {
  float sigma_prime = 1.0f; // Default for linear activation
  if (activation == ACTIVATION_SIGMOID) {
    float sigmoid = 1.0f / (1.0f + expf(-z));
    sigma_prime = sigmoid * (1.0f - sigmoid);
  } else if (activation == ACTIVATION_RELU) {
    sigma_prime = z > 0 ? 1.0f : 0.0f;
  }
  // Add other activations if necessary
  return sigma_prime;
}

__global__ void compute_output_delta(float *d_output_delta, float *d_output,
                                     float *d_z, float *d_labels, int size,
                                     ActivationType activation,
                                     LossFunction loss_function) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float delta = d_output[idx] - d_labels[idx]; // (a^L - y)
    float sigma_prime = compute_activation_derivative(d_z[idx], activation);

    if (loss_function == LOSS_MSE) {
      // δ^L = (a^L - y) * σ'(z^L)
      d_output_delta[idx] = delta * sigma_prime;
    } else if (loss_function == LOSS_CROSSENTROPY &&
               activation == ACTIVATION_SIGMOID) {
      // δ^L = (a^L - y)
      d_output_delta[idx] = delta;
    } else {
      // Handle other combinations or raise an error
      // For example, cross-entropy with softmax, etc.
      // Here, we'll default to multiplying by sigma_prime
      d_output_delta[idx] = delta * sigma_prime;
    }
  }
}

__global__ void set_ones(float *d_ones, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    d_ones[idx] = 1.0f;
  }
}

void backward_output_layer(Layer *layer, float *d_labels, int batch_size) {
  int size = batch_size * layer->output_size;
  int threads = THREADS_PER_BLOCK;
  int blocks = (size + threads - 1) / threads;

  // Launch kernel to compute δ^L
  compute_output_delta<<<blocks, threads>>>(
      layer->d_output_delta, layer->d_output, layer->d_z, d_labels, size,
      layer->activation, layer->loss_function);
  cudaDeviceSynchronize();

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  // Compute d_weights_grad = (a^{L-1})^T * δ^L
  cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    layer->input_size,  // M
                                    layer->output_size, // N
                                    batch_size,         // K
                                    &alpha,
                                    layer->d_input,        // A
                                    batch_size,            // lda
                                    layer->d_output_delta, // B
                                    batch_size,            // ldb
                                    &beta,
                                    layer->d_weights_grad, // C
                                    layer->input_size);    // ldc

  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm failed in backward_output_layer\n");
  }

  // Compute d_biases_grad = sum over batch of δ^L
  // Allocate and initialize d_ones
  float *d_ones;
  cudaMalloc(&d_ones, batch_size * sizeof(float));
  int num_blocks = (batch_size + threads - 1) / threads;
  set_ones<<<num_blocks, threads>>>(d_ones, batch_size);
  cudaDeviceSynchronize();

  // Compute d_biases_grad = δ^L^T * d_ones
  cublasStatus_t stat2 = cublasSgemv(handle, CUBLAS_OP_T,
                                     batch_size,         // m
                                     layer->output_size, // n
                                     &alpha,
                                     layer->d_output_delta, // A
                                     batch_size,            // lda
                                     d_ones,                // x
                                     1,                     // incx
                                     &beta,
                                     layer->d_biases_grad, // y
                                     1);                   // incy

  if (stat2 != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemv failed in backward_output_layer\n");
  }

  // Clean up
  cudaFree(d_ones);
  cublasDestroy(handle);
}

__global__ void compute_hidden_delta(float *d_current_delta, float *d_z,
                                     ActivationType activation, int size) {
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

__global__ void multiply_by_sigma_prime(float *d_input_grad, float *d_z,
                                        int size, ActivationType activation) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
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
    d_input_grad[idx] *= sigma_prime;
  }
}

void backward_layer(Layer *current_layer, Layer *next_layer,
                    float *d_output_delta, float *d_input_grad,
                    int batch_size) {
  // current_layer: the layer we are computing gradients for
  // next_layer: the layer after current_layer
  // d_output_delta: gradient from the next layer (δ^{l+1})
  // d_input_grad: output gradient (δ^{l}) to be computed
  // batch_size: number of samples in the batch

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  // Dimensions:
  // next_layer->output_size: n
  // next_layer->input_size: k (should be equal to current_layer->output_size)
  // batch_size: m

  // We assume that next_layer->d_weights has dimensions [k x n]
  // d_output_delta has dimensions [m x n]
  // We want to compute d_input_grad = d_output_delta * W^{l+1}

  // However, cuBLAS uses column-major order, so we need to adjust the
  // parameters accordingly.

  // Compute: d_input_grad = d_output_delta * W^{l+1}
  // In cuBLAS terms: C = alpha * A * B + beta * C
  // A: [n x m] (transpose of d_output_delta which is [m x n])
  // B: [n x k] (W^{l+1} which is [k x n] in row-major, so no transpose)
  // C: [m x k] (d_input_grad)

  // To match dimensions, we'll use CUBLAS_OP_T on d_output_delta
  // Result will be [n x k] which needs to be transposed to [k x n]

  // But to compute [m x k], we set:
  // A: d_output_delta (m x n)
  // B: next_layer->d_weights (n x k)
  // C: d_input_grad (m x k)

  // Therefore, no transposition is needed if we treat data as row-major.

  // However, cuBLAS expects column-major, so effectively:
  // A (n x m) = d_output_delta^T
  // B (k x n) = W^{l+1}^T
  // C (k x m) = d_input_grad^T

  // Therefore, we set the operation as:
  // C = W^{l+1}^T * d_output_delta^T
  // Which in cuBLAS is:
  // CUBLAS_OP_N, CUBLAS_OP_N

  // So, setting up the SGEMM parameters:
  cublasStatus_t stat =
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  current_layer->output_size, // m: k
                  batch_size,                 // n: m
                  next_layer->output_size,    // k: n
                  &alpha,
                  next_layer->d_weights,      // A: [k x n]
                  current_layer->output_size, // lda: leading dimension of A
                  d_output_delta,             // B: [n x m]
                  next_layer->output_size,    // ldb: leading dimension of B
                  &beta,
                  d_input_grad,              // C: [k x m]
                  current_layer->output_size // ldc: leading dimension of C
      );

  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm failed in backward_layer\n");
  }

  // Now, d_input_grad has dimensions [k x m] in column-major
  // We need to transpose it to [m x k] to match the expected row-major format
  // Alternatively, we can treat it as is for further processing

  // Apply the activation derivative: δ^{l} = d_input_grad^T * σ'(z^l)
  int size = current_layer->output_size * batch_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  multiply_by_sigma_prime<<<num_blocks, threads_per_block>>>(
      d_input_grad, current_layer->d_z, size, current_layer->activation);
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
  const char *activation_str;
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
    activation_str =
        "unknown or not implemented (maybe forgot to add it to print_layer)";
    break;
  }
  printf("Activation: %s\n", activation_str);

  // Print a snippet of weights and biases
  int num_weights_to_print = 5;
  printf("Weights (first %d values):\n", num_weights_to_print);
  for (int i = 0;
       i < num_weights_to_print && i < layer->input_size * layer->output_size;
       ++i) {
    printf("%f ", layer->weights[i]);
  }
  printf("\nBiases (first %d values):\n", num_weights_to_print);
  for (int i = 0; i < num_weights_to_print && i < layer->output_size; ++i) {
    printf("%f ", layer->biases[i]);
  }
  printf("\n");
}