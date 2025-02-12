#include "../../tests/unity/unity.h"
#include "../../include/layers.h"
#include <cstdio>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <math.h>

// Forward declarations
void setUp(void);
void tearDown(void);
void test_forward_layer(void);

// Test-specific helper functions
void allocate_device_memory(float **d_ptr, size_t size) {
    cudaError_t err = cudaMalloc(d_ptr, size);
    TEST_ASSERT_EQUAL(cudaSuccess, err);
}

void copy_to_device(float *d_ptr, float *h_ptr, size_t size) {
    cudaError_t err = cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    TEST_ASSERT_EQUAL(cudaSuccess, err);
}

void copy_from_device(float *h_ptr, float *d_ptr, size_t size) {
    cudaError_t err = cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    TEST_ASSERT_EQUAL(cudaSuccess, err);
}

float host_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float random_float(float min, float max) {
    return min + ((float)rand() / RAND_MAX) * (max - min);
}
/*
void test_forward_layer_large(void) {
    // Seed the random number generator
    

    // Initialize test parameters
    int batch_size = 1024;
    int input_size = 512;
    int output_size = 256;

    // Allocate host memory
    float *h_input = (float *)malloc(batch_size * input_size * sizeof(float));
    float *h_weights = (float *)malloc(output_size * input_size * sizeof(float));
    float *h_biases = (float *)malloc(output_size * sizeof(float));
    float *h_output = (float *)malloc(batch_size * output_size * sizeof(float));
    float *expected_output = (float *)malloc(batch_size * output_size * sizeof(float));
    float *expected_sigmoid_output = (float *)malloc(batch_size * output_size * sizeof(float));

    // Initialize input, weights, and biases with random values
    for (int i = 0; i < batch_size * input_size; i++) {
        h_input[i] = random_float(-1.0f, 1.0f);
    }
    for (int i = 0; i < output_size * input_size; i++) {
        h_weights[i] = random_float(-1.0f, 1.0f);
    }
    for (int i = 0; i < output_size; i++) {
        h_biases[i] = random_float(-0.5f, 0.5f);
    }

    // Compute expected output on host (linear activation)
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_size; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                sum += h_input[b * input_size + i] * h_weights[o * input_size + i];
            }
            sum += h_biases[o];
            expected_output[b * output_size + o] = sum;
        }
    }

    // Allocate device memory
    float *d_input, *d_weights, *d_biases, *d_output;
    allocate_device_memory(&d_input, batch_size * input_size * sizeof(float));
    allocate_device_memory(&d_weights, output_size * input_size * sizeof(float));
    allocate_device_memory(&d_biases, output_size * sizeof(float));
    allocate_device_memory(&d_output, batch_size * output_size * sizeof(float));

    // Copy data to device
    copy_to_device(d_input, h_input, batch_size * input_size * sizeof(float));
    copy_to_device(d_weights, h_weights, output_size * input_size * sizeof(float));
    copy_to_device(d_biases, h_biases, output_size * sizeof(float));

    // Create and initialize layer with linear activation
    Layer layer = {
        .input_size = input_size,
        .output_size = output_size,
        .activation = ACTIVATION_LINEAR,
        .d_weights = d_weights,
        .d_biases = d_biases,
    };

    // Call the function under test (forward pass with linear activation)
    forward_layer(&layer, d_input, d_output, batch_size);

    // Copy results back to host
    copy_from_device(h_output, d_output, batch_size * output_size * sizeof(float));

    // Verify results for linear activation
    for (int i = 0; i < batch_size * output_size; i++) {
        if (fabsf(expected_output[i] - h_output[i]) > 1e-4f) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected_output[i], h_output[i]);
            // You can replace the following line with your testing framework's assert
            // For example: TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_output[i], h_output[i]);
            exit(EXIT_FAILURE);
        }
    }

    // Compute expected sigmoid output on host
    for (int i = 0; i < batch_size * output_size; i++) {
        expected_sigmoid_output[i] = host_sigmoid(expected_output[i]);
    }

    // Set activation to sigmoid and perform forward pass
    layer.activation = ACTIVATION_SIGMOID;
    forward_layer(&layer, d_input, d_output, batch_size);
    copy_from_device(h_output, d_output, batch_size * output_size * sizeof(float));

    // Verify results for sigmoid activation
    for (int i = 0; i < batch_size * output_size; i++) {
        if (fabsf(expected_sigmoid_output[i] - h_output[i]) > 1e-4f) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected_sigmoid_output[i], h_output[i]);
            // Replace with your testing framework's assert if needed
            exit(EXIT_FAILURE);
        }
    }

    // Cleanup
    free(h_input);
    free(h_weights);
    free(h_biases);
    free(h_output);
    free(expected_output);
    free(expected_sigmoid_output);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
}
*/

void test_forward_layer(void) {
    // Initialize test data
    int batch_size = 2;
    int input_size = 3;
    int output_size = 2;
    // Input data (column-major order)
    float h_input[] = {
        0.1f, 0.3f, 0.5f,  // Column 1
        0.2f, 0.4f, 0.6f   // Column 2
    }; // size: input_size x batch_size (column-major order)

    // Weights (column-major order)
    float h_weights[] = {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f
    }; // size: output_size x input_size (column-major order)

    // Biases
    float h_biases[] = {
        0.01f,
        0.02f
    }; // size: output_size

    float h_output[4] = {0}; // size: output_size x batch_size

    // Expected output (column-major order)
    float expected_output[] = {
        0.36, 0.46f,   // Neuron 1 outputs
        0.45f, 0.58f  // Neuron 2 outputs
    };

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc((void**)&d_output, batch_size * output_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize layer
    Layer layer;
    layer.m = batch_size;
    layer.n_in = input_size;
    layer.n_out = output_size;
    layer.aktfunc = ACTIVATION_LINEAR;  // Assuming LINEAR is defined in ActivationType enum

    // Allocate and set weights and biases
    size_t w_size = input_size * output_size * sizeof(float);
    size_t b_size = output_size * sizeof(float);

    cudaMalloc((void**)&layer.w_d, w_size);
    cudaMalloc((void**)&layer.b_d, b_size);

    cudaMemcpy(layer.w_d, h_weights, w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(layer.b_d, h_biases, b_size, cudaMemcpyHostToDevice);

    // Allocate memory for Z and A
    cudaMalloc((void**)&layer.Z_d, output_size * batch_size * sizeof(float));
    cudaMalloc((void**)&layer.A_d, output_size * batch_size * sizeof(float));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Call the function under test
    layer_forward(&layer, d_input, handle);

    // Copy results back to host
    cudaMemcpy(h_output, layer.Z_d, output_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);


    // Verify results
    for (int i = 0; i < output_size * batch_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_output[i], h_output[i]);
    }

    layer.aktfunc = ACTIVATION_SIGMOID;  // Assuming SIGMOID is defined in ActivationType enum
    layer_forward(&layer, d_input, handle);
    cudaMemcpy(h_output, layer.A_d, output_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    float expected_sigmoid_output[4];
    for (int i = 0; i < output_size * batch_size; i++) {
        expected_sigmoid_output[i] = host_sigmoid(expected_output[i]);
    }

    for (int i = 0; i < output_size * batch_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_sigmoid_output[i], h_output[i]);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(layer.w_d);
    cudaFree(layer.b_d);
    cudaFree(layer.Z_d);
    cudaFree(layer.A_d);

    cublasDestroy(handle);
}

void test_backward_output_layer(void) {
    // Initialize test data
    int batch_size = 2;
    int input_size = 3;
    int output_size = 2;

    // Inputs: A (activation from forward pass) and Y (ground truth)
    float h_A[] = {
        0.1f, 0.2f,   // Column 1
        0.3f, 0.4f,   // Column 2
    }; // size: output_size x batch_size (column-major order)

    float h_Y[] = {
        0.15f, 0.25f, // Column 1
        0.35f, 0.45f  // Column 2
    }; // size: output_size x batch_size (column-major order)

    // Input to the previous layer
    float h_A_prev[] = {
        0.1f, 0.3f, 0.5f,  // Column 1
        0.2f, 0.4f, 0.6f   // Column 2
    }; // size: input_size x batch_size (column-major order)

    // Expected gradients (computed offline)
    float expected_dW[] = {
        -0.015f/2, -0.015f/2,
        -0.035f/2, -0.035f/2,
        -0.055f/2, -0.055f/2
    }; // size: output_size x input_size

    float expected_db[] = {
        -0.05f,
        -0.05f
    }; // size: output_size

    float h_dW[6] = {0}; // size: output_size x input_size
    float h_db[2] = {0}; // size: output_size

    // Allocate device memory
    float *d_A, *d_Y, *d_A_prev;
    cudaMalloc((void**)&d_A, output_size * batch_size * sizeof(float));
    cudaMalloc((void**)&d_Y, output_size * batch_size * sizeof(float));
    cudaMalloc((void**)&d_A_prev, input_size * batch_size * sizeof(float));

    cudaMemcpy(d_A, h_A, output_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, output_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_prev, h_A_prev, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize layer
    Layer layer;
    layer.m = batch_size;
    layer.n_in = input_size;
    layer.n_out = output_size;

    // Allocate memory for gradients
    cudaMalloc((void**)&layer.dZ_d, output_size * batch_size * sizeof(float));
    cudaMalloc((void**)&layer.dW_d, output_size * input_size * sizeof(float));
    cudaMalloc((void**)&layer.db_d, output_size * sizeof(float));
    cudaMalloc((void**)&layer.A_d, output_size * batch_size * sizeof(float));

    cudaMemcpy(layer.A_d, d_A, output_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Call the function under test
    backward_output_layer(&layer, d_Y, d_A_prev, handle);

    // Copy results back to host
    cudaMemcpy(h_dW, layer.dW_d, output_size * input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db, layer.db_d, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < output_size * input_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_dW[i], h_dW[i]);
    }

    for (int i = 0; i < output_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_db[i], h_db[i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_Y);
    cudaFree(d_A_prev);
    cudaFree(layer.dZ_d);
    cudaFree(layer.dW_d);
    cudaFree(layer.db_d);

    cublasDestroy(handle);
}

void test_backward_layer(void) {
    // Initialize test data
    int batch_size = 2;
    int current_layer_size = 3;    // n_out_current
    int next_layer_size = 2;       // n_out_next
    int previous_layer_size = 3;   // n_in

    // Weights of the next layer (next_layer_size x current_layer_size)
    float h_W_next[] = {
        0.1f, 0.3f, 0.5f,  // Neuron 1 weights
        0.2f, 0.4f, 0.6f   // Neuron 2 weights
    }; // Transposed later

    // Transpose h_W_next to match column-major order
    float h_W_next_transposed[] = {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f
    }; // Size: current_layer_size x next_layer_size

    // dZ_next_d: (next_layer_size x batch_size)
    float h_dZ_next[] = {
        0.01f, 0.02f,  // Neuron 1
        0.03f, 0.04f   // Neuron 2
    };

    // A_prev_d: (previous_layer_size x batch_size)
    float h_A_prev[] = {
        0.1f, 0.3f, 0.5f,  // Sample 1
        0.2f, 0.4f, 0.6f   // Sample 2
    };

    // Initialize layer
    Layer layer;
    layer.m = batch_size;
    layer.n_in = previous_layer_size;
    layer.n_out = current_layer_size;
    layer.aktfunc = ACTIVATION_RELU;

    // Allocate and check device memory
    size_t size_W_next = current_layer_size * next_layer_size * sizeof(float);
    size_t size_dZ_next = next_layer_size * batch_size * sizeof(float);
    size_t size_A_prev = previous_layer_size * batch_size * sizeof(float);
    size_t size_Z = current_layer_size * batch_size * sizeof(float);

    float *d_W_next, *d_dZ_next, *d_A_prev;
    cudaError_t err;

    err = cudaMalloc((void**)&d_W_next, size_W_next);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA malloc d_W_next failed\n"); exit(EXIT_FAILURE); }
    err = cudaMalloc((void**)&d_dZ_next, size_dZ_next);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA malloc d_dZ_next failed\n"); exit(EXIT_FAILURE); }
    err = cudaMalloc((void**)&d_A_prev, size_A_prev);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA malloc d_A_prev failed\n"); exit(EXIT_FAILURE); }
    // Copy data to device
    err = cudaMemcpy(d_W_next, h_W_next_transposed, size_W_next, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy d_W_next failed\n"); exit(EXIT_FAILURE); }
    err = cudaMemcpy(d_dZ_next, h_dZ_next, size_dZ_next, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy d_dZ_next failed\n"); exit(EXIT_FAILURE); }
    err = cudaMemcpy(d_A_prev, h_A_prev, size_A_prev, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy d_A_prev failed\n"); exit(EXIT_FAILURE); }
    // Allocate layer.dZ_d and layer.Z_d
    err = cudaMalloc((void**)&layer.dZ_d, size_Z);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA malloc layer.dZ_d failed\n"); exit(EXIT_FAILURE); }
    err = cudaMalloc((void**)&layer.Z_d, size_Z);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA malloc layer.Z_d failed\n"); exit(EXIT_FAILURE); }

    // Initialize layer.Z_d
    float h_Z[] = {
        0.5f, 0.6f, 0.7f,  // Sample 1
        0.6f, 0.7f, -0.8f   // Sample 2
    };

    err = cudaMemcpy(layer.Z_d, h_Z, size_Z, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy layer.Z_d failed\n"); exit(EXIT_FAILURE); }

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }

    // Call the function under test
    backward_layer(&layer, d_W_next, d_dZ_next, d_A_prev, next_layer_size, handle);

    // Copy results back to host
    float h_dZ[6];
    err = cudaMemcpy(h_dZ, layer.dZ_d, size_Z, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA memcpy h_dZ failed\n"); exit(EXIT_FAILURE); }

    // Expected output (computed manually)
    float expected_dZ[] = {
        0.005f, 0.011f, 0.017f,
        0.011f, 0.025f, 0.0f
    };

    // Verify results
    for (int i = 0; i < layer.n_out * layer.m; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_dZ[i], h_dZ[i]);
    }

    // Cleanup
    cudaFree(d_W_next);
    cudaFree(d_dZ_next);
    cudaFree(d_A_prev);
    cudaFree(layer.dZ_d);
    cudaFree(layer.Z_d);

    cublasDestroy(handle);
}

/*
void test_backward_output_layer(void) {
    
    // Initialize test data
    int batch_size = 2;
    int output_size = 2;

    // z values from the forward pass
    float h_z[] = {
        0.4f, -1.0f,
        0.5f, 0.0f
    };

    // Output activations (a^L) from the forward pass
    float h_output[4];
    for (int i = 0; i < 4; i++) {
        h_output[i] = 1.0f / (1.0f + expf(-h_z[i]));
    }

    // Labels (y)
    float h_labels[] = {
        0.5f, 0.0f,
        1.0f, 0.25f
    };
    
    // Expected δ^L for linear activation: (a^L - y) * 1
    float expected_output_delta_linear[4];
    for (int i = 0; i < 4; i++) {
        expected_output_delta_linear[i] = h_output[i] - h_labels[i];
    }
    
    // Expected δ^L for sigmoid activation: (a^L - y) * sigmoid'(z^L)
    float expected_output_delta_sigmoid[4];
    for (int i = 0; i < 4; i++) {
        float sigmoid = host_sigmoid(h_z[i]);
        float sigma_prime = sigmoid * (1.0f - sigmoid);
        expected_output_delta_sigmoid[i] = (h_output[i] - h_labels[i]) * sigma_prime;
    }
    
    // Allocate device memory
    float *d_output, *d_z, *d_labels, *d_output_delta;
    allocate_device_memory(&d_output, batch_size * output_size * sizeof(float));
    allocate_device_memory(&d_z, batch_size * output_size * sizeof(float));
    allocate_device_memory(&d_labels, batch_size * output_size * sizeof(float));
    allocate_device_memory(&d_output_delta, batch_size * output_size * sizeof(float));
    
    // Copy data to device
    copy_to_device(d_output, h_output, batch_size * output_size * sizeof(float));
    copy_to_device(d_z, h_z, batch_size * output_size * sizeof(float));
    copy_to_device(d_labels, h_labels, batch_size * output_size * sizeof(float));
    
    // Initialize d_output_delta to zero
    cudaMemset(d_output_delta, 0, batch_size * output_size * sizeof(float));
    
    // Create and initialize layer for linear activation
    Layer layer_linear = {
        .input_size = 3, // Arbitrary, not used in this test
        .output_size = output_size,
        .activation = ACTIVATION_LINEAR,
        .d_weights = NULL, // Not used in this test
        .d_biases = NULL,  // Not used in this test
        .d_output = d_output,
        .d_z = d_z
    };
    
    // Call the function under test for linear activation
    backward_output_layer(&layer_linear, d_labels, d_output_delta, batch_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after backward_output_layer (linear): %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // Copy results back to host
    float h_output_delta_linear[4] = {0};
    
    copy_from_device(h_output_delta_linear, d_output_delta, batch_size * output_size * sizeof(float));
    
    // Verify results for linear activation
    for (int i = 0; i < batch_size * output_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_output_delta_linear[i], h_output_delta_linear[i]);
    }
    
    // Reset d_output_delta to zero for sigmoid activation test
    cudaMemset(d_output_delta, 0, batch_size * output_size * sizeof(float));
    
    // Create and initialize layer for sigmoid activation
    Layer layer_sigmoid = {
        .input_size = 3, // Arbitrary, not used in this test
        .output_size = output_size,
        .activation = ACTIVATION_SIGMOID,
        .d_weights = NULL, // Not used in this test
        .d_biases = NULL,  // Not used in this test
        .d_output = d_output,
        .d_z = d_z
    };
    
    // Call the function under test for sigmoid activation
    backward_output_layer(&layer_sigmoid, d_labels, d_output_delta, batch_size);
    
    // Copy results back to host
    float h_output_delta_sigmoid[4] = {0};
    copy_from_device(h_output_delta_sigmoid, d_output_delta, batch_size * output_size * sizeof(float));
    
    // Verify results for sigmoid activation
    for (int i = 0; i < batch_size * output_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_output_delta_sigmoid[i], h_output_delta_sigmoid[i]);
    }
    
    // Cleanup
    cudaFree(d_output);
    cudaFree(d_z);
    cudaFree(d_labels);
    cudaFree(d_output_delta);
}

void test_backward_output_layer_large(void){
    int batch_size = 2;
    int output_size = 2;
    // Allocate host memory
    float *h_z = (float *)malloc(batch_size * output_size * sizeof(float));
    float *h_labels = (float *)malloc(batch_size * output_size * sizeof(float));
    float *expected_output = (float *)malloc(batch_size * output_size * sizeof(float));
    float *expected_sigmoid_output = (float *)malloc(batch_size * output_size * sizeof(float));

    // Initialize input, weights, and biases with random values
    for (int i = 0; i < batch_size * output_size; i++) {
        h_z[i] = random_float(-1.0f, 1.0f);
        h_labels[i] = random_float(0.0f, 1.0f);
    }

    float h_output[batch_size * output_size];
    for (int i = 0; i < batch_size*output_size; i++) {
        h_output[i] = 1.0f / (1.0f + expf(-h_z[i]));
    }

    float expected_output_delta_linear[batch_size * output_size];
    for (int i = 0; i < batch_size*output_size; i++) {
        expected_output_delta_linear[i] = h_output[i] - h_labels[i];
    }

    float expected_output_delta_sigmoid[batch_size * output_size];
    for (int i = 0; i < batch_size*output_size; i++) {
        float sigmoid = host_sigmoid(h_z[i]);
        float sigma_prime = sigmoid * (1.0f - sigmoid);
        expected_output_delta_sigmoid[i] = (h_output[i] - h_labels[i]) * sigma_prime;
    }

        // Allocate device memory
    float *d_output, *d_z, *d_labels, *d_output_delta;
    allocate_device_memory(&d_output, batch_size * output_size * sizeof(float));
    allocate_device_memory(&d_z, batch_size * output_size * sizeof(float));
    allocate_device_memory(&d_labels, batch_size * output_size * sizeof(float));
    allocate_device_memory(&d_output_delta, batch_size * output_size * sizeof(float));
    
    // Copy data to device
    copy_to_device(d_output, h_output, batch_size * output_size * sizeof(float));
    copy_to_device(d_z, h_z, batch_size * output_size * sizeof(float));
    copy_to_device(d_labels, h_labels, batch_size * output_size * sizeof(float));
    
    // Initialize d_output_delta to zero
    cudaMemset(d_output_delta, 0, batch_size * output_size * sizeof(float));
    
    // Create and initialize layer for linear activation
    Layer layer_linear = {
        .input_size = 3, // Arbitrary, not used in this test
        .output_size = output_size,
        .activation = ACTIVATION_LINEAR,
        .d_weights = NULL, // Not used in this test
        .d_biases = NULL,  // Not used in this test
        .d_output = d_output,
        .d_z = d_z
    };
    
    // Call the function under test for linear activation
    backward_output_layer(&layer_linear, d_labels, d_output_delta, batch_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after backward_output_layer (linear): %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // Copy results back to host
    float h_output_delta_linear[4] = {0};
    
    copy_from_device(h_output_delta_linear, d_output_delta, batch_size * output_size * sizeof(float));
    
    // Verify results for linear activation
    for (int i = 0; i < batch_size * output_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_output_delta_linear[i], h_output_delta_linear[i]);
    }
    
    // Reset d_output_delta to zero for sigmoid activation test
    cudaMemset(d_output_delta, 0, batch_size * output_size * sizeof(float));
    
    // Create and initialize layer for sigmoid activation
    Layer layer_sigmoid = {
        .input_size = 3, // Arbitrary, not used in this test
        .output_size = output_size,
        .activation = ACTIVATION_SIGMOID,
        .d_weights = NULL, // Not used in this test
        .d_biases = NULL,  // Not used in this test
        .d_output = d_output,
        .d_z = d_z
    };
    
    // Call the function under test for sigmoid activation
    backward_output_layer(&layer_sigmoid, d_labels, d_output_delta, batch_size);
    
    // Copy results back to host
    float *h_output_delta_sigmoid = (float *)malloc(batch_size * output_size * sizeof(float));
    memset(h_output_delta_sigmoid, 0, batch_size * output_size * sizeof(float));
    copy_from_device(h_output_delta_sigmoid, d_output_delta, batch_size * output_size * sizeof(float));
    
    // Verify results for sigmoid activation
    for (int i = 0; i < batch_size * output_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_output_delta_sigmoid[i], h_output_delta_sigmoid[i]);
    }
    
    // Cleanup
    cudaFree(d_output);
    cudaFree(d_z);
    cudaFree(d_labels);
    cudaFree(d_output_delta);
}

void test_compute_output_delta(void) {
    // Initialize test data
    int batch_size = 2;
    int output_size = 2;
    int size = batch_size * output_size;

    // z values from the forward pass
    float h_z[] = {
        0.4f, -1.0f,
        0.5f, 0.0f
    };

    // Output activations (a^L) from the forward pass (sigmoid activation)
    float h_output_sigmoid[4];
    for (int i = 0; i < size; i++) {
        h_output_sigmoid[i] = host_sigmoid(h_z[i]);
    }

    // Output activations (a^L) for linear activation (assuming linear activation is identity)
    float h_output_linear[4];
    for (int i = 0; i < size; i++) {
        h_output_linear[i] = h_z[i]; // For linear activation, a = z
    }

    // Labels (y)
    float h_labels[] = {
        0.5f, 0.0f,
        1.0f, 0.25f
    };

    // Expected δ^L for linear activation: (a^L - y) * 1
    float expected_output_delta_linear[4];
    for (int i = 0; i < size; i++) {
        expected_output_delta_linear[i] = h_output_linear[i] - h_labels[i];
    }

    // Expected δ^L for sigmoid activation: (a^L - y) * sigmoid'(z^L)
    float expected_output_delta_sigmoid[4];
    for (int i = 0; i < size; i++) {
        float sigmoid = h_output_sigmoid[i];
        float sigma_prime = sigmoid * (1.0f - sigmoid);
        expected_output_delta_sigmoid[i] = (h_output_sigmoid[i] - h_labels[i]) * sigma_prime;
    }

    // Allocate device memory
    float *d_output_linear, *d_z_linear, *d_labels_linear, *d_output_delta_linear;
    float *d_output_sigmoid, *d_z_sigmoid, *d_labels_sigmoid, *d_output_delta_sigmoid;

    // Linear Activation Allocation
    allocate_device_memory(&d_output_linear, size * sizeof(float));
    allocate_device_memory(&d_z_linear, size * sizeof(float));
    allocate_device_memory(&d_labels_linear, size * sizeof(float));
    allocate_device_memory(&d_output_delta_linear, size * sizeof(float));

    // Sigmoid Activation Allocation
    allocate_device_memory(&d_output_sigmoid, size * sizeof(float));
    allocate_device_memory(&d_z_sigmoid, size * sizeof(float));
    allocate_device_memory(&d_labels_sigmoid, size * sizeof(float));
    allocate_device_memory(&d_output_delta_sigmoid, size * sizeof(float));

    // Copy data to device for Linear Activation
    copy_to_device(d_output_linear, h_output_linear, size * sizeof(float));
    copy_to_device(d_z_linear, h_z, size * sizeof(float));
    copy_to_device(d_labels_linear, h_labels, size * sizeof(float));
    cudaMemset(d_output_delta_linear, 0, size * sizeof(float));

    // Copy data to device for Sigmoid Activation
    copy_to_device(d_output_sigmoid, h_output_sigmoid, size * sizeof(float));
    copy_to_device(d_z_sigmoid, h_z, size * sizeof(float));
    copy_to_device(d_labels_sigmoid, h_labels, size * sizeof(float));
    cudaMemset(d_output_delta_sigmoid, 0, size * sizeof(float));

    // Define block and grid sizes
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Create and initialize Layer for Linear Activation
    Layer layer_linear = {
        .input_size = 3, // Arbitrary, not used in this test
        .output_size = output_size,
        .activation = ACTIVATION_LINEAR,
        .loss_function = LOSS_MSE,
        .d_weights = NULL, // Not used in this test
        .d_biases = NULL,  // Not used in this test
        .d_output = d_output_linear,
        .d_z = d_z_linear
    };

    // Create and initialize Layer for Sigmoid Activation
    Layer layer_sigmoid = {
        .input_size = 3, // Arbitrary, not used in this test
        .output_size = output_size,
        .activation = ACTIVATION_SIGMOID,
        .loss_function = LOSS_MSE,
        .d_weights = NULL, // Not used in this test
        .d_biases = NULL,  // Not used in this test
        .d_output = d_output_sigmoid,
        .d_z = d_z_sigmoid
    };

    // Compute delta for Linear Activation
    compute_output_delta<<<blocks, threads>>>(
        d_output_delta_linear, // delta
        d_output_linear,       // a^L
        d_z_linear,            // z^L
        d_labels_linear,       // y
        size,
        layer_linear.activation,
        layer_linear.loss_function
    );
    cudaDeviceSynchronize();

    compute_output_delta<<<blocks, threads>>>(
        d_output_delta_sigmoid, // delta
        d_output_sigmoid,       // a^L
        d_z_sigmoid,            // z^L
        d_labels_sigmoid,       // y
        size,
        layer_sigmoid.activation,
        layer_sigmoid.loss_function
    );
    cudaDeviceSynchronize();

    // Copy results back to host
    float h_output_delta_linear[4];
    float h_output_delta_sigmoid[4];
    copy_from_device(h_output_delta_linear, d_output_delta_linear, size * sizeof(float));
    copy_from_device(h_output_delta_sigmoid, d_output_delta_sigmoid, size * sizeof(float));

    for (int i = 0; i < size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f,h_output_delta_linear[i], expected_output_delta_linear[i]);
    }


    for (int i = 0; i < size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f,h_output_delta_sigmoid[i],  expected_output_delta_sigmoid[i]);
    }

    // Free device memory
    cudaFree(d_output_linear);
    cudaFree(d_z_linear);
    cudaFree(d_labels_linear);
    cudaFree(d_output_delta_linear);

    cudaFree(d_output_sigmoid);
    cudaFree(d_z_sigmoid);
    cudaFree(d_labels_sigmoid);
    cudaFree(d_output_delta_sigmoid);
}

void test_backward_layer(void) {
    // Initialize test data
    int batch_size = 2;
    int input_size = 3;
    int output_size = 2;

    // Define next_layer with known weights
    float h_next_weights[] = {
        0.1f, 0.2f, 0.3f, // W^{l+1} for output neuron 1
        0.4f, 0.5f, 0.6f  // W^{l+1} for output neuron 2
    };

    // Define d_output_delta from next layer (δ^{l+1})
    float h_output_delta[] = {
        0.1f, -0.2f,
        0.3f, 0.4f
    };

    // Define z values for current_layer
    float h_z[] = {
        0.5f, -0.5f, 1.0f,  // Batch 1: z1, z2, z3
        1.0f, -1.0f, 0.0f   // Batch 2: z1, z2, z3
    };

    // Expected sigma_prime for sigmoid activation
    float expected_sigma_prime[input_size * batch_size];
    for (int i = 0; i < input_size * batch_size; i++) {
        float sigmoid = host_sigmoid(h_z[i]);
        expected_sigma_prime[i] = sigmoid * (1.0f - sigmoid);
    }

    // Compute expected d_input_grad on host
    // d_input_grad = W^{l+1}^T * d_output_delta
    // Then, multiply by sigma_prime(z^l)

    float expected_d_input_grad[input_size * batch_size];
    // Initialize to zero
    memset(expected_d_input_grad, 0, input_size * batch_size * sizeof(float));

    // Compute W^{l+1}^T * d_output_delta
    for (int i = 0; i < batch_size; i++) { // For each sample
        for (int j = 0; j < input_size; j++) { // For each input neuron
            for (int k = 0; k < output_size; k++) { // For each output neuron
                expected_d_input_grad[i * input_size + j] += h_next_weights[k * input_size + j] * h_output_delta[i * output_size + k];
            }
            // Multiply by sigma_prime(z^l)
            expected_d_input_grad[i * input_size + j] *= expected_sigma_prime[i * input_size + j];
        }
    }

    // Allocate device memory
    float *d_next_weights, *d_output_delta_dev, *d_z, *d_input_grad;
    allocate_device_memory(&d_next_weights, output_size * input_size * sizeof(float));
    allocate_device_memory(&d_output_delta_dev, batch_size * output_size * sizeof(float));
    allocate_device_memory(&d_z, batch_size * input_size * sizeof(float));
    allocate_device_memory(&d_input_grad, batch_size * input_size * sizeof(float));

    // Copy data to device
    copy_to_device(d_next_weights, h_next_weights, output_size * input_size * sizeof(float));
    copy_to_device(d_output_delta_dev, h_output_delta, batch_size * output_size * sizeof(float));
    copy_to_device(d_z, h_z, batch_size * input_size * sizeof(float));

    // Initialize d_input_grad to zero
    cudaMemset(d_input_grad, 0, batch_size * input_size * sizeof(float));

    // Create Layer structures
    Layer current_layer = {
        .input_size = input_size,
        .output_size = input_size, // Assuming input_size == current_layer->output_size
        .activation = ACTIVATION_SIGMOID,
        .d_weights = NULL, // Not used in this test
        .d_biases = NULL,  // Not used in this test
        .d_output = NULL,  // Not used in this test
        .d_z = d_z
    };

    Layer next_layer = {
        .input_size = input_size,
        .output_size = output_size,
        .activation = ACTIVATION_SIGMOID, // Activation of next layer
        .d_weights = d_next_weights,
        .d_biases = NULL, // Not used in this test
        .d_output = NULL,  // Not used in this test
        .d_z = NULL        // Not used in this test
    };

    // Call the function under test
    backward_layer(&current_layer, &next_layer, d_output_delta_dev, d_input_grad, batch_size);

    // Copy results back to host
    float h_input_grad[input_size * batch_size];
    copy_from_device(h_input_grad, d_input_grad, batch_size * input_size * sizeof(float));

    // Verify results
    for (int i = 0; i < batch_size * input_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_d_input_grad[i], h_input_grad[i]);
    }

    // Cleanup
    cudaFree(d_next_weights);
    cudaFree(d_output_delta_dev);
    cudaFree(d_z);
    cudaFree(d_input_grad);
}

void test_backward_layer_large(void) {
    // Initialize test parameters
    int batch_size = 128;
    int input_size = 64;
    int output_size = 32;

    // Seed the random number generator for reproducibility (optional)
    srand((unsigned int)time(NULL));

    // Allocate host memory
    float *h_next_weights = (float *)malloc(output_size * input_size * sizeof(float));
    float *h_output_delta = (float *)malloc(batch_size * output_size * sizeof(float));
    float *h_z = (float *)malloc(batch_size * input_size * sizeof(float));
    float *expected_sigma_prime = (float *)malloc(batch_size * input_size * sizeof(float));
    float *expected_d_input_grad = (float *)malloc(batch_size * input_size * sizeof(float));

    // Initialize weights, output_delta, and z with random values
    for (int i = 0; i < output_size * input_size; i++) {
        h_next_weights[i] = random_float(-1.0f, 1.0f);
    }
    for (int i = 0; i < batch_size * output_size; i++) {
        h_output_delta[i] = random_float(-0.5f, 0.5f);
    }
    for (int i = 0; i < batch_size * input_size; i++) {
        h_z[i] = random_float(-2.0f, 2.0f);
    }

    // Compute sigma_prime on host [batch_size x input_size]
    for (int i = 0; i < batch_size * input_size; i++) {
        float sigmoid = host_sigmoid(h_z[i]);
        expected_sigma_prime[i] = sigmoid * (1.0f - sigmoid);
    }

    // Compute expected d_input_grad on host
    // d_input_grad = W^{l+1}^T * d_output_delta [input_size x batch_size]
    // Then, multiply by sigma_prime(z^l) element-wise

    // Initialize to zero
    memset(expected_d_input_grad, 0, batch_size * input_size * sizeof(float));

    // Perform matrix multiplication and element-wise multiplication
    // For each sample in the batch
    for (int b = 0; b < batch_size; b++) { // For each sample
        for (int j = 0; j < input_size; j++) { // For each input neuron
            for (int k = 0; k < output_size; k++) { // For each output neuron
                // Accumulate the product of weights and output delta
                expected_d_input_grad[b * input_size + j] += h_next_weights[k * input_size + j] * h_output_delta[b * output_size + k];
            }
            // Multiply by sigma_prime(z^l)
            expected_d_input_grad[b * input_size + j] *= expected_sigma_prime[b * input_size + j];
        }
    }

    // Allocate device memory
    float *d_next_weights, *d_output_delta_dev, *d_z, *d_input_grad;
    allocate_device_memory(&d_next_weights, output_size * input_size * sizeof(float));
    allocate_device_memory(&d_output_delta_dev, batch_size * output_size * sizeof(float));
    allocate_device_memory(&d_z, batch_size * input_size * sizeof(float));
    allocate_device_memory(&d_input_grad, batch_size * input_size * sizeof(float));

    // Copy data to device
    copy_to_device(d_next_weights, h_next_weights, output_size * input_size * sizeof(float));
    copy_to_device(d_output_delta_dev, h_output_delta, batch_size * output_size * sizeof(float));
    copy_to_device(d_z, h_z, batch_size * input_size * sizeof(float));

    // Initialize d_input_grad to zero on device
    cudaMemset(d_input_grad, 0, batch_size * input_size * sizeof(float));

    // Create Layer structures
    Layer current_layer = {
        .input_size = input_size,
        .output_size = input_size, // Assuming input_size == current_layer->output_size
        .activation = ACTIVATION_SIGMOID,
        .d_weights = NULL, // Not used in this test
        .d_biases = NULL,  // Not used in this test
        .d_output = NULL,  // Not used in this test
        .d_z = d_z
    };

    Layer next_layer = {
        .input_size = input_size,
        .output_size = output_size,
        .activation = ACTIVATION_SIGMOID, // Activation of next layer
        .d_weights = d_next_weights,
        .d_biases = NULL, // Not used in this test
        .d_output = NULL,  // Not used in this test
        .d_z = NULL        // Not used in this test
    };

    // Call the function under test
    backward_layer(&current_layer, &next_layer, d_output_delta_dev, d_input_grad, batch_size);

    // Check for CUDA errors after kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after backward_layer_large: %s\n", cudaGetErrorString(err));
        TEST_FAIL_MESSAGE("CUDA Error detected.");
    }

    // Copy results back to host
    float *h_input_grad = (float *)malloc(batch_size * input_size * sizeof(float));
    copy_from_device(h_input_grad, d_input_grad, batch_size * input_size * sizeof(float));

    // Verify results
    for (int i = 0; i < batch_size * input_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected_d_input_grad[i], h_input_grad[i]);
    }

    // Cleanup
    free(h_next_weights);
    free(h_output_delta);
    free(h_z);
    free(expected_sigma_prime);
    free(expected_d_input_grad);
    free(h_input_grad);

    cudaFree(d_next_weights);
    cudaFree(d_output_delta_dev);
    cudaFree(d_z);
    cudaFree(d_input_grad);
}

//void setUp(void) {
//    // Initialize CUDA if needed
//    cudaError_t err = cudaSetDevice(0);
//    TEST_ASSERT_EQUAL(cudaSuccess, err);
//}
//
//void tearDown(void) {
//    // Reset device if needed
//    cudaError_t err = cudaDeviceReset();
//    TEST_ASSERT_EQUAL(cudaSuccess, err);
//}
//
//int main(void) {
//    srand((unsigned int)time(NULL));
//    UNITY_BEGIN();
//    RUN_TEST(test_forward_layer);
//    RUN_TEST(test_forward_layer_large);
//    RUN_TEST(test_backward_output_layer);
//    RUN_TEST(test_backward_output_layer_large);
//    RUN_TEST(test_compute_output_delta);
//    RUN_TEST(test_backward_layer);
//    RUN_TEST(test_backward_layer_large);
//    return UNITY_END();
//}

*/