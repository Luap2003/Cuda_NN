#include "../tests/unity/unity.h"
#include "../include/layers.h"
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

void test_forward_layer(void) {
    // Initialize test data
    int batch_size = 2;
    int input_size = 3;
    int output_size = 2;
    
    float h_input[] = {
        0.1f, 0.4f, 0.5f,
        0.2f, 0.3f, 0.7f
    };
    float h_weights[] = {
        0.1f, 0.2f, 0.9f,
        0.6f, 0.11f, 0.12f
    };
    float h_biases[] = {
        0.07f,
        0.03f
    };
    float h_output[4] = {0}; 

    float expected_output[] = {
        0.61f, 0.194f,
        0.78, 0.267f
    };
    
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
    
    // Create and initialize layer
    Layer layer = {
        .input_size = input_size,
        .output_size = output_size,
        .activation = "linear",
        .d_weights = d_weights,
        .d_biases = d_biases,
          // Test without activation first
    };
    
    // Call the function under test
    forward_layer(&layer, d_input, d_output, batch_size);
    
    // Copy results back to host
    copy_from_device(h_output, d_output, batch_size * output_size * sizeof(float));
    
    // Verify results
    for (int i = 0; i < batch_size * output_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_output[i], h_output[i]);
    }
    
    // Test with sigmoid activation
    layer.activation = "sigmoid";
    forward_layer(&layer, d_input, d_output, batch_size);
    copy_from_device(h_output, d_output, batch_size * output_size * sizeof(float));
    
    float expected_sigmoid_output[4];
    for (int i = 0; i < batch_size * output_size; i++) {
        expected_sigmoid_output[i] = host_sigmoid(expected_output[i]);
        printf("Sigmoid of %.2f = %.10f\n", expected_output[i], expected_sigmoid_output[i]);
    }
    
    for (int i = 0; i < batch_size * output_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected_sigmoid_output[i], h_output[i]);
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
}

void setUp(void) {
    // Initialize CUDA if needed
    cudaError_t err = cudaSetDevice(0);
    TEST_ASSERT_EQUAL(cudaSuccess, err);
}

void tearDown(void) {
    // Reset device if needed
    cudaError_t err = cudaDeviceReset();
    TEST_ASSERT_EQUAL(cudaSuccess, err);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_forward_layer);
    return UNITY_END();
}