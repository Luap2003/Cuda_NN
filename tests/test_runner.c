#include "unity/unity.h"
#include "layers/test_layers.c"
#include "neural_net/test_neural_net.c"
#include "unity/unity_internals.h"
// Setup and Teardown
void setUp(void) {
    // Initialize CUDA device before each test
    cudaError_t err = cudaSetDevice(0);
    TEST_ASSERT_EQUAL_MESSAGE(cudaSuccess, err, "cudaSetDevice failed");
}

void tearDown(void) {
    // Reset CUDA device after each test
    cudaError_t err = cudaDeviceReset();
    TEST_ASSERT_EQUAL_MESSAGE(cudaSuccess, err, "cudaDeviceReset failed");
}
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
int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_forward_layer);
    RUN_TEST(test_forward_layer_large);
    RUN_TEST(test_backward_output_layer);
    RUN_TEST(test_backward_output_layer_large);
    RUN_TEST(test_compute_output_delta);
    RUN_TEST(test_backward_layer);
    RUN_TEST(test_backward_layer_large);
    RUN_TEST(test_update_parameters);
    //RUN_TEST(test_backpropagation);
    UNITY_END();

    return 0;
}