#include "unity/unity.h"
#include "layers/test_layers.c"
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

int main(void) {
    UNITY_BEGIN();
        RUN_TEST(test_forward_layer);
    RUN_TEST(test_forward_layer_large);
    RUN_TEST(test_backward_output_layer);
    RUN_TEST(test_backward_output_layer_large);
    RUN_TEST(test_compute_output_delta);
    RUN_TEST(test_backward_layer);
    RUN_TEST(test_backward_layer_large);
    UNITY_END();

    return 0;
}