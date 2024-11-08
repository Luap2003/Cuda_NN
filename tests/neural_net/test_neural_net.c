// neural_net_test.cu
#include "../../include/neural_net.h"
#include "../../include/layers.h"
#include <cuda_runtime.h>
#include "../unity/unity.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Assuming these helper functions are defined elsewhere
extern void allocate_device_memory(float **d_ptr, size_t size);
extern void copy_to_device(float *d_ptr, float *h_ptr, size_t size);
extern void copy_from_device(float *h_ptr, float *d_ptr, size_t size);

// Function to initialize a layer
void initialize_layer(Layer *layer, int input_size, int output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = ACTIVATION_LINEAR;

    // Allocate device memory for weights, biases, and their gradients
    size_t weights_size = input_size * output_size * sizeof(float);
    size_t biases_size = output_size * sizeof(float);

    allocate_device_memory(&layer->d_weights, weights_size);
    allocate_device_memory(&layer->d_biases, biases_size);
    allocate_device_memory(&layer->d_weights_grad, weights_size);
    allocate_device_memory(&layer->d_biases_grad, biases_size);
}

// Test function for update_parameters
void test_update_parameters(void) {
    // Initialize test parameters
    int input_size = 4;
    int output_size = 3;
    float learning_rate = 0.1f;

    // Allocate and initialize host memory
    float h_weights[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f};
    float h_biases[3] = {0.5f, -0.5f, 1.0f};
    float h_weights_grad[12] = {0.1f, 0.1f, 0.1f, 0.1f,
                                0.2f, 0.2f, 0.2f, 0.2f,
                                0.3f, 0.3f, 0.3f, 0.3f};
    float h_biases_grad[3] = {0.05f, 0.05f, 0.05f};

    // Expected updated weights and biases
    float expected_weights[12];
    float expected_biases[3];
    for (int i = 0; i < 12; i++) {
        expected_weights[i] = h_weights[i] - learning_rate * h_weights_grad[i];
    }
    for (int i = 0; i < 3; i++) {
        expected_biases[i] = h_biases[i] - learning_rate * h_biases_grad[i];
    }

    // Initialize layer
    Layer layer;
    initialize_layer(&layer, input_size, output_size);

    // Copy initial weights, biases, and gradients to device
    copy_to_device(layer.d_weights, h_weights, 12 * sizeof(float));
    copy_to_device(layer.d_biases, h_biases, 3 * sizeof(float));
    copy_to_device(layer.d_weights_grad, h_weights_grad, 12 * sizeof(float));
    copy_to_device(layer.d_biases_grad, h_biases_grad, 3 * sizeof(float));

    // Call the function under test
    update_parameters(&layer, learning_rate);

    // Allocate host memory to retrieve updated weights and biases
    float h_updated_weights[12];
    float h_updated_biases[3];
    float h_updated_weights_grad[12];
    float h_updated_biases_grad[3];

    // Copy updated weights and biases back to host
    copy_from_device(h_updated_weights, layer.d_weights, 12 * sizeof(float));
    copy_from_device(h_updated_biases, layer.d_biases, 3 * sizeof(float));
    copy_from_device(h_updated_weights_grad, layer.d_weights_grad, 12 * sizeof(float));
    copy_from_device(h_updated_biases_grad, layer.d_biases_grad, 3 * sizeof(float));

    // Verify weights
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(expected_weights[i], h_updated_weights[i], 1e-5f);
    }

    // Verify biases
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(expected_biases[i], h_updated_biases[i], 1e-5f);
    }

    // Verify that gradients have been reset to zero
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(h_updated_weights_grad[i],0, 1e-6f);
    }
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(h_updated_biases_grad[i],0, 1e-6f);
    }

    // Cleanup device memory
    cudaFree(layer.d_weights);
    cudaFree(layer.d_biases);
    cudaFree(layer.d_weights_grad);
    cudaFree(layer.d_biases_grad);
}

//void test_backpropagation(void) {
//    // Network architecture
//    int input_size = 2;
//    int hidden_size = 2;
//    int output_size = 1;
//    int batch_size = 1;
//    float learning_rate = 0.5f; // Using a larger learning rate for the example
//
//    // Create layers with known weights and biases
//    Layer* hidden_layer = create_dense_layer(input_size, hidden_size, ACTIVATION_SIGMOID);
//    Layer* output_layer = create_dense_layer(hidden_size, output_size, ACTIVATION_SIGMOID);
//
//    // Initialize weights and biases to known values
//    float h_hidden_weights[4] = {0.15f, 0.2f, 0.25f, 0.3f}; // 2x2 weights
//    float h_hidden_biases[2] = {0.35f, 0.35f};
//
//    float h_output_weights[2] = {0.4f, 0.45f}; // 2x1 weights
//    float h_output_biases[1] = {0.6f};
//
//    // Copy initial weights and biases to device
//    copy_to_device(hidden_layer->d_weights, h_hidden_weights, 4 * sizeof(float));
//    copy_to_device(hidden_layer->d_biases, h_hidden_biases, 2 * sizeof(float));
//    copy_to_device(output_layer->d_weights, h_output_weights, 2 * sizeof(float));
//    copy_to_device(output_layer->d_biases, h_output_biases, sizeof(float));
//
//    // Create the network
//    NeuralNetwork network;
//    network.num_layers = 2;
//    network.layers = (Layer**)malloc(2 * sizeof(Layer*));
//    network.layers[0] = hidden_layer;
//    network.layers[1] = output_layer;
//
//    // Input and expected output
//    float h_input[2] = {0.05f, 0.1f};
//    float h_labels[1] = {0.01f};
//
//    // Copy inputs and labels to device
//    float* d_input;
//    float* d_labels;
//    cudaMalloc((void**)&d_input, batch_size * input_size * sizeof(float));
//    cudaMalloc((void**)&d_labels, batch_size * output_size * sizeof(float));
//
//    copy_to_device(d_input, h_input, batch_size * input_size * sizeof(float));
//    copy_to_device(d_labels, h_labels, batch_size * output_size * sizeof(float));
//
//    // Run backpropagation
//    backpropagation(&network, d_input, d_labels, batch_size, learning_rate);
//
//    // Expected updated weights and biases after one backpropagation step
//    float expected_hidden_weights[4] = {0.14978072f, 0.19956143f, 0.24975114f, 0.2995023f};
//    float expected_hidden_biases[2] = {0.3456143f, 0.3456143f};
//
//    float expected_output_weights[2] = {0.35891648f, 0.408666f};
//    float expected_output_biases[1] = {0.5307503f};
//
//    // Allocate host memory for updated parameters
//    float h_updated_hidden_weights[4];
//    float h_updated_hidden_biases[2];
//    float h_updated_output_weights[2];
//    float h_updated_output_biases[1];
//
//    // Copy updated parameters from device to host
//    copy_from_device(h_updated_hidden_weights, hidden_layer->d_weights, 4 * sizeof(float));
//    copy_from_device(h_updated_hidden_biases, hidden_layer->d_biases, 2 * sizeof(float));
//    copy_from_device(h_updated_output_weights, output_layer->d_weights, 2 * sizeof(float));
//    copy_from_device(h_updated_output_biases, output_layer->d_biases, sizeof(float));
//
//    // Set a tolerance for floating-point comparison
//    float tolerance = 1e-5f;
//
//    // Verify hidden layer weights
//    for (int i = 0; i < 4; i++) {
//        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected_hidden_weights[i], h_updated_hidden_weights[i]);
//    }
//
//    // Verify hidden layer biases
//    for (int i = 0; i < 2; i++) {
//        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected_hidden_biases[i], h_updated_hidden_biases[i]);
//    }
//
//    // Verify output layer weights
//    for (int i = 0; i < 2; i++) {
//        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected_output_weights[i], h_updated_output_weights[i]);
//    }
//
//    // Verify output layer biases
//    TEST_ASSERT_FLOAT_WITHIN(tolerance, expected_output_biases[0], h_updated_output_biases[0]);
//
//    // Cleanup device memory
//    cudaFree(d_input);
//    cudaFree(d_labels);
//    cudaFree(hidden_layer->d_weights);
//    cudaFree(hidden_layer->d_biases);
//    cudaFree(hidden_layer->d_weights_grad);
//    cudaFree(hidden_layer->d_biases_grad);
//    cudaFree(hidden_layer->d_output);
//    cudaFree(hidden_layer->d_z);
//    cudaFree(output_layer->d_weights);
//    cudaFree(output_layer->d_biases);
//    cudaFree(output_layer->d_weights_grad);
//    cudaFree(output_layer->d_biases_grad);
//    cudaFree(output_layer->d_output);
//    cudaFree(output_layer->d_z);
//
//    free(network.layers);
//}
//