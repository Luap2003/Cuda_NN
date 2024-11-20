// NeuralNetwork.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "layers.h"
#include "activations.h"
#include "lossFunction.h"
#include <cublas_v2.h>

/**
 * @brief Neural Network structure containing layers and training parameters.
 */
typedef struct {
    Layer *layers;               // Array of layers
    int num_layers;              // Total number of layers (including input layer)
    int *layer_sizes;            // Sizes of each layer
    ActivationType *activations; // Activation functions for each layer

    // Training parameters
    int num_epochs;
    int batch_size;
    float learning_rate;
    float initial_learning_rate;
    float decay_rate;

    // cuBLAS handle
    cublasHandle_t handle;
} NeuralNetwork;

/**
 * @brief Initializes the neural network with given architecture and training parameters.
 */
void neural_network_init(NeuralNetwork *nn, int num_layers, int *layer_sizes, ActivationType *activations, int batch_size, int num_epochs, float learning_rate, float decay_rate);

/**
 * @brief Trains the neural network using the provided training data.
 */
void neural_network_train(NeuralNetwork *nn, float *train_images, float *train_labels, int num_train_samples);

/**
 * @brief Evaluates the neural network on the test data.
 */
void neural_network_evaluate(NeuralNetwork *nn, float *test_images, float *test_labels, int num_test_samples);

/**
 * @brief Frees the resources allocated for the neural network.
 */
void free_neural_network(NeuralNetwork *nn);

#endif // NEURAL_NETWORK_H
