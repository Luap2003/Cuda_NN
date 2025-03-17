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

    float huber_delta;  // New: parameter for Huber loss

    float *scale_y_d;
    float *mean_y_d;
    float *scale_X_d;
    float *mean_X_d;
    // cuBLAS handle
    cublasHandle_t handle;
} NeuralNetwork;

/**
 * @brief Initializes the neural network with given architecture and training parameters.
 *
 * @param nn Pointer to @ref NeuralNetwork struct.
 * @param num_layers Number of layers.
 * @param layer_sizes Pointer to array holding layer sizes.
 * @param activations Pointer to array of @ref ActivationType enums.
 * @param batch_size Size of batches.
 * @param num_epochs Number of epochs.
 * @param learning_rate Lerning rate.
 * @param decay_rate Decay rate.
 */
void neural_network_init(NeuralNetwork *nn, int num_layers, int *layer_sizes, ActivationType *activations, int batch_size, int num_epochs, float learning_rate, float decay_rate, float huber_delta,
    float *scale_y, float *mean_y, int scale_y_size,
    float *scale_X, float *mean_X, int scale_X_size) ;
/**
 * @brief Trains the neural network using the provided training data.
 *
 * @param nn Pointer to @ref NeuralNetwork struct.
 * @param train_images Pointer to array of the training images.
 * @param train_labels Pointer to array of training labels
 * @param num_train_samples Number of training samples.
 */
void neural_network_train(NeuralNetwork *nn, float *train_images, float *train_labels, int num_train_samples);

/**
 * @brief Evaluates the neural network on the test data.
 *
 * @param nn Pointer to @ref NeuralNetwork struct.
 * @param test_images Pointer to array of the test images.
 * @param test_labels Pointer to array of test labels
 * @param num_test_samples Number of test samples.
 */
void neural_network_evaluate(NeuralNetwork *nn, float *test_images, float *test_labels, int num_test_samples);

/**
 * @brief Frees the resources allocated for the neural network.
 *
 * @param nn Pointer to @ref NeuralNetwork struct.
 */
void free_neural_network(NeuralNetwork *nn);

#endif // NEURAL_NETWORK_H
