/**
 * @file layers.h
 * @brief Declares functions and structures for neural network layers.
 *
 * This file provides the interface for creating, managing, and utilizing
 * dense (fully connected) layers within the neural network framework.
 */

#ifndef LAYERS_H
#define LAYERS_H
#define THREADS_PER_BLOCK 256
typedef struct {
    char *type;           // e.g. "dense"
    int input_size;
    int output_size;
    const char *activation;     // e.g. "relu"
    float *weights;       // Host weights
    float *biases;        // Host biases
    float *d_weights;     // Device weights
    float *d_biases;      // Device biases
} Layer;

/**
 * @brief Creates and initializes a dense (fully connected) layer.
 *
 * Allocates memory for the layer structure, initializes weights using Xavier initialization,
 * sets biases to zero, and transfers weights and biases to the GPU.
 *
 * @param input_size  Number of input neurons.
 * @param output_size Number of output neurons.
 * @param activation  Activation function name (e.g., "relu", "sigmoid").
 * @return Pointer to the newly created Layer, or NULL on failure.
 */
Layer* create_dense_layer(int input_size, int output_size, const char *activation);

/**
 * @brief Performs forward propagation through the given layer.
 *
 * Executes matrix multiplication between input and weights, adds biases, and applies the activation function.
 *
 * @param layer       Pointer to the Layer.
 * @param d_input     Device pointer to the input matrix.
 * @param d_output    Device pointer to store the output matrix.
 * @param batch_size  Number of samples in the batch.
 */
void forward_layer(Layer *layer, float *d_input, float *d_output, int batch_size);


void backward_layer(Layer *layer, float *d_input, float *d_output_grad, float *d_input_grad, int batch_size);


void update_layer(Layer *layer, float learning_rate);

/**
 * @brief Frees the memory allocated for a layer.
 *
 * Releases both host and device memory associated with the layer.
 *
 * @param layer Pointer to the Layer to be freed.
 */
void free_layer(Layer *layer);

/**
 * @brief Prints the properties and a snippet of the weights and biases of a layer.
 *
 * Useful for debugging and verifying layer configurations.
 *
 * @param layer Pointer to the Layer to be printed.
 */
void print_layer(Layer *layer);

#endif // LAYERS_H
