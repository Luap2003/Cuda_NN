/**
 * @file layers.h
 * @brief Declares functions and structures for neural network layers.
 *
 * This file provides the interface for creating, managing, and utilizing
 * dense (fully connected) layers within the neural network framework.
 */

#ifndef LAYERS_H
#define LAYERS_H
#include "activations.h"
#include "lossFunction.h"
#include <cublas_v2.h>
#define THREADS_PER_BLOCK 256
/**
 * @struct Layer
 * @brief Representing one Layer
 *
 * This struct is used to store the parameters of a layer.
 */
typedef struct {
    int m;              // Batch size
    int n_in;           // Number of input neurons
    int n_out;          // Number of output neurons

    // Matrices stored in column-major order
    float *w_d;         // Device weights (n_out x n_in), column-major
    float *b_d;         // Device biases (n_out x 1), column-major
    float *A_d;         // Device activations (n_in x m), column-major
    float *Z_d;         // Device pre-activations (n_out x m), column-major
    float *dZ_d;        // Device gradient w.r.t Z (n_out x m), column-major
    float *dW_d;        // Device gradient w.r.t weights (n_out x n_in), column-major
    float *db_d;        // Device gradient w.r.t biases (n_out x 1), column-major

    ActivationType aktfunc;
} Layer;
/**
 * @brief Initializes layer.
 *
 * Allocates memory on devices for @ref Layer struct and initializes the weights
 * with Glorot Uniform. The bias is initialized to zero.

 * @param layer Pointer to a @ref Layer struct to initialize. 
 * @param m Batch size.
 * @param n_in Number of input neurons.
 * @param n_out Number of output neurons.
 * @param aktfunc Aktivation Function from @ref ActivationType enum.
 */
void layer_init(Layer *layer, int m, int n_in, int n_out, ActivationType aktfunc);

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
void layer_forward(Layer *layer, float *A_prev_d, cublasHandle_t handle);

/**
 * @brief Performs backword propagation on output layer.
 *
 * Calculates the gradient of the pre-activation values and the gradient of the weights and biases. 
 *
 * @param layer Pointer to @ref Layer struct.
 * @param Y Pointer to true labels.
 * @param A_prev_d Activations from the previous layer.
 * @param handle CublasHandle.
 */
void backward_output_layer(Layer *layer, float *Y, float *A_prev_d, cublasHandle_t handle);

/**
 * @brief Performs backword propagation on hidden layers.
 *
 * Propagates gradients from previous layer and applies activation function derivative.
 * Followed by calculating the derivative of the weights and biases.
 *
 * @param layer Pointer to @ref Layer struct.
 * @param W_next_d Derivativions of the weights of the next layer.
 * @param dZ_next_d Derivativions of the pre-activation values of the next layer.
 * @param A_prev_d Derivations of the activations from the previous layer.
 * @param n_out_next Number of neurons in the next layer.
 * @param handle CublasHandle.
 */
void backward_layer(Layer *layer, float *W_next_d, float *dZ_next_d, float *A_prev_d, int n_out_next, cublasHandle_t handle);

/**
 * @brief Update weights and biases of the layers.
 *
 * @param layer Pointer to @ref Layer struct.
 * @param learning_rate Learning rate.
 */
void update(Layer *layer, float learning_rate);

/**
 * @brief Frees the memory allocated for a layer.
 *
 * Releases both host and device memory associated with the layer.
 *
 * @param layer Pointer to the Layer to be freed.
 */
void free_layer(Layer *layer);


#endif // LAYERS_H
