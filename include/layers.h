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

void backward_output_layer(Layer *layer, float *Y, float *A_prev_d, cublasHandle_t handle);
void backward_layer(Layer *layer, float *W_next_d, float *dZ_next_d, float *A_prev_d, int n_out_next, cublasHandle_t handle);


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
