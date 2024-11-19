// neural_net.cu
#include "../include/layers.h"
#include "../include/neural_net.h"

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <stdio.h>
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

// Kernel to update weights: W = W - learning_rate * dW
__global__ void update_weights_kernel(float *d_weights, float *d_weights_grad,
                                      float learning_rate, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    d_weights[idx] -= learning_rate * d_weights_grad[idx];
  }
}

// Kernel to update biases: b = b - learning_rate * db
__global__ void update_biases_kernel(float *d_biases, float *d_biases_grad,
                                     float learning_rate, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    d_biases[idx] -= learning_rate * d_biases_grad[idx];
  }
}

// Function to update parameters of a single layer
void update_parameters(Layer *layer, float learning_rate) {
  int weights_size = layer->input_size * layer->output_size;
  int biases_size = layer->output_size;

  int threads = THREADS_PER_BLOCK;
  int blocks_weights = (weights_size + threads - 1) / threads;
  int blocks_biases = (biases_size + threads - 1) / threads;

  // Update weights
  update_weights_kernel<<<blocks_weights, threads>>>(
      layer->d_weights, layer->d_weights_grad, learning_rate, weights_size);
  // Update biases
  update_biases_kernel<<<blocks_biases, threads>>>(
      layer->d_biases, layer->d_biases_grad, learning_rate, biases_size);

  // Reset gradients to zero for next iteration
  cudaMemset(layer->d_weights_grad, 0, weights_size * sizeof(float));
  cudaMemset(layer->d_biases_grad, 0, biases_size * sizeof(float));

  cudaDeviceSynchronize();
}

NeuralNetwork *create_neural_net(int num_layers) {
  NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
  network->num_layers = num_layers;
  network->layers = (Layer **)malloc(num_layers * sizeof(Layer *));
  return network;
}

void add_layer_to_neural_net(NeuralNetwork *network, Layer *layer, int index) {
  if (index >= 0 && index < network->num_layers) {
    network->layers[index] = layer;
  }
}

void free_neural_net(NeuralNetwork *network) {
  for (int i = 0; i < network->num_layers; ++i) {
    free_layer(network->layers[i]);
  }
  free(network->layers);
  free(network);
}
