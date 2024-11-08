#include "../include/neural_net.h"
#include "../include/utils.h"
#include "../include/data_loader.h"
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

// Include headers for data loading
// #include "data_loader.h"

int main() {
    srand(42);

    // Load data
    float *x_train_host, *y_train_host;
    int num_samples = 60000; // Number of samples in MNIST training set
    int input_size = 784;    // 28x28 images flattened
    int output_size = 10;    // 10 classes for digits 0-9

        // Load MNIST images and labels
    x_train_host = load_mnist_images("../data/mnist_data/train-images.idx3-ubyte", &num_samples, &input_size);
    y_train_host = load_mnist_labels("../data/mnist_data/train-labels.idx1-ubyte", &num_samples, &output_size);

    // For simplicity, allocate random data (replace this with actual data loading)
    x_train_host = (float*)malloc(num_samples * input_size * sizeof(float));
    y_train_host = (float*)malloc(num_samples * output_size * sizeof(float));
    // Initialize x_train_host and y_train_host with actual data

    // Set training parameters
    int epochs = 10;
    int batch_size = 32;
    float learning_rate = 0.01f;

    int num_batches = num_samples / batch_size;

    // Allocate device memory for data
    float *d_x_train, *d_y_train;
    cudaMalloc((void**)&d_x_train, num_samples * input_size * sizeof(float));
    cudaMalloc((void**)&d_y_train, num_samples * output_size * sizeof(float));

    cudaMemcpy(d_x_train, x_train_host, num_samples * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_train, y_train_host, num_samples * output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Create neural network
    int num_layers = 2;
    NeuralNetwork *network = create_neural_net(num_layers);

    // Create layers
    Layer *layer1 = create_dense_layer(input_size, 128, ACTIVATION_SIGMOID);
    Layer *layer2 = create_dense_layer(128, output_size, ACTIVATION_SIGMOID);

    // Add layers to the network
    add_layer_to_neural_net(network, layer1, 0);
    add_layer_to_neural_net(network, layer2, 1);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        printf("Epoch %d/%d\n", epoch + 1, epochs);

        // Optionally shuffle the data here

        for (int batch = 0; batch < num_batches; ++batch) {
            // Get batch data
            int batch_start = batch * batch_size;
            float *d_batch_input = d_x_train + batch_start * input_size;
            float *d_batch_labels = d_y_train + batch_start * output_size;

            // Forward pass
            float *current_input = d_batch_input;
            for (int i = 0; i < network->num_layers; ++i) {
                Layer *layer = network->layers[i];
                forward_layer(layer, current_input, layer->d_output, batch_size);
                current_input = layer->d_output;
            }
            printf("Forward pass done\n");
            // Compute loss (implement loss computation if desired)

            // Backward pass
            backpropagation(network, d_batch_input, d_batch_labels, batch_size, learning_rate);

            // Optionally, compute and print training loss and accuracy
        }
    }

    // Free resources
    free_neural_net(network);
    cudaFree(d_x_train);
    cudaFree(d_y_train);
    free(x_train_host);
    free(y_train_host);

    return 0;
}
