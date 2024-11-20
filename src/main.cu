// main.c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/layers.h"
#include "../include/activations.h"
#include "../include/mnist.h"
#include "../include/neural_net.h"
#include "../include/utilities.h"



int main() {
    // Load MNIST training data
    float *train_images;
    int num_train_images;
    int image_size;

    if (read_mnist_images("data//train-images.idx3-ubyte", &train_images, &num_train_images, &image_size) != 0) {
        printf("Error reading MNIST images\n");
        return EXIT_FAILURE;
    }

    float *train_labels;
    int num_train_labels;

    if (read_mnist_labels("data/train-labels.idx1-ubyte", &train_labels, &num_train_labels) != 0) {
        printf("Error reading MNIST labels\n");
        return EXIT_FAILURE;
    }

    // Load MNIST test data
    float *test_images;
    int num_test_images;
    int test_image_size;

    if (read_mnist_images("data/t10k-images.idx3-ubyte", &test_images, &num_test_images, &test_image_size) != 0) {
        printf("Error reading MNIST test images\n");
        return EXIT_FAILURE;
    }

    float *test_labels;
    int num_test_labels;

    if (read_mnist_labels("data/t10k-labels.idx1-ubyte", &test_labels, &num_test_labels) != 0) {
        printf("Error reading MNIST test labels\n");
        return EXIT_FAILURE;
    }

    // Check that number of images and labels match
    if (num_train_images != num_train_labels || num_test_images != num_test_labels) {
        printf("Number of images and labels do not match\n");
        return EXIT_FAILURE;
    }

    // Define network architecture
    int num_layers = 3; // Input layer, hidden layer, output layer
    int layer_sizes[] = { image_size, 128, 10 };
    ActivationType activations[] = {ACTIVATION_RELU, ACTIVATION_SIGMOID }; // ACTIVATION_NONE for input layer

    // Training parameters
    int batch_size = 64;
    int num_epochs = 20;
    float learning_rate = 0.01f;

    // Initialize Neural Network
    NeuralNetwork nn;
    neural_network_init(&nn, num_layers, layer_sizes, activations, batch_size, num_epochs, learning_rate);

    // Train Neural Network
    neural_network_train(&nn, train_images, train_labels, num_train_images);

    // Evaluate Neural Network
    neural_network_evaluate(&nn, test_images, test_labels, num_test_images);

    // Free resources
    free_neural_network(&nn);

    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    return 0;
}