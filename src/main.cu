#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string.h>
#include <time.h> 
#include "../include/activations.h"
#include "../include/mnist.h"
#include "../include/neural_net.h"
#include "../include/utilities.h"

// Function to print usage information
void print_usage(char* program_name) {
    printf("Usage: %s [batch_size] [epochs] [layer1_size] [layer2_size] ... [layerN_size]\n", program_name);
    printf("  - Default batch_size: 16\n");
    printf("  - Default epochs: 100\n");
    printf("  - Default layers: [input_size, 256, 128, 10]\n");
    printf("  - First layer is always input_size (fixed based on MNIST data)\n");
    printf("  - Last layer is always 10 (for MNIST digits 0-9)\n");
}

int main(int argc, char *argv[]) {
    // Start timing
    clock_t start_time = clock();

    char log_filename[256];
    char log_filename_weights[256];
    char log_filename_biases[256];

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

    // Default training parameters
    int batch_size = 16;
    int num_epochs = 100;
    float learning_rate = 0.01f;
    float decay_rate = 0.0f;

    // Default network architecture
    int num_layers = 4; // Input layer, 2 hidden layers, output layer
    int *layer_sizes = NULL;
    
    // Parse command line arguments
    if (argc > 1) {
        // Parse batch size
        batch_size = atoi(argv[1]);
        if (batch_size <= 0) {
            printf("Invalid batch size. Using default: 16\n");
            batch_size = 16;
        }
        
        // Parse epochs
        if (argc > 2) {
            num_epochs = atoi(argv[2]);
            if (num_epochs <= 0) {
                printf("Invalid number of epochs. Using default: 100\n");
                num_epochs = 100;
            }
        }
        
        // Parse hidden layer sizes
        if (argc > 3) {
            // Count how many layer sizes were provided
            int provided_layers = argc - 3;
            
            // Add 2 for input and output layers
            num_layers = provided_layers + 2;
            
            // Allocate memory for layer sizes
            layer_sizes = (int*)malloc(num_layers * sizeof(int));
            
            // Input layer is always image_size
            layer_sizes[0] = image_size;
            
            // Parse hidden layers
            for (int i = 0; i < provided_layers; i++) {
                layer_sizes[i+1] = atoi(argv[i+3]);
                if (layer_sizes[i+1] <= 0) {
                    printf("Invalid size for layer %d. Using 128.\n", i+1);
                    layer_sizes[i+1] = 128;
                }
            }
            
            // Last layer is always 10 (number of digits)
            layer_sizes[num_layers-1] = 10;
        }
    }
    
    // If no layer sizes were provided, use defaults
    if (layer_sizes == NULL) {
        layer_sizes = (int*)malloc(4 * sizeof(int));
        layer_sizes[0] = image_size;
        layer_sizes[1] = 256;
        layer_sizes[2] = 128;
        layer_sizes[3] = 10;
    }

    // Set activations (RELU for all except last layer which is SOFTMAX)
    ActivationType *activations = (ActivationType*)malloc((num_layers-1) * sizeof(ActivationType));
    for (int i = 0; i < num_layers-2; i++) {
        activations[i] = ACTIVATION_RELU;
    }
    activations[num_layers-2] = ACTIVATION_SOFTMAX;
    
    // Print network configuration
    printf("Network configuration:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Epochs: %d\n", num_epochs);
    printf("  Layers: ");
    for (int i = 0; i < num_layers; i++) {
        printf("%d ", layer_sizes[i]);
    }
    printf("\n");

    generate_log_filename(log_filename, sizeof(log_filename), batch_size, num_epochs);
    generate_weights_biases_log_filenames(log_filename_weights, sizeof(log_filename_weights), 
                                         log_filename_biases, sizeof(log_filename_biases), 
                                         batch_size, num_epochs);

    // Initialize Neural Network
    NeuralNetwork nn;
    neural_network_init(&nn, num_layers, layer_sizes, activations, batch_size, num_epochs, learning_rate, decay_rate);

    // Train Neural Network
    neural_network_train(&nn, train_images, train_labels, num_train_images);

    // Evaluate Neural Network
    neural_network_evaluate(&nn, test_images, test_labels, num_test_images);


    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Training time: %.2f seconds\n", elapsed_time);
    // Free resources
    free_neural_network(&nn);
    free(layer_sizes);
    free(activations);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    return 0;
}
