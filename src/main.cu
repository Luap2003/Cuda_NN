// main.c
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <cublas_v2.h>
#include "../include/activations.h"
#include "../include/mnist.h"
#include "../include/neural_net.h"
#include "../include/utilities.h"
#include "../include/config.h"

#define CONFIG_FILE "config.txt"



int main(int argc, char *argv[]) {
    #if defined(DEBUG) || defined(DEBUG_CONFIG)  
    printf("Running in DEBUG mode!\n\n");
    #endif

    Config config = {NULL, -1 , -1, -1, NULL};

    if (argc > 1){
        int config_read = 0;
        // Loop through arguments to find "-c"
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-c") == 0) {
                if (i + 1 < argc) { // Check if a file name follows "-c"
                    char *config_file = argv[i + 1];
                    int parser_check = parser(config_file, &config);
                    config_read = 1;
                    if(parser_check != 0){
                        printf("Error parsing config file!");
                        return EXIT_FAILURE;
                    }
                } else {
                    printf("Error: Missing argument after -c\n");
                    return EXIT_FAILURE; // Exit with error
                }
            }
            
        }
        if(!config_read){
            int parser_check = parser(CONFIG_FILE, &config);
            if(parser_check != 0){
                printf("Error parsing config file!");
                return EXIT_FAILURE;
            }
        }
    }


    #ifndef DEBUG_CONFIG
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
    int num_layers = config.number_layers;
    if(config.input_layer == -1 ){
        config.layers_sizes[0] = image_size;
    }
    int * layer_sizes = config.layers_sizes;
    ActivationType * activations = config.activation_functions;

    // Training parameters
    int batch_size = config.batch_size;
    int num_epochs = config.num_epochs;
    float learning_rate = config.learning_rate;
    float decay_rate = config.decay_rate;

    #ifdef LOG
    generate_log_filename(log_filename, sizeof(log_filename), batch_size, num_epochs);
    generate_weights_biases_log_filenames(log_filename_weights, sizeof(log_filename_weights),log_filename_biases, sizeof(log_filename_biases), batch_size, num_epochs);
    #endif

    // Initialize Neural Network
    NeuralNetwork nn;
    neural_network_init(&nn, num_layers, layer_sizes, activations, batch_size, num_epochs, learning_rate, decay_rate);

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
    #endif

    free(config.activation_functions);
    free(config.hidden_layers);

    return 0;
}
