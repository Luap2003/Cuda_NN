// main.cu
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

// File paths for data
#define TRAIN_IMAGES_PATH "/home/user/Documents/arbeit/model-training/src/train_images.bin"
#define TRAIN_LABELS_PATH "/home/user/Documents/arbeit/model-training/src/train_labels.bin"
#define TEST_IMAGES_PATH "/home/user/Documents/arbeit/model-training/src/test_images.bin"
#define TEST_LABELS_PATH "/home/user/Documents/arbeit/model-training/src/test_labels.bin"

// Shape files
#define TRAIN_IMAGES_SHAPE_PATH "/home/user/Documents/arbeit/model-training/src/train_images_shape.txt"
#define TRAIN_LABELS_SHAPE_PATH "/home/user/Documents/arbeit/model-training/src/train_labels_shape.txt"
#define TEST_IMAGES_SHAPE_PATH "/home/user/Documents/arbeit/model-training/src/test_images_shape.txt"
#define TEST_LABELS_SHAPE_PATH "/home/user/Documents/arbeit/model-training/src/test_labels_shape.txt"

// Scaler parameter files
#define SCALE_X_PATH "/home/user/Documents/arbeit/model-training/src/scale_X.txt"
#define MEAN_X_PATH "/home/user/Documents/arbeit/model-training/src/mean_X.txt"
#define SCALE_Y_PATH "/home/user/Documents/arbeit/model-training/src/scale_y.txt"
#define MEAN_Y_PATH "/home/user/Documents/arbeit/model-training/src/mean_y.txt"

int read_shape(const char* shape_file, int* rows, int* cols) {
    FILE* file = fopen(shape_file, "r");
    if (!file) {
        perror("Failed to open shape file");
        return -1;
    }
    if (fscanf(file, "%d %d", rows, cols) != 2) {
        fprintf(stderr, "Error reading shape from %s\n", shape_file);
        fclose(file);
        return -1;
    }
    fclose(file);
    return 0;
}

// Read binary data from file
float* read_binary_data(const char* bin_file, int total_elements) {
    FILE* file = fopen(bin_file, "rb");
    if (!file) {
        perror("Failed to open binary file");
        return NULL;
    }
    float* data = (float*) malloc(total_elements * sizeof(float));
    if (!data) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(file);
        return NULL;
    }
    size_t read_count = fread(data, sizeof(float), total_elements, file);
    if (read_count != (size_t) total_elements) {
        fprintf(stderr, "Error reading binary file: expected %d elements, got %zu\n", 
                total_elements, read_count);
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return data;
}

// New helper function to read scaler parameters from a text file.
// It reads all floats in the file and returns them in a dynamically allocated array.
int read_scaler_parameters(const char* file_path, float** params, int* count) {
    FILE* file = fopen(file_path, "r");
    if (!file) {
        perror("Failed to open scaler file");
        return -1;
    }
    float temp;
    int cnt = 0;
    // Count number of floats in the file
    while (fscanf(file, "%f", &temp) == 1) {
        cnt++;
    }
    rewind(file);
    *params = (float*) malloc(cnt * sizeof(float));
    if (!*params) {
        fprintf(stderr, "Memory allocation error for scaler parameters\n");
        fclose(file);
        return -1;
    }
    for (int i = 0; i < cnt; i++) {
        if (fscanf(file, "%f", &((*params)[i])) != 1) {
            fprintf(stderr, "Error reading scaler parameter at index %d in %s\n", i, file_path);
            free(*params);
            fclose(file);
            return -1;
        }
    }
    *count = cnt;
    fclose(file);
    return 0;
}

int main(int argc, char *argv[]) {
    #if defined(DEBUG) || defined(DEBUG_CONFIG)
    printf("Running in DEBUG mode!\n\n");
    #endif

    Config config = {NULL, -1, -1, -1, NULL};

    if (argc > 1) {
        int config_read = 0;
        // Loop through arguments to find "-c"
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                printf("\n");
                printf("-c \t\t Provide path to config.\n");
                printf("-h/--help \t Show this message.\n");
                printf("\n");
            }
            if (strcmp(argv[i], "-c") == 0) {
                if (i + 1 < argc) { // Check if a file name follows "-c"
                    char *config_file = argv[i + 1];
                    int parser_check = parser(config_file, &config);
                    config_read = 1;
                    if (parser_check != 0) {
                        printf("Error parsing config file!");
                        return EXIT_FAILURE;
                    }
                } else {
                    printf("Error: Missing argument after -c\n");
                    return EXIT_FAILURE; // Exit with error
                }
            }
        }
        if (!config_read) {
            int parser_check = parser(CONFIG_FILE, &config);
            if (parser_check != 0) {
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

    int train_images_rows, train_images_cols;
    int train_labels_rows, train_labels_cols;
    int test_images_rows, test_images_cols;
    int test_labels_rows, test_labels_cols;

    if (read_shape(TRAIN_IMAGES_SHAPE_PATH, &train_images_rows, &train_images_cols) != 0 ||
        read_shape(TRAIN_LABELS_SHAPE_PATH, &train_labels_rows, &train_labels_cols) != 0 ||
        read_shape(TEST_IMAGES_SHAPE_PATH, &test_images_rows, &test_images_cols) != 0 ||
        read_shape(TEST_LABELS_SHAPE_PATH, &test_labels_rows, &test_labels_cols) != 0) {
        printf("Error reading shape files\n");
        return EXIT_FAILURE;
    }

    // Assuming the image shape file is in the format: [number_of_images image_dimension]
    num_train_images = train_images_rows;
    image_size = train_images_cols;
    int num_test_images = test_images_rows;

    // Load training data
    train_images = read_binary_data(TRAIN_IMAGES_PATH, train_images_rows * train_images_cols);
    float* train_labels = read_binary_data(TRAIN_LABELS_PATH, train_labels_rows * train_labels_cols);
    
    // Load test data
    float* test_images = read_binary_data(TEST_IMAGES_PATH, test_images_rows * test_images_cols);
    float* test_labels = read_binary_data(TEST_LABELS_PATH, test_labels_rows * test_labels_cols);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        printf("Error reading binary data files\n");
        // Cleanup allocated memory
        if (train_images) free(train_images);
        if (train_labels) free(train_labels);
        if (test_images) free(test_images);
        if (test_labels) free(test_labels);
        return EXIT_FAILURE;
    }

    // Read scaler parameters
    float *scale_X, *mean_X, *scale_y, *mean_y;
    int scaleX_count, meanX_count, scaleY_count, meanY_count;
    if (read_scaler_parameters(SCALE_X_PATH, &scale_X, &scaleX_count) != 0 ||
        read_scaler_parameters(MEAN_X_PATH, &mean_X, &meanX_count) != 0 ||
        read_scaler_parameters(SCALE_Y_PATH, &scale_y, &scaleY_count) != 0 ||
        read_scaler_parameters(MEAN_Y_PATH, &mean_y, &meanY_count) != 0) {
        printf("Error reading scaler parameter files\n");
        // Free allocated data
        free(train_images);
        free(train_labels);
        free(test_images);
        free(test_labels);
        return EXIT_FAILURE;
    }

    // Define network architecture
    int num_layers = config.number_layers;
    if (config.input_layer == -1) {
        config.layers_sizes[0] = image_size;
    }
    int *layer_sizes = config.layers_sizes;
    ActivationType *activations = config.activation_functions;

    // Training parameters
    int batch_size = config.batch_size;
    int num_epochs = config.num_epochs;
    float learning_rate = config.learning_rate;
    float decay_rate = config.decay_rate;

    #ifdef LOG
    generate_log_filename(log_filename, sizeof(log_filename), batch_size, num_epochs);
    generate_weights_biases_log_filenames(log_filename_weights, sizeof(log_filename_weights),
                                            log_filename_biases, sizeof(log_filename_biases),
                                            batch_size, num_epochs);
    #endif

    // Initialize Neural Network
    NeuralNetwork nn;
    neural_network_init(&nn, num_layers, layer_sizes, activations, 
        batch_size, num_epochs, learning_rate, decay_rate,
        1.0f, // Huber delta parameter, adjust as needed
        scale_y, mean_y, scaleY_count,
        scale_X, mean_X, scaleX_count);
    printf("Neural Network initialized\n");

    // Train Neural Network
    neural_network_train(&nn, train_images, train_labels, num_train_images);

    // Evaluate Neural Network
    // neural_network_evaluate(&nn, test_images, test_labels, num_test_images);

    // Example usage of scaler parameters:
    // For your loss calculation, you might need to perform inverse scaling:
    // predicted_inverse = predicted * scale_y[0] + mean_y[0];
    // results_inverse   = results   * scale_X[0] + mean_X[0];
    // (Adjust indexing as necessary for your data dimensions)

    // Free resources
    free_neural_network(&nn);

    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    // Free scaler parameters
    free(scale_X);
    free(mean_X);
    free(scale_y);
    free(mean_y);
    #endif

    free(config.activation_functions);
    free(config.hidden_layers);

    return 0;
}
