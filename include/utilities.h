#ifndef UTILITIES_H
#define UTILITIES_H

#include <cuda_runtime.h>
#include "neural_net.h"

// Macro for CUDA error checking
#define cudaCheckError() {                                           \
    cudaError_t e = cudaGetLastError();                              \
    if(e != cudaSuccess) {                                           \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,         \
               cudaGetErrorString(e));                               \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

// Function declarations
void display_progress_bar(int current, int total, int bar_width);
void display_epoch_progress(int current_epoch, int total_epochs, float loss, float accuracy, float epoch_time, float batches_per_second);
void generate_log_filename(char *log_filename, size_t size, int batch_size, int num_epochs);
void generate_weights_biases_log_filenames(char *weights_log, size_t w_size, char *biases_log, size_t b_size, int batch_size, int num_epochs);
void log_epoch_progress(int current_epoch, int total_epochs, float loss, float accuracy, float epoch_time, float batches_per_second);
void log_weights(NeuralNetwork *nn, int epoch);
void log_biases(NeuralNetwork *nn, int epoch);
// External variables for log filenames
extern char log_filename[256];
extern char log_filename_weights[256];
extern char log_filename_biases[256];

#endif // UTILITIES_H
