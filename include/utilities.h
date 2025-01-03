#ifndef UTILITIES_H
#define UTILITIES_H

#include <cuda_runtime.h>

#define cudaCheckError() {                                           \
    cudaError_t e = cudaGetLastError();                              \
    if(e != cudaSuccess) {                                           \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,         \
               cudaGetErrorString(e));                               \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

void display_progress_bar(int current, int total, int bar_width);
void display_epoch_progress(int current_epoch, int total_epochs, float loss, float accuracy, float epoch_time, float batches_per_second);
void generate_log_filename(char *log_filename, size_t size, int batch_size, int num_epochs);
// Declare the log filename as an external variable
extern char log_filename[256];

#endif // UTILITIES_H
