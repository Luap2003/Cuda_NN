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

#endif // UTILITIES_H
