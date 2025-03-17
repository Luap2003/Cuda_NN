// lossFunction.h

#include "layers.h"
#ifndef lossFunction_H
#define lossFunction_H
typedef enum {
    LOSS_MSE,
    LOSS_CROSSENTROPY
} LossFunction;

float compute_loss(float *d_predictions, float *d_labels, int size, LossFunction loss_function);
__global__ void compute_loss_kernel(float *d_predictions, float *d_labels, float *d_loss, int size, LossFunction loss_function);


void compute_custom_loss(float *predicted_d, float *target_d, 
    float *scale_y_d, float *mean_y_d, 
    float *scale_X_d, float *mean_X_d,
    int batch_size, int output_size, 
    float delta, float *loss, float *dZ_d);

#endif // lossFunction_H
