// lossFunction.h
#ifndef lossFunction_H
#define lossFunction_H
typedef enum { LOSS_MSE, LOSS_CROSSENTROPY } LossFunction;

float compute_loss(float *d_predictions, float *d_labels, int size,
                   LossFunction loss_function);
__global__ void compute_loss_kernel(float *d_predictions, float *d_labels,
                                    float *d_loss, int size,
                                    LossFunction loss_function);

#endif // lossFunction_H
