// neural_net.h
#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "layers.h"
#include "optimizer.h"
#include "lossFunction.h"

typedef struct {
    Layer **layers;
    int num_layers;
    Optimizer optimizer;
    LossFunction *loss_function;
} NeuralNetwork;

NeuralNetwork* create_neural_net(int num_layers);
void add_layer_to_neural_net(NeuralNetwork *network, Layer *layer, int index);
void compile_neural_Net(NeuralNetwork *model, const char *optimizer_name, const char *loss_name, const char *metric_name);
void backpropagation(NeuralNetwork *network, float *d_input, float *d_labels, int batch_size, float learning_rate);
void update_parameters(Layer *layer, float learning_rate);
void fit_neural_net(NeuralNetwork *network, float *x_train, float *y_train, int num_samples, int epochs, int batch_size);
void evaluate_neural_net(NeuralNetwork *model, float *x_test, float *y_test, int num_samples, float *loss, float *accuracy);
void predict_neural_net(NeuralNetwork *network, float *x_input, float *y_output, int num_samples);
void free_neural_net(NeuralNetwork *network);

#endif // NEURAL_NET_H
