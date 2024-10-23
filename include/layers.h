// layers.h
#ifndef LAYERS_H
#define LAYERS_H

typedef struct {
    char *type;           // e.g. "dense"
    int input_size;
    int output_size;
    char *activation;     // e.g. "relu"
    float *weights;       // Host weights
    float *biases;        // Host biases
    float *d_weights;     // Device weights
    float *d_biases;      // Device biases
} Layer;

Layer* create_dense_layer(int input_size, int output_size, const char *activation);
void forward_layer(Layer *layer, float *d_input, float *d_output, int batch_size);
void backward_layer(Layer *layer, float *d_input, float *d_output_grad, float *d_input_grad, int batch_size);
void update_layer(Layer *layer, float learning_rate);
void free_layer(Layer *layer);
void print_layer(Layer *layer);

#endif // LAYERS_H
