// neural_net.cu
#include "../include/neural_net.h"
#include <cuda_runtime.h>

NeuralNetwork* create_neural_net(int num_layers) {
    NeuralNetwork *network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    network->num_layers = num_layers;
    network->layers = (Layer**)malloc(num_layers * sizeof(Layer*));
    return network;
}

void add_layer_to_neural_net(NeuralNetwork *network, Layer *layer, int index) {
    if (index >= 0 && index < network->num_layers) {
        network->layers[index] = layer;
    }
}

void free_neural_net(NeuralNetwork *network) {
    for (int i = 0; i < network->num_layers; ++i) {
        free_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}
