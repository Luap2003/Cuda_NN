// main.cu
#include "../include/neural_net.h"
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    srand(42);
    // Load data
    //float *x_train, *y_train;
    //int num_samples, input_size, output_size;
    //
    //int epochs = 10;
    //int batch_size = 32;
    
    int num_layers = 2;
    NeuralNetwork *network = create_neural_net(num_layers);

    // Create layers
    Layer *layer1 = create_dense_layer(784, 128, "sigmoid");
    Layer *layer2 = create_dense_layer(128, 10, "sigmoid");

    // Add layers to the network
    add_layer_to_neural_net(network, layer1, 0);
    add_layer_to_neural_net(network, layer2, 1);

    // Print layers
    printf("Layer 1:\n");
    print_layer(layer1);
    printf("\nLayer 2:\n");
    print_layer(layer2);

    // Free resources
    free_neural_net(network);
    //free(x_train);
    //free(y_train);

    return 0;
}
