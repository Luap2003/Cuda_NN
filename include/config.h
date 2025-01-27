// config.h
#ifndef CONFIG_H
#define CONFIG_H

#include "activations.h"

#ifdef __cplusplus
extern "C" {
#endif

// Define the Config struct
typedef struct Config {
    int *hidden_layers;
    int input_layer;
    int output_layer;
    int number_layers;
    int *layers_sizes;
    ActivationType *activation_functions;

    int batch_size;
    int num_epochs;
    float learning_rate;
    float decay_rate;
} Config;

// Declare the functions you want to use in C and C++ files
void remove_spaces(char* s);
int hidden_layers(Config *config, char *value);
int activation_functions(Config *config, char *value);
int parser(char *config_file, Config *config);

#ifdef __cplusplus
}
#endif

#endif // CONFIG_H
