// config.h
#ifndef CONFIG_H
#define CONFIG_H

#include "activations.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Struct holds all configuration options for the neural network.
 */
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

// void remove_spaces(char* s);
// int hidden_layers(Config *config, char *value);
// int activation_functions(Config *config, char *value);
 
/**
 * @brief Parses config file and populates @ref Config struct.
 *
 * @param config_file Path to config file.
 * @param config Pointer to @ref Config struct.
 */
int parser(const char *config_file, Config *config);

#ifdef __cplusplus
}
#endif

#endif // CONFIG_H
