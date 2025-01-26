// config.h
#ifndef CONFIG_H
#define CONFIG_H

#include "activations.h"

typedef struct Config{
  int *hidden_layers;
  int input_layer;
  int output_layer;
  int number_layers;
  ActivationType *activation_functions;
} Config;

int parser(char *config_file, Config *config);
#endif // CONFIG_H
