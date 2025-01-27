// config.c
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/activations.h"
#include "../include/config.h"

void remove_spaces(char* s) {
  char* d = s;
  do {
    while (*d == ' ') {
      ++d;
    }
  } while ((*s++ = *d++));
}

int parse_config(char *config_file, Config *config){

  
  char line[256];
  FILE *file = fopen("config.txt", "r");
  if(file == NULL){
    printf("Can not open file!\n");
    return -1;
  }

  while(fgets(line, sizeof(line), file)){

    char *key;
    char *value;
    remove_spaces(line);

    
    if(line[0] == '#')
      continue;
    if(line[0] == '\n')
      continue;

    
    line[strcspn(line, "\n")] = '\0';
    key = strtok(line, "=");
    value = strtok(NULL, "=");

    if(strcmp(key, "input_layer") == 0)
      config->input_layer = atoi(value);

    if(strcmp(key, "output_layer") == 0)
      config->output_layer = atoi(value);


      int count = 0;
      int size = 1;
      for (const char *p = value; *p; p++) {
          if (*p == ',') {
              size++;
          }
      }

      config->number_layers = size + 2;
      config->hidden_layers = (int *)malloc(size *sizeof(int));
      if(config->hidden_layers == NULL){
        free(config->hidden_layers);
        printf("Memory allocatoins failed! \n");
        return -1;
        return 1;
      }

      char *start = value + 1;
      char *end = start + strlen(start)-1;
      *end = '\0';

      char *token = strtok(start, ",");

      while(token != NULL){
        while(isspace((unsigned char)*token)){
          token++;
        }

        config->hidden_layers[count] = atoi(token);
        count++;

        token = strtok(NULL, ",");
      }
      
    } 

      int size = 1;
      for (const char *p = value; *p; p++) {
          if (*p == ',') {
              size++;
          }
      }

      config->activation_functions = (ActivationType *)malloc(size *sizeof(ActivationType));

      char** str_arr;
      str_arr = (char**)malloc(size * (sizeof(char*)));
      if(str_arr == NULL){
        printf("Memory allocatoins failed! \n");
        return -1;
        return 1;
      }

      char *start = value + 1;
      char *end = start + strlen(start)-1;
      *end = '\0';

      char *token = strtok(start, ",");

      int pos = 0;
      while(token != NULL){
        while(isspace((unsigned char)*token)){
          token++;
        }

        int len = strlen(token);
        str_arr[pos] = (char*)malloc(len * (sizeof(char)));
        if(str_arr[pos] == NULL){
          printf("Memory allocation failed!\n");
          return -1;
          return 1;
        }
        for(int i = 0; i<len; i++){
          str_arr[pos][i] = token[i];
        }
        if(strcmp(str_arr[pos], "ACTIVATION_RELU")==0)
          config->activation_functions[pos] = ACTIVATION_RELU;
        if(strcmp(str_arr[pos], "ACTIVATION_SIGMOID")==0)
          config->activation_functions[pos] = ACTIVATION_SIGMOID;
        if(strcmp(str_arr[pos], "ACTIVATION_LINEAR")==0)
          config->activation_functions[pos] = ACTIVATION_LINEAR;
        pos++;


        token = strtok(NULL, ",");
      }
      
int parser(char *config_file, Config *config){

  
  char line[256];
  FILE *file = fopen("config.txt", "r");
  if(file == NULL){
    printf("Can not open file!\n");
    return 1;
  }

  while(fgets(line, sizeof(line), file)){

    char *key;
    char *value;
    remove_spaces(line);

    
    if(line[0] == '#')
      continue;
    if(line[0] == '/')
      if(line[1] == '/')
        continue;
    if(line[0] == '\n')
      continue;

    
    line[strcspn(line, "\n")] = '\0';
    key = strtok(line, "=");
    value = strtok(NULL, "=");

    
    if(strcmp(key, "batch_size") == 0)
      config->batch_size = atoi(value);

    if(strcmp(key, "num_epochs") == 0)
      config->num_epochs = atoi(value);

    if(strcmp(key, "learning_rate") == 0)
      config->learning_rate = atof(value);

    if(strcmp(key, "decay_rate") == 0)
      config->decay_rate = atof(value);

    if(strcmp(key, "input_layer") == 0)
      config->input_layer = atoi(value);

    if(strcmp(key, "output_layer") == 0)
      config->output_layer = atoi(value);

    if(strcmp(key, "hidden_layers") == 0){
    if(strcmp(key, "activation_functions") == 0){
    } 
  }

  fclose(file);

  // printf("Hidden Layers:");
  // for(int i = 0; i < config->number_layers-2; i++){
  //   printf(" %d",config->hidden_layers[i]);
  //   if(i < config->number_layers -3)
  //     printf(",");
  // }
  // printf("\n");
  // printf("Input Layer: %d\n", config->input_layer);
  // printf("Output Layer: %d\n", config->output_layer);
  // printf("Number of Layers: %d\n", config->number_layers);
  // printf("Activation Functions:");
  // for(int i = 0; i < config->number_layers; i++){
  //   printf(" %d", config->activation_functions[i]);
  //   if(i < config->number_layers -1)
  //     printf(",");
  // }
  // printf("\n");

  // int layer_sizes[config->number_layers];
  // layer_sizes[0] = config->input_layer;
  // for(int i = 0; i < config->number_layers-2; i++){
  //   layer_sizes[i] = config->hidden_layers[i];
  // }
  // layer_sizes[config->number_layers-1] = config->output_layer;

  // ActivationType test[config->number_layers];
  // for(int i = 0; i<config->number_layers; i++){
  //   test[i] = (ActivationType) config->activation_functions[i];
  // }

  // free(config->hidden_layers);
  // free(config->activation_functions);
  
  
  config->layers_sizes = (int *) malloc(config->number_layers * sizeof(int));
  config->layers_sizes[0] = config->input_layer;
  if(config->hidden_layers != NULL){
    for(int i = 0; i<config->number_layers-2; i++){
      config->layers_sizes[i+1] = config->hidden_layers[i];
    }
  }
  config->layers_sizes[config->number_layers-1] = config->output_layer;



  
  #if defined(DEBUG) || defined(CONFIG_DEBUG)
  printf("Hidden Layers:");
  if(config->hidden_layers != NULL){
    for(int i = 0; i < config->number_layers-2; i++){
      printf(" %d",config->hidden_layers[i]);
      if(i < config->number_layers -3)
        printf(",");
    }
  }
  printf("\n");
  printf("Input Layer: %d\n", config->input_layer);
  printf("Output Layer: %d\n", config->output_layer);
  printf("Number of Layers: %d\n", config->number_layers);
  printf("Activation Functions:");
  if(config->activation_functions != NULL){
    for(int i = 0; i < config->number_layers-1; i++){
      printf(" %d", config->activation_functions[i]);
      if(i < config->number_layers - 2)
        printf(",");
    }
  }
  printf("\n");

  #endif

  return 0;
}

