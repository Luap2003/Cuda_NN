// NeuralNetwork.c
#include "../include/neural_net.h"
#include <stdio.h>
#include <stdlib.h>

void neural_network_init(NeuralNetwork *nn, int num_layers, int *layer_sizes, ActivationType *activations, int batch_size, int num_epochs, float learning_rate) {
    nn->num_layers = num_layers;

    // Allocate memory for layer sizes and activations
    nn->layer_sizes = (int *)malloc(num_layers * sizeof(int));
    nn->activations = (ActivationType *)malloc((num_layers - 1) * sizeof(ActivationType)); // Correct size

    // Copy layer sizes
    for (int i = 0; i < num_layers; ++i) {
        nn->layer_sizes[i] = layer_sizes[i];
    }

    // Copy activations
    for (int i = 0; i < num_layers - 1; ++i) {
        nn->activations[i] = activations[i];
    }

    nn->batch_size = batch_size;
    nn->num_epochs = num_epochs;
    nn->learning_rate = learning_rate;

    // Initialize cuBLAS handle
    cublasStatus_t status = cublasCreate(&nn->handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }

    // Allocate layers
    nn->layers = (Layer *)malloc((num_layers - 1) * sizeof(Layer));

    // Initialize layers
    for (int i = 0; i < num_layers - 1; ++i) {
        int n_in = nn->layer_sizes[i];
        int n_out = nn->layer_sizes[i + 1];
        ActivationType aktfunc = nn->activations[i]; // Correct indexing
        layer_init(&nn->layers[i], batch_size, n_in, n_out, aktfunc);
    }
}

void neural_network_train(NeuralNetwork *nn, float *train_images, float *train_labels, int num_train_samples) {
    int m = nn->batch_size;
    int num_batches = num_train_samples / m;

    int input_size = nn->layer_sizes[0];
    int output_size = nn->layer_sizes[nn->num_layers - 1];

    size_t input_batch_size = input_size * m * sizeof(float);
    size_t label_batch_size = output_size * m * sizeof(float);

    // Allocate device memory for input data and labels
    float *X_train_d;
    float *Y_train_d;

    cudaMalloc((void**)&X_train_d, input_batch_size);
    cudaMalloc((void**)&Y_train_d, label_batch_size);

    for (int epoch = 0; epoch < nn->num_epochs; ++epoch) {
        printf("Epoch %d/%d\n", epoch + 1, nn->num_epochs);

        float total_loss = 0.0f;
        int total_samples = 0;
        int correct_predictions = 0;

        for (int batch = 0; batch < num_batches; ++batch) {
            // Get batch data
            float *X_batch = train_images + batch * m * input_size;
            float *Y_batch = train_labels + batch * m * output_size;

            // Copy batch data to device
            cudaMemcpy(X_train_d, X_batch, input_batch_size, cudaMemcpyHostToDevice);
            cudaMemcpy(Y_train_d, Y_batch, label_batch_size, cudaMemcpyHostToDevice);

            // Forward propagation
            float *A_prev_d = X_train_d;
            for (int i = 0; i < nn->num_layers - 1; ++i) {
                layer_forward(&nn->layers[i], A_prev_d, nn->handle);
                A_prev_d = nn->layers[i].A_d;
            }

            // Copy predictions back to host
            Layer *output_layer = &nn->layers[nn->num_layers - 2];
            float *A_output_h = (float *)malloc(output_size * m * sizeof(float));
            cudaMemcpy(A_output_h, output_layer->A_d, output_size * m * sizeof(float), cudaMemcpyDeviceToHost);

            // Compute batch loss and accuracy
            float batch_loss = 0.0f;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < output_size; ++j) {
                    float y_ij = Y_batch[i * output_size + j];
                    float p_ij = A_output_h[i * output_size + j];
                    batch_loss -= y_ij * logf(p_ij + 1e-7f);
                }
            }
            total_loss += batch_loss;
            total_samples += m;

            // Compute accuracy
            for (int i = 0; i < m; ++i) {
                int predicted_label = 0;
                float max_prob = A_output_h[i * output_size];
                for (int j = 1; j < output_size; ++j) {
                    if (A_output_h[i * output_size + j] > max_prob) {
                        max_prob = A_output_h[i * output_size + j];
                        predicted_label = j;
                    }
                }
                // Get true label
                int true_label = 0;
                for (int j = 0; j < output_size; ++j) {
                    if (Y_batch[i * output_size + j] == 1.0f) {
                        true_label = j;
                        break;
                    }
                }
                if (predicted_label == true_label) {
                    correct_predictions++;
                }
            }

            free(A_output_h);

            // Backward propagation
            // Output layer
            backward_output_layer(&nn->layers[nn->num_layers - 2], Y_train_d, nn->layers[nn->num_layers - 3].A_d, nn->handle);
            // Hidden layers
            for (int i = nn->num_layers - 3; i >= 0; --i) {
                float *A_prev_d = (i == 0) ? X_train_d : nn->layers[i - 1].A_d;
                float *W_next_d = nn->layers[i + 1].w_d;
                float *dZ_next_d = nn->layers[i + 1].dZ_d;
                int n_out_next = nn->layers[i + 1].n_out;
                backward_layer(&nn->layers[i], W_next_d, dZ_next_d, A_prev_d, n_out_next, nn->handle);
            }

            // Update weights
            for (int i = 0; i < nn->num_layers - 1; ++i) {
                update(&nn->layers[i], nn->learning_rate);
            }
        }

        float average_loss = total_loss / total_samples;
        float accuracy = (float)correct_predictions / total_samples * 100.0f;
        printf("Loss after epoch %d: %.4f, Accuracy: %.2f%%\n", epoch + 1, average_loss, accuracy);
    }

    cudaFree(X_train_d);
    cudaFree(Y_train_d);
}

void neural_network_evaluate(NeuralNetwork *nn, float *test_images, float *test_labels, int num_test_samples) {
    int input_size = nn->layer_sizes[0];
    int output_size = nn->layer_sizes[nn->num_layers - 1];
    int m = num_test_samples;

    size_t input_size_bytes = input_size * m * sizeof(float);
    size_t label_size_bytes = output_size * m * sizeof(float);

    // Allocate device memory for test data
    float *X_test_d;
    float *Y_test_d;

    cudaMalloc((void**)&X_test_d, input_size_bytes);
    cudaMalloc((void**)&Y_test_d, label_size_bytes);

    // Copy test data to device
    cudaMemcpy(X_test_d, test_images, input_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y_test_d, test_labels, label_size_bytes, cudaMemcpyHostToDevice);

    // Update layers batch size and reallocate memory if necessary
    for (int i = 0; i < nn->num_layers - 1; ++i) {
        nn->layers[i].m = m;
        // Reallocate A_d, Z_d, dZ_d
        cudaFree(nn->layers[i].A_d);
        cudaFree(nn->layers[i].Z_d);
        cudaFree(nn->layers[i].dZ_d);

        size_t A_size = nn->layers[i].n_out * m * sizeof(float);
        cudaMalloc((void**)&nn->layers[i].A_d, A_size);
        cudaMalloc((void**)&nn->layers[i].Z_d, A_size);
        cudaMalloc((void**)&nn->layers[i].dZ_d, A_size);
    }

    // Forward propagation
    float *A_prev_d = X_test_d;
    for (int i = 0; i < nn->num_layers - 1; ++i) {
        layer_forward(&nn->layers[i], A_prev_d, nn->handle);
        A_prev_d = nn->layers[i].A_d;
    }

    // Copy predictions back to host
    Layer *output_layer = &nn->layers[nn->num_layers - 2];
    float *A_output_h = (float *)malloc(output_size * m * sizeof(float));
    cudaMemcpy(A_output_h, output_layer->A_d, output_size * m * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute accuracy
    int correct_predictions = 0;

    for (int i = 0; i < m; ++i) {
        int predicted_label = 0;
        float max_prob = A_output_h[i * output_size];
        for (int j = 1; j < output_size; ++j) {
            if (A_output_h[i * output_size + j] > max_prob) {
                max_prob = A_output_h[i * output_size + j];
                predicted_label = j;
            }
        }
        // Get true label
        int true_label = 0;
        for (int j = 0; j < output_size; ++j) {
            if (test_labels[i * output_size + j] == 1.0f) {
                true_label = j;
                break;
            }
        }
        if (predicted_label == true_label) {
            correct_predictions++;
        }
    }

    float test_accuracy = (float)correct_predictions / m * 100.0f;
    printf("Test Accuracy: %.2f%%\n", test_accuracy);

    free(A_output_h);
    cudaFree(X_test_d);
    cudaFree(Y_test_d);
}

void free_neural_network(NeuralNetwork *nn) {
    for (int i = 0; i < nn->num_layers - 1; ++i) {
        free_layer(&nn->layers[i]);
    }
    free(nn->layers);
    free(nn->layer_sizes);
    free(nn->activations);

    cublasDestroy(nn->handle);
}
