// NeuralNetwork.c
#include "../include/neural_net.h"
#include "../include/utilities.h"
#include <cstdio>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
// Combined kernel: gathers both images and labels in one launch.
__global__ void gather_batch_two_kernel(const float *src_images,
                                          const float *src_labels,
                                          const int   *indices,
                                          int          batch_start,
                                          int          current_batch_size,
                                          int          image_sample_size, // e.g. input_size
                                          int          label_sample_size, // e.g. output_size
                                          float       *dest_images,
                                          float       *dest_labels)
{
    // Total number of elements to gather from images and labels:
    int total_images = current_batch_size * image_sample_size;
    int total_labels = current_batch_size * label_sample_size;
    int total_elements = total_images + total_labels;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        if (idx < total_images) {
            // This thread gathers an element from the images buffer.
            int sample_index = idx / image_sample_size;     // which sample in the batch
            int feature_index = idx % image_sample_size;      // which feature within the sample
            // Look up the global sample index from the shuffled indices.
            int global_index = indices[batch_start + sample_index];
            dest_images[idx] = src_images[global_index * image_sample_size + feature_index];
        }
        else {
            // Adjust the index for the labels part.
            int j = idx - total_images;
            int sample_index = j / label_sample_size;         // which sample in the batch
            int feature_index = j % label_sample_size;          // which label within the sample
            int global_index = indices[batch_start + sample_index];
            dest_labels[j] = src_labels[global_index * label_sample_size + feature_index];
        }
    }
}



void neural_network_init(NeuralNetwork *nn, int num_layers, int *layer_sizes, ActivationType *activations, int batch_size, int num_epochs, float learning_rate, float decay_rate) {
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
    nn->initial_learning_rate = learning_rate;
    nn->decay_rate = decay_rate;

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

void neural_network_train(NeuralNetwork *nn,
                          float *train_images,
                          float *train_labels,
                          int num_train_samples) {
    int m = nn->batch_size;
    int num_batches = (num_train_samples + m - 1) / m;
    int input_size = nn->layer_sizes[0];
    int output_size = nn->layer_sizes[nn->num_layers - 1];

    // --- Preload the full dataset into device memory ---
    size_t full_input_size = input_size * num_train_samples * sizeof(float);
    size_t full_label_size = output_size * num_train_samples * sizeof(float);
    float *d_train_images, *d_train_labels;
    cudaMalloc(&d_train_images, full_input_size);
    cudaMalloc(&d_train_labels, full_label_size);
    cudaMemcpy(d_train_images, train_images, full_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, train_labels, full_label_size, cudaMemcpyHostToDevice);

    // --- Allocate device buffers for the current batch ---
    float *d_X_batch, *d_Y_batch;
    cudaMalloc(&d_X_batch, input_size * m * sizeof(float));
    cudaMalloc(&d_Y_batch, output_size * m * sizeof(float));

    // --- Allocate and initialize the indices array ---
    int *d_indices;
    cudaMalloc(&d_indices, num_train_samples * sizeof(int));
    int *h_indices = (int *)malloc(num_train_samples * sizeof(int));
    for (int i = 0; i < num_train_samples; i++)
        h_indices[i] = i;
    cudaMemcpy(d_indices, h_indices, num_train_samples * sizeof(int), cudaMemcpyHostToDevice);

    // Main training loop over epochs.
    for (int epoch = 0; epoch < nn->num_epochs; ++epoch) {
        clock_t epoch_start_time = clock();

        // Shuffle the indices on the host.
        for (int i = num_train_samples - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            int temp = h_indices[i];
            h_indices[i] = h_indices[j];
            h_indices[j] = temp;
        }
        cudaMemcpy(d_indices, h_indices, num_train_samples * sizeof(int), cudaMemcpyHostToDevice);

        // Optionally decay the learning rate.
        nn->learning_rate = nn->initial_learning_rate * expf(-nn->decay_rate * epoch);
        float total_loss = 0.0f;
        int total_samples = 0;
        int correct_predictions = 0;

        // Loop over batches.
        for (int batch = 0; batch < num_batches; ++batch) {
            int batch_start = batch * m;
            int current_batch_size = ((batch_start + m) > num_train_samples) ?
                                     (num_train_samples - batch_start) : m;

            // --- Gather the batch data from the full dataset on the device ---

            int total_images = current_batch_size * input_size;
            int total_labels = current_batch_size * output_size;
            int total_elements = total_images + total_labels;
            int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            // Launch the combined kernel.
            gather_batch_two_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_train_images, d_train_labels, d_indices,
                                                                batch_start, current_batch_size,
                                                                input_size, output_size,
                                                                d_X_batch, d_Y_batch);
            cudaCheckError();

            // Update the current batch size in the network and its layers.
            nn->batch_size = current_batch_size;
            for (int i = 0; i < nn->num_layers - 1; ++i)
                nn->layers[i].m = current_batch_size;

            // --- Forward propagation ---
            float *A_prev_d = d_X_batch;
            for (int i = 0; i < nn->num_layers - 1; ++i) {
                layer_forward(&nn->layers[i], A_prev_d, nn->handle);
                A_prev_d = nn->layers[i].A_d;
            }

            // Copy the network’s output (predictions) back to host to compute loss/accuracy.
            Layer *output_layer = &nn->layers[nn->num_layers - 2];
            float *A_output_h = (float *)malloc(output_size * current_batch_size * sizeof(float));
            cudaMemcpy(A_output_h, output_layer->A_d, output_size * current_batch_size * sizeof(float),
                       cudaMemcpyDeviceToHost);

            // --- Compute loss (cross-entropy) and accuracy on the host ---
            float batch_loss = 0.0f;
            for (int i = 0; i < current_batch_size; ++i) {
                for (int j = 0; j < output_size; ++j) {
                    // Since the training labels are still on the host,
                    // use the shuffled indices to locate the proper label.
                    int global_index = h_indices[batch_start + i];
                    float y_ij = train_labels[global_index * output_size + j];
                    float p_ij = A_output_h[i * output_size + j];
                    float p_ij_clamped = fmaxf(p_ij, 1e-7f);
                    batch_loss -= y_ij * logf(p_ij_clamped);
                }
            }
            total_loss += batch_loss;
            total_samples += current_batch_size;

            for (int i = 0; i < current_batch_size; ++i) {
                int predicted_label = 0;
                float max_prob = A_output_h[i * output_size];
                for (int j = 1; j < output_size; ++j) {
                    if (A_output_h[i * output_size + j] > max_prob) {
                        max_prob = A_output_h[i * output_size + j];
                        predicted_label = j;
                    }
                }
                int true_label = 0;
                for (int j = 0; j < output_size; ++j) {
                    int global_index = h_indices[batch_start + i];
                    if (train_labels[global_index * output_size + j] == 1.0f) {
                        true_label = j;
                        break;
                    }
                }
                if (predicted_label == true_label)
                    correct_predictions++;
            }
            free(A_output_h);

            // --- Backward propagation ---
            // For the output layer use d_Y_batch as the true labels.
            backward_output_layer(&nn->layers[nn->num_layers - 2],
                                    d_Y_batch,
                                    nn->layers[nn->num_layers - 3].A_d,
                                    nn->handle);
            // Propagate backwards through the hidden layers.
            for (int i = nn->num_layers - 3; i >= 0; --i) {
                float *A_prev_d = (i == 0) ? d_X_batch : nn->layers[i - 1].A_d;
                float *W_next_d = nn->layers[i + 1].w_d;
                float *dZ_next_d = nn->layers[i + 1].dZ_d;
                int n_out_next = nn->layers[i + 1].n_out;
                backward_layer(&nn->layers[i], W_next_d, dZ_next_d, A_prev_d, n_out_next, nn->handle);
            }

            // --- Update the weights ---
            for (int i = 0; i < nn->num_layers - 1; ++i)
                update(&nn->layers[i], nn->learning_rate);
        } // end for (batch)

        float average_loss = total_loss / total_samples;
        float accuracy = (float)correct_predictions / total_samples * 100.0f;
        clock_t epoch_end_time = clock();
        float epoch_time = (float)(epoch_end_time - epoch_start_time) / CLOCKS_PER_SEC;
        float batches_per_second = num_batches / epoch_time;

        // Display the epoch progress
        display_epoch_progress(epoch + 1, nn->num_epochs, average_loss, accuracy, epoch_time, batches_per_second);
        #ifdef LOG
        log_epoch_progress(epoch + 1, nn->num_epochs, average_loss, accuracy, epoch_time, batches_per_second);

        log_weights(nn, epoch + 1);
        log_biases(nn, epoch+1);
        #endif
        // Optionally, print progress information.
        //printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%, Time = %.2fs, Batches/s = %.2f\n",
        //       epoch + 1, average_loss, accuracy, epoch_time, batches_per_second);
    }
    printf("\n");

    // Free all allocated device and host buffers.
    cudaFree(d_train_images);
    cudaFree(d_train_labels);
    cudaFree(d_X_batch);
    cudaFree(d_Y_batch);
    cudaFree(d_indices);
    free(h_indices);
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
