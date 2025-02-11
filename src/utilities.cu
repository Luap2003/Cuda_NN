#include "../include/utilities.h"
#include "../include/neural_net.h"
#include <stdio.h>
#include <math.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <time.h>

#ifdef LOG
// Define the log filename with a default value
char log_filename[256];
char log_filename_weights[256];
char log_filename_biases[256];
#endif

void display_progress_bar(int current, int total, int bar_width) {
    float progress = (float)current / total;
    int pos = (int)(bar_width * progress);
    printf("[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) {
            printf("=");
        } else if (i == pos) {
            printf(">");
        } else {
            printf(" ");
        }
    }
    printf("] %d%%\r", (int)(progress * 100));
    fflush(stdout); // Force the output to be written immediately
}
/**
 * Displays the progress of the current epoch, including metrics and performance.
 */
void display_epoch_progress(int current_epoch, int total_epochs, float loss, float accuracy, float epoch_time, float batches_per_second) {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int terminal_width = w.ws_col;

    // Create prefix and measure its length
    char prefix[100];
    int prefix_length = snprintf(prefix, sizeof(prefix), "Epoch %d/%d [", current_epoch, total_epochs);

    // Create suffix and measure its length
    char suffix[200];
    int suffix_length = snprintf(suffix, sizeof(suffix), "] %.2f%% | Loss: %.4f | Accuracy: %.2f%% | %.2f b/s",
                                 ((float)current_epoch / total_epochs) * 100.0f, loss, accuracy, batches_per_second);

    // Compute bar width
    int bar_width = terminal_width - prefix_length - suffix_length;

    if (bar_width < 0) {
        bar_width = 0;
    }

    // Compute progress
    float progress = (float)current_epoch / total_epochs;
    int pos = (int)(bar_width * progress);

    // Print the progress bar
    printf("\r%s", prefix);
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) {
            printf("=");
        } else if (i == pos) {
            printf(">");
        } else {
            printf(" ");
        }
    }
    printf("%s", suffix);
    fflush(stdout);
}

#ifdef LOG
/**
 * Logs the progress of the current epoch to the log file.
 */
void log_epoch_progress(int current_epoch, int total_epochs, float loss, float accuracy, float epoch_time, float batches_per_second) {
    // Compute progress percentage
    float progress = (float)current_epoch / total_epochs;

    // Open the log file in append mode
    FILE *log_file = fopen(log_filename, "a");
    if (log_file == NULL) {
        // If the file can't be opened, print an error message and return
        fprintf(stderr, "\nError: Could not open log file %s for writing.\n", log_filename);
        return;
    }

    // Write a header if it's the first epoch
    if (current_epoch == 1) {
        fprintf(log_file, "Epoch,Progress%%,Loss,Accuracy%%,Epoch Time(s),Batches/s\n");
    }

    // Write the epoch data to the log file
    fprintf(log_file, "%d,%.2f,%.4f,%.2f,%.2f,%.2f\n",
            current_epoch,
            progress * 100.0f,
            loss,
            accuracy,
            epoch_time,
            batches_per_second);

    // Close the log file
    fclose(log_file);
}

void generate_log_filename(char *log_filename, size_t size, int batch_size, int num_epochs) {
    time_t raw_time;
    struct tm *time_info;
    char time_buffer[20]; // Buffer to hold formatted time

    // Get the current time
    time(&raw_time);
    time_info = localtime(&raw_time);

    // Format the time as YYYYMMDD_HHMMSS
    strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", time_info);

    // Create the log filename with batch size, epochs, and timestamp
    snprintf(log_filename, size, "logs/temp/training_log_cuda_%s_bs%d_ep%d.csv", time_buffer, batch_size, num_epochs);
}

void generate_weights_biases_log_filenames(char *weights_log, size_t w_size, char *biases_log, size_t b_size, int batch_size, int num_epochs) {
    time_t raw_time;
    struct tm *time_info;
    char time_buffer[20]; // Buffer to hold formatted time

    // Get the current time
    time(&raw_time);
    time_info = localtime(&raw_time);

    // Format the time as YYYYMMDD_HHMMSS
    strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", time_info);

    // Create the weights and biases log filenames with batch size, epochs, and timestamp
    snprintf(weights_log, w_size, "logs/temp/weights_log_cuda_%s_bs%d_ep%d.csv", time_buffer, batch_size, num_epochs);
    snprintf(biases_log, b_size, "logs/temp/biases_log_cuda_%s_bs%d_ep%d.csv", time_buffer, batch_size, num_epochs);
}

void log_weights(NeuralNetwork *nn, int epoch) {
    // Open the weights log file in append mode
    FILE *weights_log_file = fopen(log_filename_weights, "a");
    if (weights_log_file == NULL) {
        fprintf(stderr, "\nError: Could not open weights log file %s for writing.\n", log_filename_weights);
        return;
    }

    // Write header if it's the first epoch
    if (epoch == 1) {
        fprintf(weights_log_file, "Epoch");
        
        // Calculate total number of weights
        int total_weights = 0;
        for (int i = 0; i < nn->num_layers - 1; ++i) {
            total_weights += nn->layers[i].n_in * nn->layers[i].n_out;
        }

        // Optionally, you can write individual weight headers like Weight1, Weight2, ..., WeightN
        // For simplicity and to avoid excessively long headers, we'll skip individual weight names
        fprintf(weights_log_file, ",Weights\n");
    }

    // Start writing the epoch number
    fprintf(weights_log_file, "%d", epoch);

    // Iterate over each layer and append all weights
    for (int i = 0; i < nn->num_layers - 1; ++i) { // Excluding input layer
        Layer *layer = &nn->layers[i];
        int n_in = layer->n_in;   // Number of input connections
        int n_out = layer->n_out; // Number of neurons in this layer

        // Total number of weights in this layer
        size_t num_weights = (size_t)n_in * n_out;

        // Allocate host memory to hold weights
        float *weights_h = (float *)malloc(num_weights * sizeof(float));
        if (weights_h == NULL) {
            fprintf(stderr, "\nError: Could not allocate host memory for weights.\n");
            fclose(weights_log_file);
            return;
        }

        // Ensure all CUDA operations are complete before copying
        cudaError_t cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "\nError: CUDA synchronization failed: %s\n", cudaGetErrorString(cuda_status));
            free(weights_h);
            fclose(weights_log_file);
            return;
        }

        // Copy weights from device to host
        cuda_status = cudaMemcpy(weights_h, layer->w_d, num_weights * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "\nError: CUDA memcpy failed: %s\n", cudaGetErrorString(cuda_status));
            free(weights_h);
            fclose(weights_log_file);
            return;
        }

        // Append weights to the log file, separated by commas
        for (size_t w = 0; w < num_weights; w+=10) {
            fprintf(weights_log_file, ",%.6f", weights_h[w]);
        }

        // Free host memory for weights
        free(weights_h);
    }

    // End the line for this epoch
    fprintf(weights_log_file, "\n");

    // Close the weights log file
    fclose(weights_log_file);
}

/**
 * Logs all biases of the neural network into a CSV file.
 * Each line corresponds to an epoch and contains the epoch number followed by all biases.
 *
 * @param nn    Pointer to the NeuralNetwork structure.
 * @param epoch Current epoch number.
 */
void log_biases(NeuralNetwork *nn, int epoch) {
    // Open the biases log file in append mode
    FILE *biases_log_file = fopen(log_filename_biases, "a");
    if (biases_log_file == NULL) {
        fprintf(stderr, "\nError: Could not open biases log file %s for writing.\n", log_filename_biases);
        return;
    }

    // Write header if it's the first epoch
    if (epoch == 1) {
        fprintf(biases_log_file, "Epoch");
        
        // Calculate total number of biases
        int total_biases = 0;
        for (int i = 0; i < nn->num_layers - 1; ++i) {
            total_biases += nn->layers[i].n_out;
        }

        // Optionally, you can write individual bias headers like Bias1, Bias2, ..., BiasN
        // For simplicity and to avoid excessively long headers, we'll skip individual bias names
        fprintf(biases_log_file, ",Biases\n");
    }

    // Start writing the epoch number
    fprintf(biases_log_file, "%d", epoch);

    // Iterate over each layer and append all biases
    for (int i = 0; i < nn->num_layers - 1; ++i) { // Excluding input layer
        Layer *layer = &nn->layers[i];
        int n_out = layer->n_out; // Number of neurons in this layer

        // Total number of biases in this layer
        size_t num_biases = (size_t)n_out;

        // Allocate host memory to hold biases
        float *biases_h = (float *)malloc(num_biases * sizeof(float));
        if (biases_h == NULL) {
            fprintf(stderr, "\nError: Could not allocate host memory for biases.\n");
            fclose(biases_log_file);
            return;
        }

        // Ensure all CUDA operations are complete before copying
        cudaError_t cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "\nError: CUDA synchronization failed: %s\n", cudaGetErrorString(cuda_status));
            free(biases_h);
            fclose(biases_log_file);
            return;
        }

        // Copy biases from device to host
        cuda_status = cudaMemcpy(biases_h, layer->b_d, num_biases * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "\nError: CUDA memcpy failed: %s\n", cudaGetErrorString(cuda_status));
            free(biases_h);
            fclose(biases_log_file);
            return;
        }

        // Append biases to the log file, separated by commas
        for (size_t b = 0; b < num_biases; ++b) {
            fprintf(biases_log_file, ",%.6f", biases_h[b]);
        }

        // Free host memory for biases
        free(biases_h);
    }

    // End the line for this epoch
    fprintf(biases_log_file, "\n");

    // Close the biases log file
    fclose(biases_log_file);
}
#endif
