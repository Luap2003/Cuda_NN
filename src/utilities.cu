#include "../include/utilities.h"
#include <stdio.h>
#include <math.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <time.h>
// Define the log filename with a default value
char log_filename[256] = "training_log_cuda.csv";

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

    // Use the global log_filename variable instead of a hard-coded string
    // const char *log_filename = "training_log_cuda.csv"; // Remove this line

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

    // **Logging Section End**
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
