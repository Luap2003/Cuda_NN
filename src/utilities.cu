#include "../include/utilities.h"
#include <stdio.h>
#include <math.h>
#include <sys/ioctl.h>
#include <unistd.h>

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
}
