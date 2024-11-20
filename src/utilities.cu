#include "../include/utilities.h"
#include <stdio.h>
#include <math.h>

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

void display_epoch_progress(int current_epoch, int total_epochs, float loss, float accuracy, float epoch_time) {
    int bar_width = 100;
    float progress = (float)current_epoch / total_epochs;
    int pos = (int)(bar_width * progress);

    printf("\rEpoch %d/%d [", current_epoch, total_epochs);
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) {
            printf("=");
        } else if (i == pos) {
            printf(">");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%% | Loss: %.4f | Accuracy: %.2f%% | %.2f it/s", 
           progress * 100, loss, accuracy, 1.0f / epoch_time);
    fflush(stdout); // Force immediate output
}