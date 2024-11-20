#include <stdio.h>
#include <stdlib.h>
#include "../include/mnist.h"


int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function to read MNIST images
int read_mnist_images(const char *filename, float **data, int *num_images, int *image_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Could not open %s\n", filename);
        return -1;
    }

    int magic_number = 0;
    if (fread(&magic_number, sizeof(int), 1, fp) != 1) {
        printf("Error reading magic number from %s\n", filename);
        fclose(fp);
        return -1;
    }
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
        printf("Invalid magic number in %s\n", filename);
        fclose(fp);
        return -1;
    }

    int num_images_read = 0;
    if (fread(&num_images_read, sizeof(int), 1, fp) != 1) {
        printf("Error reading number of images from %s\n", filename);
        fclose(fp);
        return -1;
    }
    num_images_read = reverse_int(num_images_read);
    *num_images = num_images_read;

    int num_rows = 0;
    if (fread(&num_rows, sizeof(int), 1, fp) != 1) {
        printf("Error reading number of rows from %s\n", filename);
        fclose(fp);
        return -1;
    }
    num_rows = reverse_int(num_rows);

    int num_cols = 0;
    if (fread(&num_cols, sizeof(int), 1, fp) != 1) {
        printf("Error reading number of columns from %s\n", filename);
        fclose(fp);
        return -1;
    }
    num_cols = reverse_int(num_cols);

    *image_size = num_rows * num_cols;

    int total_size = (*num_images) * (*image_size);
    unsigned char *temp_data = (unsigned char *)malloc(total_size * sizeof(unsigned char));
    if (fread(temp_data, sizeof(unsigned char), total_size, fp) != total_size) {
        printf("Error reading image data from %s\n", filename);
        free(temp_data);
        fclose(fp);
        return -1;
    }

    *data = (float *)malloc(total_size * sizeof(float));
    for (int i = 0; i < total_size; ++i) {
        (*data)[i] = (float)temp_data[i] / 255.0f; // Normalize pixel values
    }

    free(temp_data);
    fclose(fp);
    return 0;
}

// Function to read MNIST labels
int read_mnist_labels(const char *filename, float **labels, int *num_labels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Could not open %s\n", filename);
        return -1;
    }

    int magic_number = 0;
    if (fread(&magic_number, sizeof(int), 1, fp) != 1) {
        printf("Error reading magic number from %s\n", filename);
        fclose(fp);
        return -1;
    }
    magic_number = reverse_int(magic_number);
    if (magic_number != 2049) {
        printf("Invalid magic number in %s\n", filename);
        fclose(fp);
        return -1;
    }

    int num_labels_read = 0;
    if (fread(&num_labels_read, sizeof(int), 1, fp) != 1) {
        printf("Error reading number of labels from %s\n", filename);
        fclose(fp);
        return -1;
    }
    num_labels_read = reverse_int(num_labels_read);
    *num_labels = num_labels_read;

    unsigned char *temp_labels = (unsigned char *)malloc((*num_labels) * sizeof(unsigned char));
    if (fread(temp_labels, sizeof(unsigned char), *num_labels, fp) != *num_labels) {
        printf("Error reading label data from %s\n", filename);
        free(temp_labels);
        fclose(fp);
        return -1;
    }

    // One-hot encode labels
    *labels = (float *)calloc((*num_labels) * 10, sizeof(float)); // 10 classes for digits 0-9
    for (int i = 0; i < *num_labels; ++i) {
        (*labels)[i * 10 + temp_labels[i]] = 1.0f;
    }

    free(temp_labels);
    fclose(fp);
    return 0;
}
