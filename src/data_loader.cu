#include "../include/data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Function to swap endianness of a 32-bit unsigned integer
uint32_t swap_uint32(uint32_t val) {
    return ((val << 24) & 0xff000000 ) |
           ((val << 8)  & 0x00ff0000 ) |
           ((val >> 8)  & 0x0000ff00 ) |
           ((val >> 24) & 0x000000ff );
}

float* load_mnist_images(const char* filename, int* number_of_images, int* image_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Could not open %s\n", filename);
        exit(1);
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    if (fread(&magic_number, sizeof(uint32_t), 1, fp) != 1) {
        printf("Error reading magic number from %s\n", filename);
        fclose(fp);
        exit(1);
    }
    magic_number = swap_uint32(magic_number);

    if (magic_number != 2051) {
        printf("Invalid MNIST image file: %s\n", filename);
        fclose(fp);
        exit(1);
    }

    if (fread(&num_images, sizeof(uint32_t), 1, fp) != 1) {
        printf("Error reading number of images from %s\n", filename);
        fclose(fp);
        exit(1);
    }
    num_images = swap_uint32(num_images);

    if (fread(&num_rows, sizeof(uint32_t), 1, fp) != 1) {
        printf("Error reading number of rows from %s\n", filename);
        fclose(fp);
        exit(1);
    }
    num_rows = swap_uint32(num_rows);

    if (fread(&num_cols, sizeof(uint32_t), 1, fp) != 1) {
        printf("Error reading number of columns from %s\n", filename);
        fclose(fp);
        exit(1);
    }
    num_cols = swap_uint32(num_cols);

    *number_of_images = num_images;
    *image_size = num_rows * num_cols;

    printf("Number of images: %d\n", num_images);
    printf("Image size: %d x %d\n", num_rows, num_cols);

    float *images = (float*)malloc(num_images * num_rows * num_cols * sizeof(float));
    if (!images) {
        printf("Failed to allocate memory for images\n");
        fclose(fp);
        exit(1);
    }

    unsigned char *temp_image = (unsigned char*)malloc(num_rows * num_cols);
    if (!temp_image) {
        printf("Failed to allocate memory for a temporary image\n");
        free(images);
        fclose(fp);
        exit(1);
    }

    for (int i = 0; i < num_images; i++) {
        size_t read = fread(temp_image, sizeof(unsigned char), num_rows * num_cols, fp);
        if (read != num_rows * num_cols) {
            printf("Failed to read image data for image %d\n", i);
            free(temp_image);
            free(images);
            fclose(fp);
            exit(1);
        }
        for (int j = 0; j < num_rows * num_cols; j++) {
            images[i * num_rows * num_cols + j] = temp_image[j] / 255.0f; // Normalize pixel values to [0,1]
        }
    }

    free(temp_image);
    fclose(fp);
    return images;
}

float* load_mnist_labels(const char* filename, int* number_of_labels, int* num_classes) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Could not open %s\n", filename);
        exit(1);
    }

    uint32_t magic_number = 0;
    uint32_t num_labels = 0;

    if (fread(&magic_number, sizeof(uint32_t), 1, fp) != 1) {
        printf("Error reading magic number from %s\n", filename);
        fclose(fp);
        exit(1);
    }
    magic_number = swap_uint32(magic_number);

    if (magic_number != 2049) {
        printf("Invalid MNIST label file: %s\n", filename);
        fclose(fp);
        exit(1);
    }

    if (fread(&num_labels, sizeof(uint32_t), 1, fp) != 1) {
        printf("Error reading number of labels from %s\n", filename);
        fclose(fp);
        exit(1);
    }
    num_labels = swap_uint32(num_labels);

    *number_of_labels = num_labels;
    *num_classes = 10; // MNIST has 10 classes (digits 0-9)

    printf("Number of labels: %d\n", num_labels);

    float *labels = (float*)calloc(num_labels * (*num_classes), sizeof(float));
    if (!labels) {
        printf("Failed to allocate memory for labels\n");
        fclose(fp);
        exit(1);
    }

    for (int i = 0; i < num_labels; i++) {
        unsigned char temp_label = 0;
        if (fread(&temp_label, sizeof(unsigned char), 1, fp) != 1) {
            printf("Failed to read label data for label %d\n", i);
            free(labels);
            fclose(fp);
            exit(1);
        }
        labels[i * (*num_classes) + temp_label] = 1.0f; // One-hot encoding
    }

    fclose(fp);
    return labels;
}
