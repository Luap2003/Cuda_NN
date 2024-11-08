#ifndef DATA_LOADER_H
#define DATA_LOADER_H

float* load_mnist_images(const char* filename, int* number_of_images, int* image_size);
float* load_mnist_labels(const char* filename, int* number_of_labels, int* num_classes);

#endif // DATA_LOADER_H
