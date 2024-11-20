#ifndef MNIST_H
#define MNIST_H

int reverse_int(int i);
int read_mnist_images(const char *filename, float **data, int *num_images, int *image_size);
int read_mnist_labels(const char *filename, float **labels, int *num_labels);

#endif // MNIST_H
