#ifndef MNIST_H
#define MNIST_H

/**
 * @brief Reverses int from big-endian to little-endian
 *
 * @param i Integer to be reversed.
 *
 * @return Reversed integer.
 */
int reverse_int(int i);

/**
 * @brief Reads the mnist images from the dataset.
 *
 * @param filename Path to files to be read.
 * @param data Double Pointer to where the data should be stored.
 * @param num_images Pointer to the number of images.
 * @param image_size Pointer to the image size.
 */
int read_mnist_images(const char *filename, float **data, int *num_images, int *image_size);

/**
 * @brief Reads the mnist labels from the dataset.
 *
 * @param filename Path to files to be read.
 * @param labels Double Pointer to where the labels should be stored.
 * @param num_labels Pointer to the number of labels.
 */
int read_mnist_labels(const char *filename, float **labels, int *num_labels);

#endif // MNIST_H
