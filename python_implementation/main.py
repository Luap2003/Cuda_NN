import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from layer import Layer
from nn import NeuralNetwork
def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten images and normalize pixel values
    x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0  # Shape: (784, 60000)
    x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0     # Shape: (784, 10000)

    # One-hot encode the labels
    num_classes = 10
    m_train = x_train.shape[1]
    m_test = x_test.shape[1]

    y_train_encoded = np.zeros((num_classes, m_train))
    y_train_encoded[y_train, np.arange(m_train)] = 1

    y_test_encoded = np.zeros((num_classes, m_test))
    y_test_encoded[y_test, np.arange(m_test)] = 1

    # Set hyperparameters
    num_epochs = 1000
    learning_rate = 0.3

    # Initialize layers
    n_input = 784   # Input layer size
    n_1 = 64   
    n_2 = 10   

    # Activation functions
    def relu(Z):
        return np.maximum(0, Z)

    def softmax(Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability improvement
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    NN = NeuralNetwork(m_train, num_epochs)
    NN.add_layer(n_input, 64, relu)
    NN.add_layer(64, n_2, softmax)

    NN.train(learning_rate, x_train, y_train_encoded)

    NN.test(m_test, x_test, y_test_encoded)
# Visualization
    # Select 100 random test samples
    num_images = 100
    indices = np.random.choice(m_test, num_images, replace=False)
    sample_images = x_test[:, indices]  # Shape: (784, 100)
    sample_true_labels = y_test[indices]  # Shape: (100,)

    # Get predictions for the sample images
    input_data = sample_images
    for layer in NN.layers:
        layer.forward_prog(input_data)
        input_data = layer.A
    sample_predictions = np.argmax(NN.layers[-1].A, axis=0)  # Shape: (100,)

    # Plot the images with predicted and true labels
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(num_images):
        img = sample_images[:, i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Pred: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}', fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.savefig("temp.png")

if __name__ == "__main__":
    main()