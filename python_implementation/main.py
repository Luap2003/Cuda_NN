import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from layer import Layer

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
    learning_rate = 0.1

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

    # Initialize layers with random weights
    layer1 = Layer(m_train, n_input, n_1, akt_func=relu)
    layer2 = Layer(m_train, n_1, n_2, akt_func=softmax)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        layer1.forward_prog(x_train)
        layer2.forward_prog(layer1.A)

        # Compute loss (cross-entropy loss)
        loss = -np.sum(y_train_encoded * np.log(layer2.A + 1e-8)) / m_train
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        # Backward pass
        # Output layer
        layer2.backword_prog_output(y_train_encoded, layer1.A)
        # Hidden layer
        layer1.backward_prog(layer2.w, layer2.dZ, x_train)

        # Update weights and biases
        layer1.update(learning_rate)
        layer2.update(learning_rate)

    # Evaluate on test set
    # Forward pass
    layer1.m = m_test  # Update the number of examples for test data
    layer2.m = m_test
    layer1.forward_prog(x_test)
    layer2.forward_prog(layer1.A)

    # Predictions
    predictions = np.argmax(layer2.A, axis=0)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
