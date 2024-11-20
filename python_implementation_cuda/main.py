import numpy as np
import cupy as cp
import time
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from layer import Layer
from nn import NeuralNetwork

# Define activation functions
def relu(Z, xp):
    return xp.maximum(0, Z)

def softmax(Z, xp):
    expZ = xp.exp(Z - xp.max(Z, axis=0, keepdims=True))  # Stability improvement
    return expZ / xp.sum(expZ, axis=0, keepdims=True)

# Load and preprocess data
def load_and_preprocess_data(xp, dtype):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = xp.array(x_train, dtype=dtype), xp.array(x_test, dtype=dtype)
    y_train, y_test = xp.array(y_train), xp.array(y_test)

    x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0

    # One-hot encode the labels
    num_classes = 10
    m_train = x_train.shape[1]
    m_test = x_test.shape[1]

    y_train_encoded = xp.zeros((num_classes, m_train), dtype=dtype)
    y_train_encoded[y_train, xp.arange(m_train)] = 1

    y_test_encoded = xp.zeros((num_classes, m_test), dtype=dtype)
    y_test_encoded[y_test, xp.arange(m_test)] = 1

    return x_train, y_train_encoded, x_test, y_test_encoded, y_test, m_train, m_test

# Train and test the neural network
def train_and_evaluate(NN, x_train, y_train, x_test, y_test, m_test, xp, dtype, epochs=100, lr=0.3, batch_size=128):

    NN.train(lr, x_train, y_train, batch_size)
    accuracy = NN.test(m_test, x_test, y_test)
    return accuracy

# Visualize sample predictions
def visualize_predictions(NN, x_test, y_test, xp):
    num_images = 100
    indices = xp.random.choice(x_test.shape[1], num_images, replace=False)
    sample_images = x_test[:, indices]
    sample_true_labels = y_test[indices]

    sample_images_cpu = cp.asnumpy(sample_images) if xp == cp else sample_images
    sample_true_labels_cpu = cp.asnumpy(sample_true_labels) if xp == cp else sample_true_labels

    # Get predictions
    input_data = sample_images
    for layer in NN.layers:
        layer.forward_prog(input_data)
        input_data = layer.A
    sample_predictions = xp.argmax(NN.layers[-1].A, axis=0)
    sample_predictions_cpu = cp.asnumpy(sample_predictions) if xp == cp else sample_predictions

    # Plot images with predicted and true labels
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(num_images):
        img = sample_images_cpu[:, i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Pred: {sample_predictions_cpu[i]}\nTrue: {sample_true_labels_cpu[i]}', fontsize=8)

    plt.tight_layout()
    plt.show()

# Benchmark function
def benchmark(run_fn, *args, **kwargs):
    start = time.time()
    accuracy = run_fn(*args, **kwargs)
    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")
    return accuracy

# Main function to run the whole pipeline
def main():
    xp = cp  # Select CuPy for GPU acceleration or np for CPU
    dtype = cp.float32 if xp == cp else np.float32  # Choose floating-point precision

    epochs = 20
    batch_size = 64
    learning_rate = 0.1

    # Load and preprocess data
    x_train, y_train_encoded, x_test, y_test_encoded, y_test, m_train, m_test = benchmark(
        load_and_preprocess_data, xp, dtype
    )

    n_input, n_hidden, n_output = 784, 64, 10
    NN = NeuralNetwork(m_train, epochs, xp=xp, dtype=dtype)
    NN.add_layer(n_input, n_hidden, lambda Z: relu(Z, xp))
    NN.add_layer(n_hidden, n_output, lambda Z: softmax(Z, xp))

    # Train and evaluate the neural network
    accuracy = benchmark(
        train_and_evaluate, NN, x_train, y_train_encoded, x_test, y_test_encoded, m_test, xp, dtype, epochs=epochs, lr=learning_rate, batch_size=batch_size
    )
    print(f"Test Accuracy: {accuracy:.4f}")

    # Visualize predictions
    visualize_predictions(NN, x_test, y_test, xp)

if __name__ == "__main__":
    main()
