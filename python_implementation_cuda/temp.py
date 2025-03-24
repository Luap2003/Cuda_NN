import tensorflow as tf
from tensorflow import keras
import os
import time
import sys
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Neural Network benchmark for MNIST dataset")
parser.add_argument('batch_size', type=int, nargs='?', default=16, 
                    help='Batch size for training')
parser.add_argument('epochs', type=int, nargs='?', default=100, 
                    help='Number of epochs for training')
parser.add_argument('hidden_layers', type=int, nargs='*', 
                    help='Sizes of hidden layers')
args = parser.parse_args()

# Parse parameters
batch_size = args.batch_size
epochs = args.epochs
hidden_layers = args.hidden_layers if args.hidden_layers else [256, 128]

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

# Configure GPU settings
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Convert labels to one-hot vectors
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the model with dynamic architecture
model_layers = [keras.layers.Flatten(input_shape=(28, 28, 1))]
for layer_size in hidden_layers:
    model_layers.append(keras.layers.Dense(layer_size, activation="relu"))
model_layers.append(keras.layers.Dense(num_classes, activation="softmax"))

model = keras.Sequential(model_layers)

model.compile(
    loss="categorical_crossentropy",
    optimizer="SGD",
    metrics=["accuracy"],
    jit_compile=True
)

# Train the model and measure time
start_time = time.time()
model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs)
training_time = time.time() - start_time

# Evaluate on the test set
_, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# Output in format compatible with the benchmark script
print(f"Training time: {training_time:.2f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
