from layer import Layer
import numpy as np
import cupy as cp
from tqdm import tqdm
import time
class NeuralNetwork:
    def __init__(self, m, num_epochs, xp=cp, dtype=cp.float32):
        self.layers = []
        self.m = m  # This will be updated during training
        self.num_epochs = num_epochs
        self.epoch = 0
        self.xp = xp
        self.dtype = dtype
        self.losses = []

    def add_layer(self, input_size, output_size, akt_func):
        self.layers.append(Layer(self.m, input_size, output_size, akt_func, xp=self.xp, dtype=self.dtype))

    def back_prop(self, x_batch, y_batch, learning_rate):
        # Forward pass
        activations = [x_batch]
        input = x_batch
        for layer in self.layers:
            layer.forward_prog(input)
            input = layer.A
            activations.append(input)

        # Loss calculation
        loss = -self.xp.sum(y_batch * self.xp.log(self.layers[-1].A + 1e-8)) / self.m
        self.losses.append(loss)
        #print(f"Epoch {self.epoch + 1}/{self.num_epochs}, Loss: {loss:.4f}")

        # Backward pass for the output layer
        self.layers[-1].backword_prog_output(y_batch, activations[-2])

        # Backward pass for hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].backward_prog(
                self.layers[i + 1].w,
                self.layers[i + 1].dZ,
                activations[i]
            )

        # Update weights and biases
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, learning_rate, x_train, y_train, batch_size):
        num_samples = x_train.shape[1]
        num_batches = int(self.xp.ceil(num_samples / batch_size))
        with tqdm(total=self.num_epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(self.num_epochs):
                # Shuffle the data
                permutation = self.xp.random.permutation(num_samples)
                x_train_shuffled = x_train[:, permutation]
                y_train_shuffled = y_train[:, permutation]
                batch_times = []
                for batch in range(num_batches):
                    batch_start_time = time.time()
                    start = batch * batch_size
                    end = min(start + batch_size, num_samples)
                    x_batch = x_train_shuffled[:, start:end]
                    y_batch = y_train_shuffled[:, start:end]
                    current_batch_size = end - start
                    # Update self.m and layers' m to current_batch_size
                    self.m = current_batch_size
                    for layer in self.layers:
                        layer.m = current_batch_size
                    self.back_prop(x_batch, y_batch, learning_rate)
                    batch_end_time = time.time()
                    batch_times.append(batch_end_time - batch_start_time)
                # Calculate average batch time
                avg_batch_time = sum(batch_times) / len(batch_times)
                batches_per_second = 1 / avg_batch_time if avg_batch_time > 0 else float('inf')
                # Update progress bar description
                pbar.set_description(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {self.losses[-1]:.4f} - {batches_per_second:.2f} batches/s")
                pbar.update(1)
            self.epoch = epoch

    def test(self, m_test, x_test, y_test):
        # Update the number of examples
        self.m = m_test
        for layer in self.layers:
            layer.m = m_test

        # Forward pass through the network
        input = x_test
        for layer in self.layers:
            layer.forward_prog(input)
            input = layer.A

        # Compute accuracy
        predictions = self.xp.argmax(self.layers[-1].A, axis=0)
        true_labels = self.xp.argmax(y_test, axis=0)
        accuracy = self.xp.mean(predictions == true_labels)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy
