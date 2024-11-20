# nn.py
from layer import Layer
import numpy as np

class NeuralNetwork:
    
    def __init__(self, m, num_epochs):
        self.layers = []
        self.m = m
        self.num_epochs = num_epochs
        self.epoch = 0
        
    def add_layer(self, input_size, output_size, akt_func):
        self.layers.append(Layer(self.m, input_size, output_size, akt_func))
    
    def back_prop(self, x_train: np.array, y_train_encoded, learning_rate):
        # Forward pass
        activations = [x_train]  # List to store activations of each layer
        input = x_train
        for layer in self.layers:
            layer.forward_prog(input)
            input = layer.A
            activations.append(input)
        
        loss = -np.sum(y_train_encoded * np.log(self.layers[-1].A + 1e-8)) / self.m

        # Backward pass
        self.layers[-1].backword_prog_output(y_train_encoded, activations[-2])
        
        # Hidden layers
        for i in range(len(self.layers)-2, -1, -1):
            self.layers[i].backward_prog(
                self.layers[i+1].w,
                self.layers[i+1].dZ,
                activations[i]
            )
        
        # Update weights and biases
        for layer in self.layers:
            layer.update(learning_rate)
        
        return loss  # Return loss to accumulate and print per epoch
    
    def train(self, learning_rate, x_train, y_train, batch_size):
        num_examples = x_train.shape[1]
        for i in range(self.num_epochs):
            # Shuffle the data
            permutation = np.random.permutation(num_examples)
            x_train_shuffled = x_train[:, permutation]
            y_train_shuffled = y_train[:, permutation]
            
            total_loss = 0
            num_batches = 0
            
            # Loop over mini-batches
            for j in range(0, num_examples, batch_size):
                x_batch = x_train_shuffled[:, j:j+batch_size]
                y_batch = y_train_shuffled[:, j:j+batch_size]
                
                # Update self.m and layers' m
                batch_m = x_batch.shape[1]
                self.m = batch_m
                for layer in self.layers:
                    layer.m = batch_m
                
                # Perform backpropagation on the batch
                loss = self.back_prop(x_batch, y_batch, learning_rate)
                total_loss += loss
                num_batches += 1
            
            average_loss = total_loss / num_batches
            print(f"Epoch {i+1}/{self.num_epochs}, Loss: {average_loss:.4f}")
            self.epoch = i
    
    def test(self, m_test, x_test, y_test):
        # Update the number of examples
        self.m = m_test
        for layer in self.layers:
            layer.m = m_test

        # Forward pass
        input = x_test
        for layer in self.layers:
            layer.forward_prog(input)
            input = layer.A

        # Predictions
        predictions = np.argmax(self.layers[-1].A, axis=0)
        true_labels = np.argmax(y_test, axis=0)
        accuracy = np.mean(predictions == true_labels)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
