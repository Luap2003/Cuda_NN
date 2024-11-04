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
        print(f"Epoch {self.epoch+1}/{self.num_epochs}, Loss: {loss:.4f}")

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
    
    def train(self, learning_rate, x_train, y_train):
        for i in range(self.num_epochs):
            self.back_prop(x_train, y_train, learning_rate)
            self.epoch=i
    
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