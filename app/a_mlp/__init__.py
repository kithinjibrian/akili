from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt

import numpy as np
from .layer import Layer

from .optims import (
    Adam,
    AdamW,
    RMSprop,
    SGD,
    Optimizer
)

class NeuralNetwork:
    def __init__(
        self, 
        input_size: int,
        optimizer: Optimizer = Adam(learning_rate=0.001)
    ):
        self.input_size: int = input_size
        self.layers: List[Layer] = []
        self.loss_history: List[Union[float, np.floating[Any]]] = []
        self.accuracy_history: List[Union[float, np.floating[Any]]] = []

        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

        self.optimizer: Optimizer = optimizer

    def add_layer(
        self, 
        n_neurons: int, 
        activation='relu', 
        optimizer: Optional[Optimizer] = None
    ) -> 'NeuralNetwork':
        if len(self.layers) == 0:
            # First layer uses input_size
            n_inputs = self.input_size
        else:
            n_inputs = self.layers[-1].W.shape[1]
        
        layer = Layer(n_inputs, n_neurons, activation)

        if optimizer is not None:
            layer.optimizer = optimizer
        else:
            layer.optimizer = self.optimizer

        layer.optimizer.initialize(layer)

        self.layers.append(layer)
        return self
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        current_input = X
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
        
        self.output = current_input
        return self.output

    def backward(self, y_true: np.ndarray) -> None:
        if self.output is None:
            raise ValueError("Forward pass must be called before backward pass")
            
        if self.layers[-1].activation == 'softmax':
            # Categorical cross-entropy with softmax derivative
            # For one-hot encoded labels: dL/dz = y_pred - y_true
            dL_dz = self.output - y_true

            self.layers[-1].dL_dz = dL_dz
            if self.layers[-1].input is not None:
                self.layers[-1].dW = self.layers[-1].input.T.dot(dL_dz)
                self.layers[-1].db = np.sum(dL_dz, axis=0, keepdims=True)

            current_gradient = dL_dz.dot(self.layers[-1].W.T)

            for layer in reversed(self.layers[:-1]):
                current_gradient = layer.backward(current_gradient)
        else:
            if self.layers[-1].activation == 'sigmoid':
                # Binary cross-entropy loss derivative
                dL_da = -(y_true / (self.output + 1e-15) - (1 - y_true) / (1 - self.output + 1e-15))
            else:
                # Mean squared error loss derivative
                dL_da = 2 * (self.output - y_true) / y_true.shape[0]
            
            current_gradient = dL_da
            for layer in reversed(self.layers):
                current_gradient = layer.backward(current_gradient)

    def update_weights(self) -> None:
            """Update weights using each layer's optimizer"""
            for layer in self.layers:
                if layer.dW is None or layer.db is None:
                    raise ValueError("Backward pass must be called before updating weights")
                if layer.optimizer is None:
                    raise ValueError("Layer optimizer not initialized")
                layer.optimizer.update(layer)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
        if self.layers[-1].activation == 'sigmoid':
            # Binary cross-entropy loss
            epsilon = 1e-15  # Prevent log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.layers[-1].activation == 'softmax':
            # Categorical cross-entropy loss
            epsilon = 1e-15  # Prevent log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            # Mean squared error loss
            return np.mean((y_true - y_pred)**2)
        
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
        if self.layers[-1].activation == 'sigmoid':
            # Binary classification
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions == y_true)
        elif self.layers[-1].activation == 'softmax':
            # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == true_labels)
        else:
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            return 1 - (ss_res / (ss_tot + 1e-15))

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              learning_rate: float = 0.01, batch_size: Optional[int] = None, 
              verbose: bool = True) -> None:
        n_samples = X.shape[0]
        
        if batch_size is None:
            batch_size = n_samples
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                # Forward pass
                y_pred = self.forward(batch_X)
                
                # Backward pass
                self.backward(batch_y)
                
                # Update weights
                self.update_weights()
                
                # Accumulate metrics
                batch_loss = self.compute_loss(batch_y, y_pred)
                batch_accuracy = self.compute_accuracy(batch_y, y_pred)
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                n_batches += 1
            
            # Average metrics over batches
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_accuracy)
            
            if verbose and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch:4d}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def plot_training_history(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()