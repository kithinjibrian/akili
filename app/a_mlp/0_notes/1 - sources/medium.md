# What is a Multi-Layer Perceptron?

An MLP is a type of neural network consisting of at least three layers: an input layer, one or more hidden layers, and an output layer. Each neuron in one layer is connected to every neuron in the next layer. The MLP learns by adjusting the weights of these connections to minimize the error of its predictions, using a process called backprop.

![Backprop image](https://miro.medium.com/v2/resize:fit:720/format:webp/1*p37WvwF_4vZ4K5Ls5n6x4A.png "Neural network layers")

# Key Components of MLP
1. Layers and nodes
    - Input layer: Receives input data
    - Hidden layers: Perform intermediate computations
    - Output layer: Produces the final prediction

2. Activation functions
    - sigmoid: Is a mathematical function that maps any real-valued number to a value between 0 and 1, producing an S-shaped curve. Useful for binary classification.
        ![Sigmoid function](https://miro.medium.com/v2/resize:fit:720/format:webp/1*takIKOwtkHy6RgXplzC4lw.png "Sigmoid function")

    - ReLU:
        ![ReLU](https://miro.medium.com/v2/resize:fit:720/format:webp/1*B5uj_q2OTGzTjqADesE0vw.png "ReLU")

    - Softmax:
        ![Softmax](https://miro.medium.com/v2/resize:fit:640/format:webp/1*HSVJDdOh08MH53sngkqJeQ.png)

3. Loss function:
    - Cross-Entropy loss: Measures the perfomance of a classification model.

4. Optimization:
    - Gradient descent: Updates weights to minimize the loss function.
    - Backpropagation: Computes gradients of the loss function with respect to each weight.

# Step-by-Step Implementation

## Step 1: Initialize Network Parameters

```python
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Randomly initialize the weights and biases of all layers. The MLP has one hidden layer for now.

        Args:
            input_size - The number of neurons in the input layer
            hidden_size - Number of neurons in the hidden layer.
            output_size - Number of neurons in the output layer.
        """
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        """
        σ(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        """
        s(xi) = e^xi / (E e^xj)
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
```

## Step 2: Forward Propagation

```python
def forward(self, X):
    self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
    self.hidden_output = self.sigmoid(self.hidden_input)
    self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
    self.final_output = self.softmax(self.final_input)
    return self.final_output
```

## Step 3: Backward Propagation
```python
def backward(self, X, y, output, learning_rate):
    output_error = output - y
    hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (1 - self.hidden_output)
    
    self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
    self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
    self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)
    self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)
```

![Backward prop](https://miro.medium.com/v2/resize:fit:640/format:webp/1*kWPgeDnujt6eYpoSIbG4mA.png "Backward propagation for updating weights and biases.")

## Step 4: Training the Model
```python
def train(self, X, y, epochs, learning_rate):
    for epoch in range(epochs):
        output = self.forward(X)
        self.backward(X, y, output, learning_rate)
        if (epoch+1) % 100 == 0:
            loss = -np.sum(y * np.log(output)) / X.shape[0]
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
```

## Step 5: Predicting
```python
def predict(self, X):
    output = self.forward(X)
    return np.argmax(output, axis=1)
```

# sources
[building a multi layer perceptron from scratch with nump](https://elcaiseri.medium.com/building-a-multi-layer-perceptron-from-scratch-with-numpy-e4cee82ab06d)