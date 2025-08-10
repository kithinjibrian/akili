# Single-layer and Multi-layer perceptrons
A single layer perceptron (SLP) is a feed-forward network based on a threshold transfer function. SLP is the simplest type of artificial neural networks and can only classify linearly separable cases with a binary target (1, 0). [c], [d]

Because SLP is a linear classifier and if the cases are not linearly separable the learning process will never reach a point where all the cases are classified properly. The most famous example of the inability of perceptron to solve problems with linearly non-separable cases is the XOR problem.

A multi-layer perceptron (MLP) has the same structure of a single layer perceptron with one or more hidden layers. The backpropagation algorithm consists of two phases: the forward phase where the activations are propagated from the input to the output layer, and the backward phase, where the error between the observed actual and the requested nominal value in the output layer is propagated backwards in order to modify the weights and bias values.

# Perceptron
The perceptron is a basic function that mimics the human neuron. It receives n inputs, associated to the dendrites inputs to the neuron. Each dendrite, due to lernging, is weighted by a number that signals its input relevance for the neuron [1].

The perceptron wants to mimic it. Receiving a vector (i.e. array) xi of signals, where i stands for the i-th dendrites, it weights each of them by a vector of weights wi. It adds also a bias to remove near-zero issues (the bias shifts the decision boundary away from the origin and does not depend on any input value).

!["A perceptron"](https://miro.medium.com/max/2870/1*n6sJ4yZQzwKL9wnF5wnVNg.png "A perceptron")

```python
import numpy as np

np.random.seed(42)

def perceptron(X):
    # X is a 1D array of inputs, e.g., np.array([0.2, 0.5, 0.8])
    
    # Random weights in range [-1, 1]
    w = 2 * np.random.random(len(X)) - 1

    # Random bias in range [-1, 1]
    b = 2 * np.random.random() - 1

    # Weighted sum z = (i1​ × w1​) + ( i2 ​× w2​) + ⋯ + (iN ​× wN) + b
    z = np.dot(X, w) + b

    # Sigmoid activation
    output = 1 / (1 + np.exp(-z))

    return output

X = np.array([0.2, 0.5, 0.8])

print("Perceptron output:", perceptron(X))
```

# Neural Network's Layer(s)
1. An **Input Layer**, that pass the features to the NN
2. An arbitrary number of **Hidden Layers**, containing an arbitrary number of neurons for each layer, that receives the inputs and elaborate them. We will introduce Hidden Layers with ReLU activator, since in the hidden part of the NN we don't need the output to be contained in the [0,1]
range.
3. An **Output Layer**: these layers contains a number of neurons equal to the number of possible labels we want to have a prediction to; this is because the output of the NN is thus a vector whose dimension is the same as the cardinality of the set of labels, and its entries are the probability for each label for the element whose feateures we have passed to the NN. This means that we will use a sigmoid activator to the Output layer, so we squeeze each perceptron's output between 0 and 1. 

![](https://miro.medium.com/proxy/1*DW0Ccmj1hZ0OvSXi7Kz5MQ.jpeg)

```python
X = np.array([0.2, 0.5, 0.8])   # shape (3,)

W = np.array([
    [0.1, 0.4, -0.3],  # neuron 1 weights
    [-0.2, 0.5, 0.2],  # neuron 2 weights
    [0.7, -0.1, 0.5]   # neuron 3 weights
])  # shape (3, 3)

b = np.array([0.1, -0.3, 0.05])  # shape (3,)
```

# Step 1 — Multiply and sum for each neuron

`sum z = (i1​ × w1​) + ( i2 ​× w2​) + ⋯ + (iN ​× wN) + b`

## Neuron 1:
`(0.1)(0.2)+(0.4)(0.5)+(−0.3)(0.8)+0.1=0.02+0.20−0.24+0.1=0.08`

## Neuron 2:
`(−0.2)(0.2)+(0.5)(0.5)+(0.2)(0.8)−0.3=−0.04+0.25+0.16−0.3=0.07`

## Neuron 3:
`(0.7)(0.2)+(−0.1)(0.5)+(0.5)(0.8)+0.05=0.14−0.05+0.40+0.05=0.54`

# Step 2 — Put results into vector
`Z=[0.08,0.07,0.54]`

# Step 3 — Apply activation (sigmoid here):
`output=σ(Z)≈[0.5199,0.5175,0.6318]`

# NumPy does Step 1 automatically with:
```python
Z = np.dot(W, X) + b
```

`np.dot(W, X)` takes each row of `W` (each neuron’s weights) and dot-products it with `X`.

```python
# one dense layer
import numpy as np

np.random.seed(42)

def layer(X, num_neurons):
    num_inputs = len(X)
    
    # Random weights: shape (num_neurons, num_inputs)
    W = 2 * np.random.random((num_neurons, num_inputs)) - 1
    
    # Random biases: shape (num_neurons,)
    b = 2 * np.random.random(num_neurons) - 1
    
    # Weighted sum for all neurons at once
    Z = np.dot(W, X) + b
    
    # Sigmoid activation for all neurons
    output = 1 / (1 + np.exp(-Z))
    
    return output

X = np.array([0.2, 0.5, 0.8])

output = layer(X, num_neurons=3)

print("Layer output:", output)
```

# Multi layer percerptron

```python
import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dense_layer(X, num_neurons):
    num_inputs = X.shape[0]  # length of input vector
    W = 2 * np.random.random((num_neurons, num_inputs)) - 1
    b = 2 * np.random.random(num_neurons) - 1
    Z = np.dot(W, X) + b
    return sigmoid(Z)

X = np.array([0.2, 0.5, 0.8])

hidden_output = dense_layer(X, 4)

final_output = dense_layer(hidden_output, 2)

print("Input layer:", X)
print("Hidden layer output:", hidden_output)
print("Final output:", final_output)

# Input layer: [0.2 0.5 0.8]
# Hidden layer output: [0.80792851 0.19305997 0.42919006 0.43106147]
# Final output: [0.34682589 0.57890019]
```

```python
import numpy as np

# Random seed so results are repeatable
np.random.seed(42)

# Step 1: Input (batch_size=2, features=3)
X = np.array([
    [0.2, 0.5, 0.8],
    [0.1, 0.4, 0.7]
])  # Shape: (2, 3)

# Step 2: Weights and biases for first layer (3 -> 4)
W1 = np.random.randn(3, 4)  # Shape: (3, 4)
b1 = np.random.randn(4)     # Shape: (4,)

# Step 3: First layer computation
Z1 = np.dot(X, W1) + b1  # Shape: (2, 4)
A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation

# Step 4: Weights and biases for second layer (4 -> 2)
W2 = np.random.randn(4, 2)  # Shape: (4, 2)
b2 = np.random.randn(2)     # Shape: (2,)

# Step 5: Second layer computation
Z2 = np.dot(A1, W2) + b2  # Shape: (2, 2)
A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation

# Print results
print("Input X:\n", X)
print("W1:\n", W1)
print("b1:\n", b1)
print("Z1:\n", Z1)
print("A1:\n", A1)
print("W2:\n", W2)
print("b2:\n", b2)
print("Z2:\n", Z2)
print("A2:\n", A2)

"""
Input X:
 [[0.2 0.5 0.8]
 [0.1 0.4 0.7]]
W1:
 [[ 0.49671415 -0.1382643   0.64768854  1.52302986]
 [-0.23415337 -0.23413696  1.57921282  0.76743473]
 [-0.46947439  0.54256004 -0.46341769 -0.46572975]]
b1:
 [ 0.24196227 -1.91328024 -1.72491783 -0.56228753]
Z1:
 [[-0.15135109 -1.62395355 -1.17650787 -0.246548  ]
 [-0.13065973 -1.64096943 -1.35285624 -0.42902148]]
A1:
 [[0.46223429 0.16466035 0.23568067 0.43867334]
 [0.46738146 0.1623332  0.2054038  0.39436002]]
W2:
 [[-1.01283112  0.31424733]
 [-0.90802408 -1.4123037 ]
 [ 1.46564877 -0.2257763 ]
 [ 0.0675282  -1.42474819]]
b2:
 [-0.54438272  0.11092259]
Z2:
 [[-0.78701565 -0.65458209]
 [-0.8374834  -0.57970684]]
A2:
 [[0.31280983 0.34195772]
 [0.30206507 0.35900005]]
"""
```

```python
import numpy as np

np.random.seed(42)

class Layer:
    def __init__(self, n_inputs, n_neurons):
        # weights shape: (n_inputs, n_neurons)
        self.W = np.random.randn(n_inputs, n_neurons)
        self.b = np.zeros((1, n_neurons))

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.W) + self.b
        self.a = self._apply_activation(self.z)
        return self.a

    def backward(self, dL_da):
        da_dz = self.z > 0
        da_dz = da_dz.astype(float)
        dL_dz = dL_da * da_dz
        self.dL_dz = dL_dz

        # Compute gradients
        self.dW = self.input.T.dot(dL_dz)
        self.db = np.sum(dL_dz, axis=0, keepdims=True)

        # Gradient for previous layer input
        dL_dinput = dL_dz.dot(self.W.T)

        return dL_dinput

    def _apply_activation(self, z):
        return np.maximum(0, z)

# Input data (2 samples, 3 features)
X = np.array([[0.2, 0.5, 0.8],
              [0.1, 0.4, 0.7]])

# Layers
layer1 = DenseLayer(3, 4, activation="relu")
layer2 = DenseLayer(4, 2, activation="sigmoid")

# Forward pass
out1 = layer1.forward(X)
out2 = layer2.forward(out1)

print(out2)
```

# Step 0: Our made-up weights/biases
```
W1​=[0.10.3​0.20.4​],b1​=[0.01​0.02​]
W2=[0.50.6],b2=0.03
W2​=[0.50.6​]
```

# Step 1: Forward pass for one input
Let’s use input:

`x=[1.0,2.0],ytrue​=1.0`

Hidden layer (z-values)

`z1​=x⋅W1​+b1`


# Sources
[simple nn with python multi layer perceptron](https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron#How-to-build-a-simple-Neural-Network-with-Python:-Multi-layer-Perceptron)