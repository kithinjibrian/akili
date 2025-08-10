# What is a Multi-Layer Perceptron (MLP)?

A Multi-Layer Perceptron is a feedforward neural network made of an input layer, one or more fully connected hidden layers, and an output layer. Each layer computes a linear transform followed by a nonlinearity. The network learns by adjusting weights and biases to minimize a loss function, using gradient-based optimization computed with backpropagation.

Intuitively:

* Input features are combined linearly by weights and biases.
* Nonlinear activations let the network represent complex functions.
* Backpropagation gives gradients of the loss with respect to every parameter so we can update them using gradient descent.

Below I state notation, derive forward and backward passes with equations, explain the special-case softmax + cross-entropy simplification, show shapes, give corrected code, and list practical tips.

# Notation and shapes

Let the network have L layers of learnable parameters (one output layer included). For convenience we use the "row-as-example" convention used in your code:

* $m$ - number of examples in a minibatch.
* $n^{[0]}$ - input dimension (features).
* $n^{[l]}$ - number of neurons in layer $l$ (so $n^{[L]}$ is number of outputs / classes).
* Input matrix $X \in \mathbb{R}^{m \times n^{[0]}}$. Row $i$ is example $i$.
* Weight matrix for layer $l$: $W^{[l]} \in \mathbb{R}^{n^{[l-1]} \times n^{[l]}}$.
* Bias row vector for layer $l$: $b^{[l]} \in \mathbb{R}^{1 \times n^{[l]}}$.
* Pre-activation at layer $l$: $Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$. So $Z^{[l]} \in \mathbb{R}^{m \times n^{[l]}}$.
* Activation at layer $l$: $A^{[l]} = \phi^{[l]}(Z^{[l]}) \in \mathbb{R}^{m \times n^{[l]}}$.
* By convention $A^{[0]} = X$. Network output (probabilities) $\hat{Y} = A^{[L]}$.

This notation matches your code where you compute `np.dot(X, W)` and add bias (broadcasted).

# Forward propagation - equations

For each layer $l = 1,\dots,L$:

$$
Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]} \quad\text{(shape \(m\times n^{[l]}\))}
$$

$$
A^{[l]} = \phi^{[l]}(Z^{[l]}) \quad\text{(elementwise activation)}
$$

Common activations:

* Sigmoid: $\sigma(z) = \dfrac{1}{1 + e^{-z}}$.
* ReLU: $\mathrm{ReLU}(z) = \max(0,z)$.
* Softmax (usually for final layer in multiclass classification):

  $$
  \mathrm{softmax}(z)_j = \frac{e^{z_j}}{\sum_{k} e^{z_k}}\quad\text{(apply per row/example)}
  $$

  Use the log-sum-exp trick for numerical stability:

  $$
  \mathrm{softmax}(z)_j = \frac{e^{z_j - \max_k z_k}}{\sum_{k} e^{z_k - \max_k z_k}} .
  $$

# Loss function - cross-entropy (multiclass)

For one example with one-hot label vector $y$ and predicted probabilities $\hat{y}$:

$$
\ell(\hat{y}, y) = -\sum_{k} y_k \log \hat{y}_k .
$$

For a minibatch of $m$ examples:

$$
\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \ell(\hat{y}^{(i)}, y^{(i)}) \;=\; -\frac{1}{m} \sum_{i=1}^{m} \sum_{k} y^{(i)}_k \log \hat{y}^{(i)}_k .
$$

# Key derivative: softmax plus cross-entropy

The combination softmax + cross-entropy produces a major simplification. For a single example:

* Let $\hat{y} = \mathrm{softmax}(z)$.
* Loss $\ell = -\sum_k y_k \log \hat{y}_k$.

Then

$$
\frac{\partial \ell}{\partial z_j} = \hat{y}_j - y_j .
$$

Derivation sketch:

$$
\frac{\partial \ell}{\partial z_j}
= -\sum_k y_k \frac{1}{\hat{y}_k} \frac{\partial \hat{y}_k}{\partial z_j}
= -\sum_k y_k (\delta_{kj} - \hat{y}_j)
= \hat{y}_j - y_j ,
$$

where we used $\partial \hat{y}_k/\partial z_j = \hat{y}_k(\delta_{kj} - \hat{y}_j)$. For a batch, stack rows and divide by $m$ where appropriate. This is why in practice `output_error = output - y` is correct when `output` is softmax probabilities and `y` is one-hot.

# Backpropagation - vectorized equations

Define the error term at layer $l$ as:

$$
\delta^{[l]} \;=\; \frac{\partial \mathcal{L}}{\partial Z^{[l]}} \quad\text{(shape \(m \times n^{[l]}\))}
$$

Final layer (softmax + cross-entropy):

$$
\delta^{[L]} = A^{[L]} - Y \quad\text{(rowwise, averaged later by dividing by \(m\) or including \(1/m\) in gradients)}
$$

For earlier layers $l < L$:

$$
\delta^{[l]} = \big(\delta^{[l+1]} (W^{[l+1]})^\top\big) \odot \phi'^{[l]}(Z^{[l]})
$$

where $\odot$ is elementwise product and $\phi'^{[l]}$ is derivative of the activation applied elementwise.

Gradients of parameters:

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} (A^{[l-1]})^\top \delta^{[l]} \quad\text{(shape \(n^{[l-1]} \times n^{[l]}\))}
$$

$$
\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{[l]}_{i,\cdot} \quad \text{(row vector of length \(n^{[l]}\))}
$$

Update step (vanilla gradient descent):

$$
W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial W^{[l]}}
$$

$$
b^{[l]} \leftarrow b^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial b^{[l]}}
$$

where $\eta$ is the learning rate.

# Activation derivatives

* Sigmoid: if $s=\sigma(z)$, then $\sigma'(z) = s(1-s)$.
* ReLU: $\mathrm{ReLU}'(z) = 1$ if $z>0$, else $0$. Use subgradient at zero or define it as 0 or 1 consistently.

# Shape bookkeeping example

Suppose $m=32$, input dim $n^{[0]}=100$, one hidden layer of $n^{[1]}=50$, output classes $n^{[2]}=10$.

Shapes:

* $X$: (32, 100)
* $W^{[1]}$: (100, 50)
* $b^{[1]}$: (1, 50)
* $Z^{[1]}=X W^{[1]} + b^{[1]}$: (32, 50)
* $A^{[1]}$: (32, 50)
* $W^{[2]}$: (50, 10)
* $Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}$: (32, 10)
* $A^{[2]} = \mathrm{softmax}(Z^{[2]})$: (32, 10)
* $\delta^{[2]} = A^{[2]} - Y$: (32, 10)
* $\partial L /\partial W^{[2]} = (A^{[1]})^\top \delta^{[2]} / m$: (50, 10)

# Practical improvements and numerical issues

1. Softmax numerical stability - always subtract rowwise max before exponentiating:

   $$
   \tilde{z} = z - \max_j z_j, \quad \hat{y} = \frac{e^{\tilde{z}}}{\sum_j e^{\tilde{z}_j}} .
   $$

2. Initialize weights properly:

   * Xavier/Glorot (good for sigmoid/tanh): $\mathrm{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$.
   * He initialization for ReLU: $\mathrm{Var}(W) = \frac{2}{n_{\text{in}}}$.

3. Use minibatches, not full-batch unless dataset small. Average gradients over batch by dividing by $m$.

4. Use advanced optimizers like Adam or momentum for faster convergence.

5. Regularization: L2 adds $\lambda W$ to gradient, dropout randomly drops activations during training.

6. Watch for vanishing gradients with sigmoid/tanh in deep nets. ReLU variants help.

7. Use gradient clipping if gradients explode.

# Corrected, stable, vectorized implementation (one hidden layer)

Below is a corrected, clear implementation that:

* Uses numerically stable softmax.
* Averages gradients over batch (`/ m`).
* Uses Xavier/He style weight scaling as a suggestion.
* Comments map to math.

```python
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, init='xavier'):
        # Weight shapes: (n_in, n_out) to match X @ W
        if init == 'xavier':
            scale1 = np.sqrt(2.0 / (input_size + hidden_size))
            scale2 = np.sqrt(2.0 / (hidden_size + output_size))
        elif init == 'he':
            scale1 = np.sqrt(2.0 / input_size)
            scale2 = np.sqrt(2.0 / hidden_size)
        else:
            scale1 = scale2 = 0.01

        self.W1 = np.random.randn(input_size, hidden_size) * scale1   # W^{[1]}
        self.b1 = np.zeros((1, hidden_size))                          # b^{[1]}
        self.W2 = np.random.randn(hidden_size, output_size) * scale2  # W^{[2]}
        self.b2 = np.zeros((1, output_size))                          # b^{[2]}

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime_from_activation(self, a):
        # derivative using activation value a = sigma(z)
        return a * (1.0 - a)

    def softmax(self, z):
        # z shape (m, n_classes)
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)               # numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # Forward pass
        self.A0 = X                              # A^{[0]}
        self.Z1 = self.A0.dot(self.W1) + self.b1 # Z^{[1]} = A0 W1 + b1
        self.A1 = self.sigmoid(self.Z1)          # A^{[1]} = sigmoid(Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2 # Z^{[2]} = A1 W2 + b2
        self.A2 = self.softmax(self.Z2)          # A^{[2]} = softmax(Z2)
        return self.A2

    def backward(self, Y, learning_rate=1e-2):
        # Y shape (m, n_classes), one-hot
        m = Y.shape[0]

        # delta at output: delta^{[2]} = A2 - Y (derived from softmax + cross-entropy)
        delta2 = (self.A2 - Y) / m               # shape (m, n_out)

        # Gradients for W2, b2
        dW2 = self.A1.T.dot(delta2)              # (n_hidden, n_out)
        db2 = np.sum(delta2, axis=0, keepdims=True)  # (1, n_out)

        # Backpropagate to hidden layer
        # delta1 = (delta2 @ W2^T) * phi'(Z1)
        delta1 = delta2.dot(self.W2.T) * self.sigmoid_prime_from_activation(self.A1)  # (m, n_hidden)

        dW1 = self.A0.T.dot(delta1)              # (n_in, n_hidden)
        db1 = np.sum(delta1, axis=0, keepdims=True)  # (1, n_hidden)

        # Gradient descent parameter update
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def compute_loss(self, Y):
        # Cross-entropy loss, averaged over batch
        m = Y.shape[0]
        # Clip to avoid log(0)
        eps = 1e-12
        A2 = np.clip(self.A2, eps, 1.0 - eps)
        return -np.sum(Y * np.log(A2)) / m

    def train(self, X, Y, epochs=1000, learning_rate=1e-2, verbose=True):
        for epoch in range(1, epochs + 1):
            self.forward(X)
            self.backward(Y, learning_rate)
            if verbose and (epoch % 100 == 0 or epoch == 1):
                loss = self.compute_loss(Y)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
```

Key differences vs your original code:

* Softmax is numerically stabilized with `z - max`.
* Gradients are divided by `m` so they are properly averaged. This makes the effective learning rate behave consistently across batch sizes.
* Weight initialization uses scale based on layer sizes.
* `compute_loss` clips probabilities to avoid `log(0)`.

# Why `output - y` works (again, short)

Because for softmax plus cross-entropy:

$$
\frac{\partial \ell}{\partial z_j} = \hat{y}_j - y_j .
$$

So the network's output-error (the gradient of the loss with respect to logits) is simply `output - y`. This is the `delta` that starts backprop.

# Additional tips and common pitfalls

* If classes are not one-hot, convert them (or use cross-entropy that accepts integer class labels).
* For binary classification you can use a single sigmoid output with binary cross-entropy instead of softmax.
* Sigmoid outputs can saturate, causing vanishing gradients for deep nets. Prefer ReLU or variants for hidden layers.
* If training is unstable try lowering learning rate or using Adam.
* Regularize with weight decay (L2) or dropout if you overfit.
* Check gradient correctness with numerical gradient checking on a small model and small data.

# Summary - the math you need to remember

Forward, per layer:

$$
Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]},\quad A^{[l]}=\phi^{[l]}(Z^{[l]})
$$

Softmax + cross-entropy error at output:

$$
\delta^{[L]} = A^{[L]} - Y
$$

Backprop recursion:

$$
\delta^{[l]} = (\delta^{[l+1]} (W^{[l+1]})^\top) \odot \phi'^{[l]}(Z^{[l]})
$$

Parameter gradients:

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} (A^{[l-1]})^\top \delta^{[l]},\quad
\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^m \delta^{[l]}_{i,\cdot}
$$

Update:

$$
W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial W^{[l]}}
$$

---

# BREAK

Alright, let’s slow things down and go step-by-step on **initialization** for a Multi-Layer Perceptron.

---

## 1. Why Initialization Matters

When we build an MLP, each connection between neurons has a **weight** and each neuron has a **bias**. At the start of training, we must give these parameters some initial values.
The way we choose these initial values affects:

* **How fast training starts** (too small → almost no learning, too large → unstable gradients).
* **Whether gradients vanish or explode** as we backpropagate.
* **Symmetry breaking** between neurons so they learn different features.

If all weights started as zero, every neuron in a layer would receive identical gradients and update identically, which means they would remain identical forever. Random initialization avoids this.

---

## 2. What Happens If We Choose Bad Initial Values

Let’s say we choose weights that are too small (e.g., all near zero):

* Activations stay tiny, especially with **sigmoid** or **tanh**, leading to **vanishing gradients**.
* Network learns extremely slowly.

If weights are too large:

* Activations blow up, especially in deep nets, causing **exploding gradients**.
* For sigmoid/tanh, large inputs saturate the activation function (derivatives near zero) → gradients vanish anyway.

---

## 3. The Role of Variance in Initialization

We want the signal (both forward activations and backward gradients) to maintain a stable variance as it flows through layers.
If the variance grows or shrinks a lot from one layer to the next, we get exploding or vanishing activations/gradients.

Let:

* $n_{\text{in}}$ = number of inputs to a neuron (fan-in)
* $n_{\text{out}}$ = number of outputs from a neuron (fan-out)

We want:

$$
\text{Var}(Z^{[l]}) \approx \text{Var}(A^{[l-1]})
$$

and

$$
\text{Var}(\text{gradients}) \; \text{roughly constant across layers}.
$$

The choice of **weight variance** depends on the activation function.

---

## 4. Common Initialization Methods

### (a) **Standard Normal or Uniform**

* $W \sim \mathcal{N}(0, 1)$ or $W \sim U(-1, 1)$
* Simple, but usually unstable for deep networks.

### (b) **Xavier / Glorot Initialization** (for sigmoid/tanh)

Proposed to keep forward and backward variances equal.
For weights:

$$
\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

If using normal distribution:

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
$$

If using uniform:

$$
W \sim U\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)
$$

This keeps the signal variance balanced.

---

### (c) **He Initialization** (for ReLU family)

ReLU discards half of inputs (sets negatives to zero), so we scale more aggressively:

$$
\text{Var}(W) = \frac{2}{n_{\text{in}}}
$$

If using normal:

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
$$

If using uniform:

$$
W \sim U\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \; \sqrt{\frac{6}{n_{\text{in}}}}\right)
$$

---

## 5. Bias Initialization

Biases are usually set to **zero** initially:

* This does not cause symmetry problems because weights are already random.
* For ReLU, sometimes small positive biases (like 0.01) are used to reduce the chance of “dead” neurons early on.

---

## 6. How It Looks in Code

From my earlier MLP example:

```python
if init == 'xavier':
    scale1 = np.sqrt(2.0 / (input_size + hidden_size))
    scale2 = np.sqrt(2.0 / (hidden_size + output_size))
elif init == 'he':
    scale1 = np.sqrt(2.0 / input_size)
    scale2 = np.sqrt(2.0 / hidden_size)
else:
    scale1 = scale2 = 0.01

self.W1 = np.random.randn(input_size, hidden_size) * scale1
self.b1 = np.zeros((1, hidden_size))
self.W2 = np.random.randn(hidden_size, output_size) * scale2
self.b2 = np.zeros((1, output_size))
```

Here:

* `np.random.randn(...)` draws from $\mathcal{N}(0, 1)$.
* Multiplying by `scale1` or `scale2` adjusts variance according to the chosen initialization strategy.
* Biases are zeros.

---

## 7. Summary Table

| Method       | Variance formula         | Best for                     |
| ------------ | ------------------------ | ---------------------------- |
| Xavier       | $2 / (n_{in} + n_{out})$ | sigmoid / tanh               |
| He           | $2 / n_{in}$             | ReLU family                  |
| Small random | arbitrary small number   | shallow nets or testing only |

---

If you want, I can **derive why Xavier’s formula is $2 / (n_{in} + n_{out})$** using variance propagation for forward and backward passes, so you see the math that justifies it. That’s the part most tutorials skip but it makes the reasoning click. Do you want me to go into that derivation next?

---

# BREAK

Alright, let’s break down the **forward pass** for a Multi-Layer Perceptron step-by-step, going from the math to the shapes and then to the code.

---

## 1. Goal of the Forward Pass

The forward pass is the process where we take input data and push it through the network to get predictions.

Mathematically:

1. Multiply inputs by weights and add biases (**linear transformation**).
2. Apply a nonlinear **activation function**.
3. Repeat for each layer until the output layer.
4. At the output layer, produce the final prediction (probabilities for classification).

---

## 2. Notation

We will use:

* $m$: number of examples in the batch.
* $n^{[l]}$: number of neurons in layer $l$ (where $l=0$ is the input layer).
* $ W^{[l]} \in \mathbb{R}^{n^{[l-1]} \times n^{[l]}}$: weight matrix from layer $l-1$ to layer $l$.
* $ b^{[l]} \in \mathbb{R}^{1 \times n^{[l]}}$: bias vector for layer $l$.
* $Z^{[l]}$: pre-activation values.
* $A^{[l]}$: activations (output of layer $l$).

---

## 3. Forward Pass Equations

For each layer $l$ (starting from 1 to $L$):

$$
Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}
$$

$$
A^{[l]} = \phi^{[l]}(Z^{[l]})
$$

Here:

* $A^{[0]} = X$ (the input data).
* $\phi^{[l]}$ is the activation function (sigmoid, ReLU, tanh, softmax, etc.).

---

### Example with 1 hidden layer and softmax output

Given:

* Input $X \in \mathbb{R}^{m \times n_{\text{input}}}$
* Hidden layer: $n_{\text{hidden}}$ neurons with **sigmoid** activation.
* Output layer: $n_{\text{output}}$ neurons with **softmax** activation.

Steps:

**Hidden layer:**

$$
Z^{[1]} = X W^{[1]} + b^{[1]} \quad\text{(shape: \(m \times n_{\text{hidden}}\))}
$$

$$
A^{[1]} = \sigma(Z^{[1]}) \quad\text{(sigmoid applied elementwise)}
$$

**Output layer:**

$$
Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]} \quad\text{(shape: \(m \times n_{\text{output}}\))}
$$

$$
A^{[2]} = \text{softmax}(Z^{[2]}) \quad\text{(rowwise normalization to get probabilities)}
$$

---

## 4. Why We Add Bias

If we had only $X W$, all outputs would pass through the origin. Bias $b$ shifts the activation function left or right, allowing the neuron to represent functions not forced through the origin.
Example: without bias, a sigmoid neuron can only be symmetric around 0, limiting flexibility.

---

## 5. Shape Bookkeeping

Example:

* Batch size $m = 4$
* $n_{\text{input}} = 3$, $n_{\text{hidden}} = 5$, $n_{\text{output}} = 2$

Shapes:

* $X$: (4, 3)
* $W^{[1]}$: (3, 5)
* $b^{[1]}$: (1, 5)
* $Z^{[1]}$: (4, 5)
* $A^{[1]}$: (4, 5)
* $W^{[2]}$: (5, 2)
* $b^{[2]}$: (1, 2)
* $Z^{[2]}$: (4, 2)
* $A^{[2]}$: (4, 2) → final predictions

---

## 6. Numerical Example (Mini)

Say:

$$
X = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{bmatrix} \quad (m=2, n_{\text{input}}=3)
$$

Random:

$$
W^{[1]} = \text{shape }(3,2),\quad b^{[1]}=(1,2)
$$

Step 1:

$$
Z^{[1]} = X W^{[1]} + b^{[1]} \quad \to \quad (2,2)
$$

Step 2:

$$
A^{[1]} = \sigma(Z^{[1]})
$$

Step 3:

$$
Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}
$$

Step 4:

$$
A^{[2]} = \text{softmax}(Z^{[2]})
$$

Now $A^{[2]}$ contains the probabilities for each class per example.

---

## 7. Code Mapping

From earlier:

```python
def forward(self, X):
    # Input layer
    self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
    self.hidden_output = self.sigmoid(self.hidden_input)

    # Output layer
    self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
    self.final_output = self.softmax(self.final_input)

    return self.final_output
```

Code-to-math mapping:

* `self.hidden_input` → $Z^{[1]}$
* `self.hidden_output` → $A^{[1]}$
* `self.final_input` → $Z^{[2]}$
* `self.final_output` → $A^{[2]}$

---

If you like, I can next **derive the softmax equation with its stability trick** so you understand exactly why we subtract the max before exponentiating in the forward pass. That’s crucial for real training stability. Do you want me to go into that?

