# Building a MLP from Scratch

## Goal
Build a complete Multi-Layer Perceptron (MLP) from first principles to understand forward pass, backpropagation, loss, and learning.

# What to will learn
- Neuron and perceptron intuition
- Forward propagation through multiple layers
- Activation functions and their derivatives (ReLU, sigmoid, tanh)
- Loss functions (cross-entropy, mean squared error) and why cross-entropy fits classification
- Backpropagation derived from the chain rule and implemented end-to-end
- Weight initialization and why it matters
- Basic optimizers: vanilla gradient descent and Adam
- Regularization: L2 weight decay and dropout (optional)
- Training loop, batching, and evaluation metrics (accuracy, loss curves)
- Simple visualizations: loss/accuracy plots and weight/filter inspection

# Prerequisites
- Basic linear algebra: vectors, matrices, matrix multiplication, transpose
- Basic calculus: derivatives and chain rule
- Familiarity with NumPy (we will implement numerical code with NumPy)
- Optional: familiarity with PyPlot / matplotlib for plotting

# Milestones and tasks

## Milestone 0 - Single-layer perceptron (forward only)
- Implement a single dense layer Dense that performs y = xW + b.
- Implement activation functions: sigmoid and ReLU and their derivatives.
- Implement a forward-only classifier with random weights and a function that computes cross-entropy loss.
- Success criteria: forward pass runs and loss is computed for a batch.

## Milestone 1 - Training loop and gradient check
- Implement numerical gradient checking for a small network (finite differences) to verify analytic gradients.
- Implement basic training loop using vanilla gradient descent (batch gradient descent or small batch).
- Success criteria: analytic gradients approximate numerical gradients; loss decreases on a toy problem.

## Milestone 2 - Full MLP with backpropagation
- Implement multi-layer support: stack Dense and Activation layers.
- Implement backprop propagation for each layer and parameter update interface.
- Add batch training, mini-batch support, and shuffling.
- Success criteria: training reduces loss on MNIST training split; accuracy improves over random baseline.

## Milestone 3 - Optimizers and weight init
- Implement Adam optimizer and optionally SGD with momentum.
- Implement Xavier/Glorot and He initialization and compare.
- Success criteria: Adam or momentum helps training converge faster and steadier.

## Milestone 4 - Regularization and early-stopping
- Add L2 weight decay and optional dropout.
- Add a validation split and early-stopping criterion.
- Success criteria: validation loss stops improving and model with best validation performance is saved.

## Milestone 5 - Evaluation and visualization
- Plot training and validation loss/accuracy curves.
- Display a confusion matrix and some misclassified examples with softmax probabilities.
- Visualize first-layer weights as 28x28 images to inspect learned features.
- Success criteria: clear visualizations that help diagnose model behavior.

## Milestone 6 - Robustness experiments and write-up
- Run ablations: change learning rate, initialization, activation, optimizer, batch size.
- Document what changed and how performance changed.
- Prepare the final README and short report summarizing results and lessons learned.
- Success criteria: you have a short report and a reproducible script to re-run experiments.

# Sources
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville - foundational textbook for theory and practice.
- "Neural Networks and Deep Learning" by Michael Nielsen - excellent intuitive treatment and walkthroughs of backprop.
- Original papers and blog posts on initialization: Xavier/Glorot and He initialization notes