import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from .a_mlp import NeuralNetwork

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_metrics_enhanced(history, figsize=(15, 5)):
    """Enhanced training metrics visualization"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Accuracy plot
    if 'accuracy' in history:
        axes[1].plot(epochs, history['accuracy'], 'g-', linewidth=2, label='Training Accuracy', marker='s', markersize=4)
        axes[1].set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # Learning rate decay (if available)
    axes[2].plot(epochs, [0.01 * (0.95 ** epoch) for epoch in epochs], 'r-', linewidth=2, label='Learning Rate')
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def plot_prediction_samples(X_test, y_test, predictions, n_samples=50, figsize=(16, 8)):
    """Plot sample predictions with confidence scores"""
    fig, axes = plt.subplots(4, 5, figsize=figsize)
    fig.suptitle('Sample Predictions vs Actual Labels', fontsize=16, fontweight='bold')
    
    pred_classes = np.argmax(predictions, axis=1)
    pred_confidence = np.max(predictions, axis=1)
    
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            image = X_test[i].reshape(28, 28)
            ax.imshow(image, cmap='gray')
            
            actual = y_test[i] if len(y_test.shape) == 1 else np.argmax(y_test[i])
            predicted = pred_classes[i]
            confidence = pred_confidence[i]
            
            # Color code: green for correct, red for incorrect
            color = 'green' if actual == predicted else 'red'
            ax.set_title(f'True: {actual}, Pred: {predicted}\nConf: {confidence:.3f}', 
                        color=color, fontsize=10, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def demo_mnist() -> Tuple[NeuralNetwork, np.ndarray, np.ndarray, np.ndarray]:   
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Use a subset for faster training (first 10,000 samples)
    X_train = X_train[:60000]
    y_train = y_train[:60000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]
    
    # Preprocess data
    X_train_processed = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test_processed = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)
    
    print(f"\nDataset Information:")
    print(f"Training data shape: {X_train_processed.shape}")
    print(f"Training labels shape: {y_train_onehot.shape}")
    print(f"Test data shape: {X_test_processed.shape}")
    print(f"Test labels shape: {y_test_onehot.shape}")
    
    # Create neural network for MNIST
    nn = NeuralNetwork(input_size=784)  # 28x28 pixels flattened
    nn.add_layer(128, 'relu')   # Hidden layer 1
    nn.add_layer(64, 'relu')    # Hidden layer 2
    nn.add_layer(32, 'relu')    # Hidden layer 3
    nn.add_layer(10, 'softmax') # Output layer: 10 classes (0-9)
    
    nn.train(X_train_processed, y_train_onehot, epochs=50, learning_rate=0.01, batch_size=64, verbose=True)
    
    # Plot enhanced training history
    if hasattr(nn, 'history') and nn.history:
        print("\nDisplaying training metrics...")
        plot_training_metrics_enhanced(nn.history)
    
    test_predictions = nn.predict(X_test_processed)
    test_accuracy = nn.compute_accuracy(y_test_onehot, test_predictions)
    test_loss = nn.compute_loss(y_test_onehot, test_predictions)
    
    print(f"Final Test Results:")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    plot_prediction_samples(X_test_processed, y_test, test_predictions)
    
    return nn, X_test_processed, y_test_onehot, test_predictions

def main():
    """Main function to run the enhanced MNIST demo"""
    try:
        demo_mnist()
    except Exception as e:
        print(f"Error occurred during demo: {str(e)}")
        raise