from datetime import datetime
import json
from pathlib import Path
import pickle
from typing import Any, List, Optional, Tuple, Union

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

        self.val_loss_history: List[Union[float, np.floating[Any]]] = []
        self.val_accuracy_history: List[Union[float, np.floating[Any]]] = []

        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

        self.optimizer: Optimizer = optimizer

        self.best_val_loss: float = float('inf')
        self.best_val_accuracy: float = -float('inf')
        self.best_epoch: int = 0
        self.best_model_state: Optional[dict] = None
        self.patience_counter: int = 0

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

    def train(self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 1000, 
        batch_size: Optional[int] = None, 
        verbose: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        save_best_model: bool = False,
        model_save_path: str = "best_model",
        monitor: str = 'val_loss',
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.001
    ) -> None:
        """
        Train the neural network with optional validation tracking and best model saving
        
        Args:
            X, y: Training data
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            verbose: Whether to print training progress
            validation_data: Tuple of (X_val, y_val) for validation
            save_best_model: Whether to automatically save the best model
            model_save_path: Path to save the best model
            monitor: Metric to monitor ('val_loss', 'val_accuracy', 'loss', 'accuracy')
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            
        Returns:
            Dictionary with training history and results
        """
        n_samples = X.shape[0]
        
        if batch_size is None:
            batch_size = n_samples

        X_val, y_val = validation_data if validation_data else (None, None)
        has_validation = X_val is not None and y_val is not None

        if not self.loss_history:  # Starting fresh
            self.best_val_loss = float('inf')
            self.best_val_accuracy = -float('inf')
            self.best_epoch = 0
            self.patience_counter = 0

        print(f"Starting training for {epochs} epochs...")
        if has_validation:
            print(f"Validation data provided: {len(X_val)} samples")
        if save_best_model:
            print(f"Best model will be saved to: {model_save_path}")
        if early_stopping:
            print(f"Early stopping enabled - patience: {patience}, monitoring: {monitor}")
        
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

            val_loss, val_accuracy = None, None
            if has_validation:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_accuracy = self.compute_accuracy(y_val, val_pred)
                
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_accuracy)

            current_epoch = len(self.loss_history) - 1
            is_best_model = self._check_best_model(avg_loss, avg_accuracy, val_loss, val_accuracy, 
                                                  monitor, min_delta, current_epoch)
            
            if save_best_model and is_best_model:
                self._save_current_best_model(model_save_path, current_epoch, 
                                            val_loss, val_accuracy)
                
            if early_stopping and has_validation:
                if self._should_stop_early(monitor, patience):
                    print(f"\nEarly stopping triggered at epoch {current_epoch}")
                    print(f"Best {monitor}: {self._get_best_metric_value(monitor):.4f} at epoch {self.best_epoch}")
                    break

            if verbose and (current_epoch % max(1, epochs // 10) == 0 or current_epoch < 10):
                output = f"Epoch {current_epoch:4d}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}"
                if has_validation:
                    output += f", Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}"
                if is_best_model:
                    output += " ★ (Best)"
                print(output)
        
        final_epoch = len(self.loss_history) - 1
        print(f"\nTraining completed after {final_epoch + 1} epochs")
        print(f"Final - Loss: {self.loss_history[-1]:.4f}, Accuracy: {self.accuracy_history[-1]:.4f}")
        
        if has_validation:
            print(f"Final - Val Loss: {self.val_loss_history[-1]:.4f}, Val Acc: {self.val_accuracy_history[-1]:.4f}")
            print(f"Best {monitor}: {self._get_best_metric_value(monitor):.4f} at epoch {self.best_epoch}")

        return {
            'final_loss': float(self.loss_history[-1]),
            'final_accuracy': float(self.accuracy_history[-1]),
            'final_val_loss': float(self.val_loss_history[-1]) if has_validation else None,
            'final_val_accuracy': float(self.val_accuracy_history[-1]) if has_validation else None,
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss) if self.best_val_loss != float('inf') else None,
            'best_val_accuracy': float(self.best_val_accuracy) if self.best_val_accuracy != -float('inf') else None,
            'total_epochs': len(self.loss_history),
            'early_stopped': final_epoch + 1 < epochs
        }
    
    def _check_best_model(self, train_loss: float, train_acc: float, 
                         val_loss: Optional[float], val_acc: Optional[float],
                         monitor: str, min_delta: float, epoch: int) -> bool:
        """Check if current model is the best so far based on monitoring metric"""
        # Determine current metric value
        if monitor == 'val_loss' and val_loss is not None:
            current_metric = val_loss
            is_better = current_metric < (self.best_val_loss - min_delta)
        elif monitor == 'val_accuracy' and val_acc is not None:
            current_metric = val_acc
            is_better = current_metric > (self.best_val_accuracy + min_delta)
        elif monitor == 'loss':
            current_metric = train_loss
            best_train_loss = min(self.loss_history[:-1]) if len(self.loss_history) > 1 else float('inf')
            is_better = current_metric < (best_train_loss - min_delta)
        elif monitor == 'accuracy':
            current_metric = train_acc
            best_train_acc = max(self.accuracy_history[:-1]) if len(self.accuracy_history) > 1 else -float('inf')
            is_better = current_metric > (best_train_acc + min_delta)
        else:
            return False
        
        if is_better:
            # Update best metrics
            if val_loss is not None:
                self.best_val_loss = val_loss
            if val_acc is not None:
                self.best_val_accuracy = val_acc
            self.best_epoch = epoch
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def _should_stop_early(self, monitor: str, patience: int) -> bool:
        """Check if early stopping criteria is met"""
        return self.patience_counter >= patience

    def _get_best_metric_value(self, monitor: str) -> float:
        """Get the best value for the monitored metric"""
        if monitor == 'val_loss':
            return self.best_val_loss
        elif monitor == 'val_accuracy':
            return self.best_val_accuracy
        elif monitor == 'loss':
            return min(self.loss_history)
        elif monitor == 'accuracy':
            return max(self.accuracy_history)
        else:
            return 0.0
        
    def _save_current_best_model(self, save_path: str, epoch: int, 
                                val_loss: Optional[float], val_acc: Optional[float]) -> None:
        """Save the current model as the best model"""
        try:
            # Add timestamp and metrics to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f"{save_path}_best_epoch_{epoch}_{timestamp}"
            
            # Save with validation metrics if available
            self.save_model(best_model_path, save_optimizer_state=True, include_metadata=True)
            
            # Create a symlink or copy to a consistent "best" filename
            consistent_path = f"{save_path}_best"
            self.save_model(consistent_path, save_optimizer_state=True, include_metadata=True)
            
            # if verbose_best_model_saving:
            metrics_str = ""
            if val_loss is not None and val_acc is not None:
                metrics_str = f" (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})"
            print(f"  → Best model saved to {consistent_path}{metrics_str}")
                
        except Exception as e:
            print(f"Warning: Failed to save best model: {e}")

    def load_best_model(self, save_path: str) -> bool:
        """Load the best saved model"""
        try:
            best_model_path = f"{save_path}_best"
            self.load_model(best_model_path, resume_training=False)
            print(f"Best model loaded from {best_model_path}")
            return True
        except Exception as e:
            print(f"Failed to load best model: {e}")
            return False
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def save_model(
        self, 
        filepath: str,
        save_optimizer_state: bool = True,
        include_metadata: bool = True
    ) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'input_size': self.input_size,
            'layers': [],
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }

        for i, layer in enumerate(self.layers):
            layer_data = {
                'n_neurons': layer.W.shape[1],
                'activation': layer.activation,
                'weights': layer.W,
                'biases': layer.b,
                'optimizer_class': layer.optimizer.__class__.__name__,
                'optimizer_params': self._get_optimizer_params(layer.optimizer)
            }

            if save_optimizer_state:
                layer_data['optimizer_state'] = layer.optimizer_state.copy()
            
            model_data['layers'].append(layer_data)

        if include_metadata:
            model_data['metadata'] = {
                'created_at': datetime.now().isoformat(),
                'total_epochs': len(self.loss_history),
                'final_loss': self.loss_history[-1] if self.loss_history else None,
                'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
                'architecture_summary': self._get_architecture_summary()
            }

        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(model_data, f)

        json_data = self._prepare_json_data(model_data)
        with open(f"{filepath}_info.json", 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Model saved to {filepath}.pkl")
        print(f"Model info saved to {filepath}_info.json")

    def load_model(self, filepath: str, resume_training: bool = True) -> None:
        filepath = Path(filepath)
        
        if not filepath.with_suffix('.pkl').exists():
            raise FileNotFoundError(f"Model file {filepath}.pkl not found")
        
        # Load model data
        with open(f"{filepath}.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore basic properties
        self.input_size = model_data['input_size']
        self.loss_history = model_data.get('loss_history', [])
        self.accuracy_history = model_data.get('accuracy_history', [])
        
        # Clear existing layers
        self.layers = []
        
        # Reconstruct layers
        for i, layer_data in enumerate(model_data['layers']):
            if i == 0:
                n_inputs = self.input_size
            else:
                n_inputs = model_data['layers'][i-1]['n_neurons']
            
            # Create layer
            layer = Layer(n_inputs, layer_data['n_neurons'], layer_data['activation'])
            
            # Restore weights and biases
            layer.W = layer_data['weights']
            layer.b = layer_data['biases']
            
            # Restore optimizer
            optimizer_class = layer_data['optimizer_class']
            optimizer_params = layer_data['optimizer_params']
            layer.optimizer = self._create_optimizer(optimizer_class, optimizer_params)
            layer.optimizer.initialize(layer)
            
            # Restore optimizer state if available and requested
            if resume_training and 'optimizer_state' in layer_data:
                layer.optimizer_state = layer_data['optimizer_state']
            
            self.layers.append(layer)
        
        # Set the network's default optimizer to match the first layer's optimizer
        if self.layers:
            first_optimizer = self.layers[0].optimizer
            self.optimizer = self._create_optimizer(
                first_optimizer.__class__.__name__, 
                self._get_optimizer_params(first_optimizer)
            )
        
        print(f"Model loaded from {filepath}.pkl")
        if 'metadata' in model_data:
            metadata = model_data['metadata']
            print(f"Created: {metadata.get('created_at', 'Unknown')}")
            print(f"Total epochs: {metadata.get('total_epochs', 'Unknown')}")
            print(f"Architecture: {metadata.get('architecture_summary', 'Unknown')}")

    def save_checkpoint(self, filepath: str, epoch: int, 
                       validation_loss: float = None, validation_acc: float = None) -> None:
        checkpoint_path = f"{filepath}_checkpoint_epoch_{epoch}"
        
        # Add checkpoint-specific metadata
        checkpoint_data = {
            'epoch': epoch,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        self.save_model(checkpoint_path, save_optimizer_state=True, include_metadata=True)
        
        # Add checkpoint info to the JSON file
        json_path = Path(f"{checkpoint_path}_info.json")
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            data['checkpoint'] = checkpoint_data
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"Checkpoint saved at epoch {epoch}")

    def _get_optimizer_params(self, optimizer: Optimizer) -> dict:
        """Extract optimizer parameters for serialization"""
        params = {'learning_rate': optimizer.learning_rate}
        
        if isinstance(optimizer, SGD):
            params['momentum'] = getattr(optimizer, 'momentum', 0.0)
        elif isinstance(optimizer, Adam):
            params.update({
                'beta1': getattr(optimizer, 'beta1', 0.9),
                'beta2': getattr(optimizer, 'beta2', 0.999),
                'epsilon': getattr(optimizer, 'epsilon', 1e-8)
            })
        elif isinstance(optimizer, AdamW):
            params.update({
                'beta1': getattr(optimizer, 'beta1', 0.9),
                'beta2': getattr(optimizer, 'beta2', 0.999),
                'epsilon': getattr(optimizer, 'epsilon', 1e-8),
                'weight_decay': getattr(optimizer, 'weight_decay', 0.01)
            })
        elif isinstance(optimizer, RMSprop):
            params.update({
                'decay_rate': getattr(optimizer, 'decay_rate', 0.9),
                'epsilon': getattr(optimizer, 'epsilon', 1e-8)
            })
        
        return params
    
    def _create_optimizer(self, optimizer_class: str, params: dict) -> Optimizer:
        """Create optimizer instance from class name and parameters"""
        if optimizer_class == 'SGD':
            return SGD(**params)
        elif optimizer_class == 'Adam':
            return Adam(**params)
        elif optimizer_class == 'AdamW':
            return AdamW(**params)
        elif optimizer_class == 'RMSprop':
            return RMSprop(**params)
        else:
            # Default to Adam if unknown optimizer
            print(f"Warning: Unknown optimizer {optimizer_class}, defaulting to Adam")
            return Adam()

    def _get_architecture_summary(self) -> str:
        """Get a string summary of the network architecture"""
        if not self.layers:
            return "Empty network"
        
        summary_parts = [f"Input({self.input_size})"]
        for layer in self.layers:
            n_neurons = layer.W.shape[1]
            activation = layer.activation
            summary_parts.append(f"{activation.upper()}({n_neurons})")
        
        return " -> ".join(summary_parts)
    
    def _prepare_json_data(self, model_data: dict) -> dict:
        """Prepare model data for JSON serialization (remove numpy arrays)"""
        json_data = {
            'input_size': model_data['input_size'],
            'architecture': self._get_architecture_summary(),
            'total_parameters': sum(layer.W.size + layer.b.size for layer in self.layers),
            'layers_info': []
        }
        
        for i, layer_data in enumerate(model_data['layers']):
            layer_info = {
                'layer_index': i,
                'neurons': layer_data['n_neurons'],
                'activation': layer_data['activation'],
                'optimizer': layer_data['optimizer_class'],
                'optimizer_params': layer_data['optimizer_params'],
                'weight_shape': list(layer_data['weights'].shape),
                'bias_shape': list(layer_data['biases'].shape)
            }
            json_data['layers_info'].append(layer_info)
        
        if 'metadata' in model_data:
            json_data['metadata'] = model_data['metadata']
        
        # Training history summary
        if model_data['loss_history']:
            json_data['training_summary'] = {
                'total_epochs': len(model_data['loss_history']),
                'initial_loss': float(model_data['loss_history'][0]),
                'final_loss': float(model_data['loss_history'][-1]),
                'best_loss': float(min(model_data['loss_history'])),
                'initial_accuracy': float(model_data['accuracy_history'][0]) if model_data['accuracy_history'] else None,
                'final_accuracy': float(model_data['accuracy_history'][-1]) if model_data['accuracy_history'] else None,
                'best_accuracy': float(max(model_data['accuracy_history'])) if model_data['accuracy_history'] else None
            }
        
        return json_data

    def plot_training_history(self, show_validation: bool = True) -> None:
        """Plot training history with optional validation curves"""
        has_validation = len(self.val_loss_history) > 0
        
        if has_validation and show_validation:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(len(self.loss_history))
        
        # Plot loss
        axes[0].plot(epochs, self.loss_history, 'b-', label='Training Loss', linewidth=2)
        if has_validation and show_validation:
            val_epochs = range(len(self.val_loss_history))
            axes[0].plot(val_epochs, self.val_loss_history, 'r--', label='Validation Loss', linewidth=2)
            
            # Mark best epoch
            if self.best_epoch < len(self.val_loss_history):
                best_val_loss = self.val_loss_history[self.best_epoch]
                axes[0].axvline(x=self.best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best Epoch ({self.best_epoch})')
                axes[0].scatter([self.best_epoch], [best_val_loss], color='green', s=100, zorder=5)
        
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot accuracy
        axes[1].plot(epochs, self.accuracy_history, 'b-', label='Training Accuracy', linewidth=2)
        if has_validation and show_validation:
            val_epochs = range(len(self.val_accuracy_history))
            axes[1].plot(val_epochs, self.val_accuracy_history, 'r--', label='Validation Accuracy', linewidth=2)
            
            # Mark best epoch
            if self.best_epoch < len(self.val_accuracy_history):
                best_val_acc = self.val_accuracy_history[self.best_epoch]
                axes[1].axvline(x=self.best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best Epoch ({self.best_epoch})')
                axes[1].scatter([self.best_epoch], [best_val_acc], color='green', s=100, zorder=5)
        
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        if has_validation:
            print(f"\nTraining Summary:")
            print(f"Best Validation Loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
            print(f"Best Validation Accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}")
            print(f"Final Training Loss: {self.loss_history[-1]:.4f}")
            print(f"Final Training Accuracy: {self.accuracy_history[-1]:.4f}")
            if len(self.val_loss_history) > 0:
                print(f"Final Validation Loss: {self.val_loss_history[-1]:.4f}")
                print(f"Final Validation Accuracy: {self.val_accuracy_history[-1]:.4f}")