import numpy as np

from .layer import Layer

class RNN:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        activation_function: str = 'tanh',
        output_activation: str = None
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_function = activation_function
        self.output_activation = output_activation
        
        # Create layers
        self.layers = []
        layer_input_size = input_size
        
        for hidden_size in hidden_sizes:
            layer = Layer(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                activation_function=activation_function
            )
            self.layers.append(layer)
            layer_input_size = hidden_size
        
        # Output layer (linear projection)
        self.W_out = np.random.randn(output_size, hidden_sizes[-1]) * 0.1
        self.b_out = np.zeros(output_size)
        
        # Training history
        self.training_history = {'loss': [], 'epoch': []}
    
    def reset_states(self):
        """Reset all layer states"""
        for layer in self.layers:
            layer.reset_state()
    
    def forward(self, x: np.ndarray, store_for_backprop: bool = False) -> np.ndarray:
        """
        Forward pass through all layers
        
        Args:
            x: Input sequence, shape (seq_len, input_size)
            store_for_backprop: Whether to store intermediate values for backprop
            
        Returns:
            Output sequence, shape (seq_len, output_size)
        """
        # Pass through RNN layers
        layer_input = x
        for layer in self.layers:
            layer_input = layer.forward(layer_input, store_for_backprop)
        
        # Output projection
        output = []
        for h_t in layer_input:
            o_t = self.W_out @ h_t + self.b_out
            if self.output_activation:
                o_t = self._apply_activation(o_t, self.output_activation)
            output.append(o_t)
        
        return np.array(output)
    
    def backward(self, d_output: np.ndarray) -> None:
        """
        Backward pass through all layers
        
        Args:
            d_output: Gradient w.r.t. outputs, shape (seq_len, output_size)
        """
        seq_len = d_output.shape[0]
        
        # Gradient w.r.t. output layer
        d_W_out = np.zeros_like(self.W_out)
        d_b_out = np.zeros_like(self.b_out)
        
        # Get final layer outputs for output gradient computation
        final_layer_output = []
        for layer in self.layers[-1].hidden_states:
            final_layer_output.append(layer)
        final_layer_output.append(self.layers[-1].h)  # Add final state
        
        # Compute output layer gradients
        d_hidden = np.zeros((seq_len, self.hidden_sizes[-1]))
        for t in range(seq_len):
            d_o = d_output[t]
            if self.output_activation:
                # Apply derivative of output activation
                o_t = self.W_out @ final_layer_output[t+1] + self.b_out
                d_o = d_o * self._activation_derivative(o_t, self.output_activation)
            
            d_W_out += np.outer(d_o, final_layer_output[t+1])
            d_b_out += d_o
            d_hidden[t] = self.W_out.T @ d_o
        
        # Backpropagate through RNN layers
        d_layer_output = d_hidden
        layer_gradients = []
        
        for layer in reversed(self.layers):
            d_W_x, d_W_h, d_b_h, d_input = layer.backward(d_layer_output)
            layer_gradients.append((d_W_x, d_W_h, d_b_h))
            d_layer_output = d_input
        
        layer_gradients.reverse()
        
        # Store gradients for parameter update
        self.output_gradients = (d_W_out, d_b_out)
        self.layer_gradients = layer_gradients
    
    def update_parameters(self, learning_rate: float = 0.01):
        """Update all parameters using stored gradients"""
        # Update output layer
        d_W_out, d_b_out = self.output_gradients
        self.W_out -= learning_rate * d_W_out
        self.b_out -= learning_rate * d_b_out
        
        # Update RNN layers
        for layer, (d_W_x, d_W_h, d_b_h) in zip(self.layers, self.layer_gradients):
            layer.update_parameters(d_W_x, d_W_h, d_b_h, learning_rate)
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, loss_type: str = 'mse') -> float:
        """Compute loss between predictions and targets"""
        if loss_type == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif loss_type == 'mae':
            return np.mean(np.abs(y_pred - y_true))
        elif loss_type == 'cross_entropy':
            # Assuming y_true is one-hot encoded
            epsilon = 1e-15  # Prevent log(0)
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=-1))
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_loss_gradient(self, y_pred: np.ndarray, y_true: np.ndarray, loss_type: str = 'mse') -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions"""
        if loss_type == 'mse':
            return 2 * (y_pred - y_true) / len(y_pred)
        elif loss_type == 'mae':
            return np.sign(y_pred - y_true) / len(y_pred)
        elif loss_type == 'cross_entropy':
            return (y_pred - y_true) / len(y_pred)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        loss_type: str = 'mse',
        verbose: bool = True,
        validation_data: tuple = None
    ):
        """
        Train the RNN
        
        Args:
            X: Input sequences, shape (n_sequences, seq_len, input_size) or (seq_len, input_size)
            y: Target sequences, shape (n_sequences, seq_len, output_size) or (seq_len, output_size)
            epochs: Number of training epochs
            learning_rate: Learning rate for parameter updates
            loss_type: Type of loss function ('mse', 'mae', 'cross_entropy')
            verbose: Whether to print training progress
            validation_data: Optional (X_val, y_val) tuple for validation
        """
        # Handle single sequence input
        if X.ndim == 2:
            X = X[np.newaxis, :]
            y = y[np.newaxis, :]
        
        n_sequences = X.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(n_sequences):
                # Reset states for each sequence
                self.reset_states()
                
                # Forward pass
                y_pred = self.forward(X[i], store_for_backprop=True)
                
                # Compute loss
                loss = self.compute_loss(y_pred, y[i], loss_type)
                total_loss += loss
                
                # Backward pass
                d_output = self.compute_loss_gradient(y_pred, y[i], loss_type)
                self.backward(d_output)
                
                # Update parameters
                self.update_parameters(learning_rate)
            
            avg_loss = total_loss / n_sequences
            self.training_history['loss'].append(avg_loss)
            self.training_history['epoch'].append(epoch)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:4d}, Loss: {avg_loss:.6f}")
                
                if validation_data:
                    val_loss = self.evaluate(*validation_data, loss_type)
                    print(f"             Val Loss: {val_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input sequences"""
        if X.ndim == 2:
            X = X[np.newaxis, :]
        
        predictions = []
        for i in range(X.shape[0]):
            self.reset_states()
            pred = self.forward(X[i], store_for_backprop=False)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, loss_type: str = 'mse') -> float:
        """Evaluate the model on given data"""
        predictions = self.predict(X)
        if X.ndim == 2:  # Single sequence
            return self.compute_loss(predictions[0], y, loss_type)
        else:  # Multiple sequences
            total_loss = 0
            for i in range(len(predictions)):
                total_loss += self.compute_loss(predictions[i], y[i], loss_type)
            return total_loss / len(predictions)
    
    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function"""
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'softmax':
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _activation_derivative(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Compute derivative of activation function"""
        if activation == 'sigmoid':
            sig_x = self._apply_activation(x, 'sigmoid')
            return sig_x * (1 - sig_x)
        elif activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            return (x > 0).astype(float)
        else:
            # For softmax, derivative is more complex and context-dependent
            return np.ones_like(x)
    
    def __repr__(self):
        return f"RNN(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, output_size={self.output_size})"
