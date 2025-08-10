import numpy as np

class Layer:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation_function: str = 'tanh',
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        
        # Initialize weights with better scaling
        if activation_function == 'tanh':
            # Xavier initialization
            self.W_x = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
            self.W_h = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        else:  # ReLU
            # He initialization
            self.W_x = np.random.randn(hidden_size, input_size) * np.sqrt(2.0/input_size)
            self.W_h = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
            
        self.b_h = np.zeros(hidden_size)
        self.h = np.zeros(hidden_size)
        
        # Store states for backprop
        self.hidden_states = []
        self.inputs = []
        self.z_values = []  # Pre-activation values
        
    def reset_state(self):
        self.h = np.zeros(self.hidden_size)
        self.hidden_states = []
        self.inputs = []
        self.z_values = []
        
    def step(self, x: np.ndarray, store_for_backprop: bool = False) -> np.ndarray:
        if store_for_backprop:
            self.inputs.append(x.copy())
            self.hidden_states.append(self.h.copy())
        
        z_t = self.W_x @ x + self.W_h @ self.h + self.b_h
        
        if store_for_backprop:
            self.z_values.append(z_t.copy())
            
        self.h = self.activation(z_t)
        return self.h
    
    def forward(self, x: np.ndarray, store_for_backprop: bool = False) -> np.ndarray:
        if store_for_backprop:
            self.reset_state()
            
        output = []
        for x_t in x:
            h_t = self.step(x_t, store_for_backprop)
            output.append(h_t)
        return np.array(output)
    
    def backward(self, d_output: np.ndarray) -> tuple:
        """
        Backpropagation through time
        
        Args:
            d_output: Gradient w.r.t. outputs, shape (seq_len, hidden_size)
            
        Returns:
            tuple: (d_W_x, d_W_h, d_b_h, d_input)
        """
        seq_len = len(self.hidden_states)
        
        # Initialize gradients
        d_W_x = np.zeros_like(self.W_x)
        d_W_h = np.zeros_like(self.W_h)
        d_b_h = np.zeros_like(self.b_h)
        d_input = np.zeros((seq_len, self.input_size))
        
        # Initialize hidden state gradient
        d_h_next = np.zeros(self.hidden_size)
        
        # Backward pass through time
        for t in reversed(range(seq_len)):
            # Total gradient at this timestep
            d_h = d_output[t] + d_h_next
            
            # Gradient through activation
            d_z = d_h * self.activation_derivative(self.z_values[t])
            
            # Gradients w.r.t. parameters
            d_W_x += np.outer(d_z, self.inputs[t])
            d_W_h += np.outer(d_z, self.hidden_states[t])
            d_b_h += d_z
            
            # Gradient w.r.t. input
            d_input[t] = self.W_x.T @ d_z
            
            # Gradient w.r.t. previous hidden state
            d_h_next = self.W_h.T @ d_z
            
        return d_W_x, d_W_h, d_b_h, d_input
    
    def update_parameters(self, d_W_x, d_W_h, d_b_h, learning_rate: float = 0.01):
        """Update parameters using gradients"""
        self.W_x -= learning_rate * d_W_x
        self.W_h -= learning_rate * d_W_h
        self.b_h -= learning_rate * d_b_h
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function"""
        if self.activation_function == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_function == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")
    
    def __repr__(self):
        return f"Layer(input_size={self.input_size}, hidden_size={self.hidden_size}, activation_function='{self.activation_function}')"
