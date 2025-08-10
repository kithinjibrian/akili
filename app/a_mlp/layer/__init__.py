from typing import Any, Dict, Optional
import numpy as np

class Layer:
    def __init__(
        self, 
        n_inputs: int, 
        n_neurons: int, 
        activation: str = 'relu'
    ):
        if activation == 'relu':
            self.W: np.ndarray = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        else:
            self.W: np.ndarray = np.random.randn(n_inputs, n_neurons) * np.sqrt(1.0 / n_inputs)

        self.b: np.ndarray = np.zeros((1, n_neurons))
        self.activation: str = activation
        self.optimizer_state: Dict[str, Any] = {}

        self.input: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.a: Optional[np.ndarray] = None
        self.dL_dz: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        self.z = np.dot(X, self.W) + self.b        
        self.a = self._apply_activation(self.z)
        return self.a

    def backward(self, dL_da: np.ndarray) -> np.ndarray:
        da_dz = self._activation_derivative()
        dL_dz = dL_da * da_dz
        
        self.dL_dz = dL_dz
        self.dW = self.input.T.dot(dL_dz)
        self.db = np.sum(dL_dz, axis=0, keepdims=True)
        
        dL_dinput = dL_dz.dot(self.W.T)
        return dL_dinput

    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_derivative(self) -> np.ndarray:        
        if self.activation == 'relu':
            return (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = self._apply_activation(self.z)
            return sig * (1 - sig)
        elif self.activation == 'softmax':
            # For softmax, we handle the derivative in the loss function
            # Return identity since we use categorical cross-entropy
            return np.ones_like(self.z)
        elif self.activation == 'tanh':
            return 1 - np.tanh(self.z)**2
        elif self.activation == 'linear':
            return np.ones_like(self.z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")