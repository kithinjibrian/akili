import numpy as np
from .base import Optimizer
from app.a_mlp.layer import Layer

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
    
    def initialize(self, layer: 'Layer') -> None:
        layer.optimizer_state['s_W'] = np.zeros_like(layer.W)
        layer.optimizer_state['s_b'] = np.zeros_like(layer.b)
    
    def update(self, layer: 'Layer') -> None:
        # Update moving average of squared gradients
        layer.optimizer_state['s_W'] = (self.decay_rate * layer.optimizer_state['s_W'] + 
                                       (1 - self.decay_rate) * layer.dW**2)
        layer.optimizer_state['s_b'] = (self.decay_rate * layer.optimizer_state['s_b'] + 
                                       (1 - self.decay_rate) * layer.db**2)
        
        # Update parameters
        layer.W -= self.learning_rate * layer.dW / (np.sqrt(layer.optimizer_state['s_W']) + self.epsilon)
        layer.b -= self.learning_rate * layer.db / (np.sqrt(layer.optimizer_state['s_b']) + self.epsilon)
