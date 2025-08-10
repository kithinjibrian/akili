import numpy as np
from .base import Optimizer
from app.a_mlp.layer import Layer

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
    
    def initialize(self, layer: 'Layer') -> None:
        if self.momentum > 0:
            layer.optimizer_state['v_W'] = np.zeros_like(layer.W)
            layer.optimizer_state['v_b'] = np.zeros_like(layer.b)
    
    def update(self, layer: 'Layer') -> None:
        if self.momentum > 0:
            # SGD with momentum
            layer.optimizer_state['v_W'] = (self.momentum * layer.optimizer_state['v_W'] - 
                                          self.learning_rate * layer.dW)
            layer.optimizer_state['v_b'] = (self.momentum * layer.optimizer_state['v_b'] - 
                                          self.learning_rate * layer.db)
            
            layer.W += layer.optimizer_state['v_W']
            layer.b += layer.optimizer_state['v_b']
        else:
            # Standard SGD
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db