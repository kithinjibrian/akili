import numpy as np
from .base import Optimizer
from app.a_mlp.layer import Layer

class AdamW(Optimizer):
    """AdamW optimizer (Adam with weight decay)"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
    
    def initialize(self, layer: 'Layer') -> None:
        layer.optimizer_state['m_W'] = np.zeros_like(layer.W)
        layer.optimizer_state['v_W'] = np.zeros_like(layer.W)
        layer.optimizer_state['m_b'] = np.zeros_like(layer.b)
        layer.optimizer_state['v_b'] = np.zeros_like(layer.b)
        layer.optimizer_state['t'] = 0
    
    def update(self, layer: 'Layer') -> None:
        layer.optimizer_state['t'] += 1
        t = layer.optimizer_state['t']
        
        # Add weight decay to gradients (only for weights, not biases)
        dW_with_decay = layer.dW + self.weight_decay * layer.W
        
        # Update biased first moment estimate
        layer.optimizer_state['m_W'] = (self.beta1 * layer.optimizer_state['m_W'] + 
                                       (1 - self.beta1) * dW_with_decay)
        layer.optimizer_state['m_b'] = (self.beta1 * layer.optimizer_state['m_b'] + 
                                       (1 - self.beta1) * layer.db)
        
        # Update biased second raw moment estimate
        layer.optimizer_state['v_W'] = (self.beta2 * layer.optimizer_state['v_W'] + 
                                       (1 - self.beta2) * dW_with_decay**2)
        layer.optimizer_state['v_b'] = (self.beta2 * layer.optimizer_state['v_b'] + 
                                       (1 - self.beta2) * layer.db**2)
        
        # Compute bias-corrected estimates
        m_W_corrected = layer.optimizer_state['m_W'] / (1 - self.beta1**t)
        m_b_corrected = layer.optimizer_state['m_b'] / (1 - self.beta1**t)
        v_W_corrected = layer.optimizer_state['v_W'] / (1 - self.beta2**t)
        v_b_corrected = layer.optimizer_state['v_b'] / (1 - self.beta2**t)
        
        # Update parameters
        layer.W -= self.learning_rate * m_W_corrected / (np.sqrt(v_W_corrected) + self.epsilon)
        layer.b -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
