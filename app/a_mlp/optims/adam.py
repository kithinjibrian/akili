import numpy as np
from .base import Optimizer
from app.a_mlp.layer import Layer

class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    
    def initialize(self, layer: 'Layer') -> None:
        layer.optimizer_state['m_W'] = np.zeros_like(layer.W)
        layer.optimizer_state['v_W'] = np.zeros_like(layer.W)
        layer.optimizer_state['m_b'] = np.zeros_like(layer.b)
        layer.optimizer_state['v_b'] = np.zeros_like(layer.b)
        layer.optimizer_state['t'] = 0  # time step
    
    def update(self, layer: 'Layer') -> None:
        layer.optimizer_state['t'] += 1
        t = layer.optimizer_state['t']
        
        # Update biased first moment estimate
        layer.optimizer_state['m_W'] = (self.beta1 * layer.optimizer_state['m_W'] + 
                                       (1 - self.beta1) * layer.dW)
        layer.optimizer_state['m_b'] = (self.beta1 * layer.optimizer_state['m_b'] + 
                                       (1 - self.beta1) * layer.db)
        
        # Update biased second raw moment estimate
        layer.optimizer_state['v_W'] = (self.beta2 * layer.optimizer_state['v_W'] + 
                                       (1 - self.beta2) * layer.dW**2)
        layer.optimizer_state['v_b'] = (self.beta2 * layer.optimizer_state['v_b'] + 
                                       (1 - self.beta2) * layer.db**2)
        
        # Compute bias-corrected first moment estimate
        m_W_corrected = layer.optimizer_state['m_W'] / (1 - self.beta1**t)
        m_b_corrected = layer.optimizer_state['m_b'] / (1 - self.beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_W_corrected = layer.optimizer_state['v_W'] / (1 - self.beta2**t)
        v_b_corrected = layer.optimizer_state['v_b'] / (1 - self.beta2**t)
        
        # Update parameters
        layer.W -= self.learning_rate * m_W_corrected / (np.sqrt(v_W_corrected) + self.epsilon)
        layer.b -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
