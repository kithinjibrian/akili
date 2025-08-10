from abc import ABC, abstractmethod
from app.a_mlp.layer import Layer

class Optimizer(ABC):
    """Abstract base class for optimizers"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    @abstractmethod
    def initialize(self, layer: 'Layer') -> None:
        """Initialize optimizer state for a layer"""
        pass
    
    @abstractmethod
    def update(self, layer: 'Layer') -> None:
        """Update layer weights using computed gradients"""
        pass