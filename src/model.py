"""
Model definitions for predictive maintenance
"""

import torch
import torch.nn as nn
from typing import Tuple


class PredictiveMaintenanceModel(nn.Module):
    """Deep neural network for predictive maintenance classification"""
    
    def __init__(self, input_size: int, hidden_sizes: list = None, dropout: float = 0.2):
        """
        Initialize the model
        
        Args:
            input_size: Number of input features
            hidden_sizes: Sizes of hidden layers [layer1, layer2, ...]
            dropout: Dropout probability
        """
        super(PredictiveMaintenanceModel, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        # Input layer to first hidden
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Final output layer (binary classification)
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Logits for binary classification
        """
        return self.network(x)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class RandomForestModel:
    """Wrapper for scikit-learn RandomForest for consistency"""
    
    def __init__(self, model=None):
        self.model = model
    
    def forward(self, x):
        """For consistency with PyTorch interface"""
        return self.model.predict(x)
    
    def get_probabilities(self, x):
        """Get probability predictions"""
        return self.model.predict_proba(x)
