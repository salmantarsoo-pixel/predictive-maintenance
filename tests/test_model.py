"""
Tests for model module
"""

import unittest
import torch
import numpy as np
from src.model import PredictiveMaintenanceModel, RandomForestModel
from sklearn.ensemble import RandomForestClassifier


class TestNeuralNetModel(unittest.TestCase):
    """Test cases for neural network model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_size = 12
        self.model = PredictiveMaintenanceModel(input_size=self.input_size)
    
    def test_model_creation(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass"""
        X = torch.randn(4, self.input_size)  # Batch of 4 samples
        output = self.model(X)
        
        # Check output shape
        self.assertEqual(output.shape, (4, 2))  # 4 samples, 2 classes
    
    def test_get_probabilities(self):
        """Test probability output"""
        X = torch.randn(4, self.input_size)
        proba = self.model.get_probabilities(X)
        
        # Check shape
        self.assertEqual(proba.shape, (4, 2))
        
        # Check that probabilities sum to 1
        sums = proba.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(4), rtol=1e-4, atol=1e-4)
    
    def test_different_hidden_sizes(self):
        """Test model with custom hidden layers"""
        model = PredictiveMaintenanceModel(
            input_size=self.input_size,
            hidden_sizes=[64, 32]
        )
        
        X = torch.randn(4, self.input_size)
        output = model(X)
        
        self.assertEqual(output.shape, (4, 2))


class TestRandomForestModel(unittest.TestCase):
    """Test cases for random forest wrapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model = RandomForestModel(model=base_model)
    
    def test_model_creation(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model.model)


if __name__ == '__main__':
    unittest.main()
