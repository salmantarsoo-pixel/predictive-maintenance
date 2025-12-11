"""
Training utilities for predictive maintenance models
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple
import joblib


class NeuralNetTrainer:
    """Trainer for PyTorch neural networks"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.history = {"train_loss": [], "val_loss": []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device).float()
            batch_y = batch_y.to(self.device).long()
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader):
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device).long()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 50, batch_size: int = 32, verbose: bool = True) -> Dict:
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            Training history
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device).float()
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device).float()
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()


class RandomForestTrainer:
    """Trainer for Random Forest models"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        """Initialize Random Forest"""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        """Train the model"""
        self.model.fit(X_train, y_train)
        if verbose:
            print(f"✓ Random Forest trained with {self.model.n_estimators} trees")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Save model"""
        joblib.dump(self.model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        self.model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions
        y_proba: Probability predictions (for AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
    
    return metrics
