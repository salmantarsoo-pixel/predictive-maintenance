"""
Production inference module for predictive maintenance
"""

import torch
import numpy as np
from typing import Dict, Any
import joblib


class PredictiveMaintenanceInference:
    """Production inference engine for predictive maintenance"""
    
    def __init__(self, model_path: str, scaler_path: str, device: str = 'cpu'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.class_names = {0: "No Failure", 1: "Failure"}
    
    def _load_model(self, model_path: str):
        """Load PyTorch model"""
        if model_path.endswith('.pth'):
            model = torch.load(model_path, map_location=self.device)
            model.eval()
        else:
            # Assume joblib model (Random Forest)
            model = joblib.load(model_path)
        return model
    
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess input features
        
        Args:
            features: Input features (raw or batch)
            
        Returns:
            Scaled features
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        return self.scaler.transform(features)
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make a single prediction
        
        Args:
            features: Input features
            
        Returns:
            Prediction dictionary
        """
        features_scaled = self.preprocess(features)
        
        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                X_tensor = torch.from_numpy(features_scaled).to(self.device).float()
                outputs = self.model(X_tensor)
                proba = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)
            
            prediction = pred.cpu().item()
            confidence = proba[0, prediction].cpu().item()
        else:
            # Sklearn model
            prediction = self.model.predict(features_scaled)[0]
            proba = self.model.predict_proba(features_scaled)
            confidence = proba[0, prediction]
        
        return {
            "prediction": prediction,
            "class_name": self.class_names[prediction],
            "confidence": float(confidence),
            "probabilities": {
                self.class_names[i]: float(p) 
                for i, p in enumerate(proba[0] if not isinstance(self.model, torch.nn.Module) else 
                                          self.model(torch.from_numpy(features_scaled).to(self.device).float()).softmax(dim=1)[0].cpu().numpy())
            }
        }
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Make batch predictions
        
        Args:
            features: Batch of features
            
        Returns:
            Array of predictions
        """
        features_scaled = self.preprocess(features)
        
        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                X_tensor = torch.from_numpy(features_scaled).to(self.device).float()
                outputs = self.model(X_tensor)
                predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()
        else:
            return self.model.predict(features_scaled)
    
    def get_risk_score(self, features: np.ndarray) -> float:
        """
        Get risk score (probability of failure)
        
        Args:
            features: Input features
            
        Returns:
            Risk score (0-1)
        """
        features_scaled = self.preprocess(features)
        
        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                X_tensor = torch.from_numpy(features_scaled).to(self.device).float()
                outputs = self.model(X_tensor)
                proba = torch.softmax(outputs, dim=1)
            return proba[0, 1].cpu().item()  # Probability of failure
        else:
            proba = self.model.predict_proba(features_scaled)
            return proba[0, 1]  # Probability of failure
