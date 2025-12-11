"""
Data loading and preprocessing module for predictive maintenance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import joblib
import os


class DataLoader:
    """Load and preprocess predictive maintenance data"""
    
    def __init__(self, filepath: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataLoader
        
        Args:
            filepath: Path to CSV file
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.filepath = filepath
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_name = "Machine failure"
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Loaded {len(self.df)} samples with {self.df.shape[1]} columns")
        return self.df
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        if self.df is None:
            self.load_data()
        
        return {
            "total_samples": len(self.df),
            "total_features": self.df.shape[1],
            "missing_values": self.df.isnull().sum().to_dict(),
            "target_distribution": self.df[self.target_name].value_counts().to_dict(),
            "feature_types": self.df.dtypes.to_dict(),
        }
    
    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data: handle categorical features, scale numerical features
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.df is None:
            self.load_data()
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_name, "UDI"])  # Drop ID and target
        y = self.df[self.target_name]
        
        # Handle categorical features (Type column)
        if "Type" in X.columns:
            le = LabelEncoder()
            X["Type"] = le.fit_transform(X["Type"])
            self.label_encoders["Type"] = le
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.values, y.values, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Training set: {self.X_train.shape}")
        print(f"✓ Test set: {self.X_test.shape}")
        print(f"✓ Classes: {np.unique(self.y_train)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_scaler(self, filepath: str) -> None:
        """Save scaler for production use"""
        joblib.dump(self.scaler, filepath)
        print(f"✓ Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """Load pre-trained scaler"""
        self.scaler = joblib.load(filepath)
        print(f"✓ Scaler loaded from {filepath}")
