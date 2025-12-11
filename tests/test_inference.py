"""
Tests for inference module
"""

import unittest
import numpy as np
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


class TestInference(unittest.TestCase):
    """Test cases for inference module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create dummy model and scaler
        self.temp_dir = tempfile.mkdtemp()
        
        # Create and save a dummy sklearn model
        X = np.random.randn(100, 12)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        self.model_path = os.path.join(self.temp_dir, "model.pkl")
        joblib.dump(model, self.model_path)
        
        # Create and save scaler
        scaler = StandardScaler()
        scaler.fit(X)
        self.scaler_path = os.path.join(self.temp_dir, "scaler.pkl")
        joblib.dump(scaler, self.scaler_path)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_inference_initialization(self):
        """Test inference engine initialization"""
        from src.inference import PredictiveMaintenanceInference
        
        # Note: This will load the joblib model, not PyTorch
        inference = PredictiveMaintenanceInference(
            model_path=self.model_path,
            scaler_path=self.scaler_path
        )
        
        self.assertIsNotNone(inference.model)
        self.assertIsNotNone(inference.scaler)
    
    def test_preprocess(self):
        """Test preprocessing"""
        from src.inference import PredictiveMaintenanceInference
        
        inference = PredictiveMaintenanceInference(
            model_path=self.model_path,
            scaler_path=self.scaler_path
        )
        
        X = np.random.randn(4, 12)
        X_scaled = inference.preprocess(X)
        
        self.assertEqual(X_scaled.shape, (4, 12))
    
    def test_batch_predict(self):
        """Test batch prediction"""
        from src.inference import PredictiveMaintenanceInference
        
        inference = PredictiveMaintenanceInference(
            model_path=self.model_path,
            scaler_path=self.scaler_path
        )
        
        X = np.random.randn(4, 12)
        predictions = inference.predict_batch(X)
        
        self.assertEqual(predictions.shape, (4,))
        self.assertTrue(all(p in [0, 1] for p in predictions))


if __name__ == '__main__':
    unittest.main()
