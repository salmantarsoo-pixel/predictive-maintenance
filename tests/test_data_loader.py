"""
Tests for data loader module
"""

import unittest
import numpy as np
import pandas as pd
from src.data_loader import DataLoader
import tempfile
import os


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a small test dataset
        self.test_data = {
            'UDI': [1, 2, 3, 4, 5],
            'Type': ['M', 'L', 'H', 'M', 'L'],
            'Air temperature [K]': [298.1, 298.2, 298.1, 298.2, 298.2],
            'Process temperature [K]': [308.6, 308.7, 308.5, 308.6, 308.7],
            'Rotational speed [rpm]': [1551, 1408, 1498, 1433, 1408],
            'Torque [Nm]': [42.8, 46.3, 49.4, 39.5, 40.0],
            'Tool wear [min]': [0, 3, 5, 7, 9],
            'TWF': [0, 0, 0, 0, 0],
            'HDF': [0, 0, 0, 0, 0],
            'PWF': [0, 0, 0, 0, 0],
            'OSF': [0, 0, 0, 0, 0],
            'RNF': [0, 0, 0, 0, 0],
            'Machine failure': [0, 0, 1, 0, 0],
        }
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(self.test_data)
        df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_data(self):
        """Test data loading"""
        loader = DataLoader(self.temp_file.name)
        df = loader.load_data()
        
        self.assertEqual(len(df), 5)
        self.assertEqual(df.shape[1], 13)
    
    def test_get_statistics(self):
        """Test statistics computation"""
        loader = DataLoader(self.temp_file.name)
        stats = loader.get_statistics()
        
        self.assertEqual(stats['total_samples'], 5)
        self.assertEqual(stats['total_features'], 13)
        self.assertIn('Machine failure', stats['target_distribution'])
    
    def test_preprocess(self):
        """Test preprocessing"""
        loader = DataLoader(self.temp_file.name)
        X_train, X_test, y_train, y_test = loader.preprocess()
        
        # Check shapes
        self.assertEqual(X_train.shape[0] + X_test.shape[0], 5)
        self.assertEqual(X_train.shape[1], 12)  # Features without UDI
        
        # Check that y values are binary
        self.assertTrue(set(y_train).issubset({0, 1}))
        self.assertTrue(set(y_test).issubset({0, 1}))
    
    def test_scaler_save_load(self):
        """Test scaler persistence"""
        loader = DataLoader(self.temp_file.name)
        loader.preprocess()
        
        # Save scaler
        scaler_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        scaler_file.close()
        
        try:
            loader.save_scaler(scaler_file.name)
            
            # Create new loader and load scaler
            loader2 = DataLoader(self.temp_file.name)
            loader2.load_scaler(scaler_file.name)
            
            self.assertIsNotNone(loader2.scaler)
        finally:
            if os.path.exists(scaler_file.name):
                os.unlink(scaler_file.name)


if __name__ == '__main__':
    unittest.main()
