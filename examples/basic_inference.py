"""
Example 1: Basic Inference
Shows how to load a trained model and make predictions
"""

import numpy as np
from src.inference import PredictiveMaintenanceInference


def main():
    # Initialize inference engine
    inference = PredictiveMaintenanceInference(
        model_path="models/predictive_maintenance.pth",
        scaler_path="models/scaler.pkl",
        device="cpu"
    )
    
    # Example 1: Single prediction
    print("=== Example 1: Single Prediction ===")
    sample = np.array([[298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    result = inference.predict(sample)
    
    print(f"Prediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Risk Score: {inference.get_risk_score(sample):.2%}")
    print()
    
    # Example 2: Batch predictions
    print("=== Example 2: Batch Predictions ===")
    batch = np.array([
        [298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [300.2, 310.5, 1200, 50.3, 50, 0, 1, 0, 0, 0, 0, 0, 1],
        [302.5, 312.1, 2000, 30.5, 150, 1, 0, 0, 0, 0, 0, 1, 0],
    ])
    
    predictions = inference.predict_batch(batch)
    print(f"Predictions: {predictions}")
    print()
    
    # Example 3: Risk scores
    print("=== Example 3: Risk Assessment ===")
    for i, sample in enumerate(batch):
        risk = inference.get_risk_score(sample.reshape(1, -1))
        status = "⚠️  HIGH RISK" if risk > 0.5 else "✓ LOW RISK"
        print(f"Sample {i+1}: Risk Score = {risk:.2%} {status}")


if __name__ == "__main__":
    main()
