"""
Example 2: Batch Prediction
Shows how to predict on a large batch of samples from CSV
"""

import pandas as pd
import numpy as np
from src.inference import PredictiveMaintenanceInference


def main():
    # Initialize inference engine
    inference = PredictiveMaintenanceInference(
        model_path="models/predictive_maintenance.pth",
        scaler_path="models/scaler.pkl",
        device="cpu"
    )
    
    # Load test data
    print("Loading test data...")
    df = pd.read_csv("data/ai4i2020.csv")
    
    # Prepare features (same as training)
    X = df.drop(columns=["Machine failure", "UDI"]).values
    
    # Make batch predictions
    print("Making predictions...")
    predictions = inference.predict_batch(X)
    
    # Get risk scores
    risk_scores = np.array([
        inference.get_risk_score(X[i:i+1]) for i in range(len(X))
    ])
    
    # Create results dataframe
    results = pd.DataFrame({
        "prediction": predictions,
        "class_name": ["Failure" if p == 1 else "No Failure" for p in predictions],
        "risk_score": risk_scores,
    })
    
    # Summary statistics
    print("\n=== Prediction Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Predicted failures: {(results['prediction'] == 1).sum()}")
    print(f"Predicted no failures: {(results['prediction'] == 0).sum()}")
    print(f"Average risk score: {risk_scores.mean():.2%}")
    print(f"High risk samples (>50%): {(risk_scores > 0.5).sum()}")
    
    # Save results
    print("\nSaving results to results.csv...")
    results.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
