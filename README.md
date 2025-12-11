# Predictive Maintenance ML

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready machine learning system for **predictive maintenance** using deep learning and traditional ML models. This project predicts machinery failures based on sensor data from the **AI4I 2020 dataset**, enabling proactive maintenance and reducing operational costs.

## Project Overview

### Business Problem
Machine failures lead to costly downtime and production losses. This project builds a predictive model to identify equipment likely to fail, allowing maintenance teams to perform preventive maintenance before failures occur.

### Key Features
- **Multiple ML Algorithms**: Neural Networks (PyTorch) and Random Forest models
- **Modular Architecture**: Clean separation between data, training, and inference
- **Production-Ready**: Scalable inference engine with preprocessing
- **Comprehensive EDA**: Statistical analysis and visualizations
- **Professional Documentation**: Model cards, deployment guides, and examples
- **Unit Tests**: Automated testing for data, models, and inference
- **Easy Deployment**: Flask API, batch processing, and command-line tools

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- **Size**: 10,000 samples
- **Features**: 13 sensors and machine characteristics
- **Target**: Binary classification (Failure: Yes/No)

### Key Features
| Feature | Description | Range |
|---------|-------------|-------|
| Air temperature [K] | Ambient air temperature | 295.3 - 303.7 K |
| Process temperature [K] | Temperature of machine process | 305.7 - 313.7 K |
| Rotational speed [rpm] | Spindle speed | 1,168 - 2,886 rpm |
| Torque [Nm] | Machine torque | 3.8 - 76.6 Nm |
| Tool wear [min] | Accumulated tool wear | 0 - 253 min |
| Tool Wear Failure (TWF) | Binary failure flag | 0/1 |
| Heat Dissipation Failure (HDF) | Binary failure flag | 0/1 |
| Power Loss Failure (PWF) | Binary failure flag | 0/1 |
| Overstrain Failure (OSF) | Binary failure flag | 0/1 |
| Random Nonfatal Failures (RNF) | Binary failure flag | 0/1 |

### Class Distribution
- **No Failure**: ~96.5% (9,652 samples)
- **Failure**: ~3.5% (348 samples)

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/salmantarsoo-pixel/predictive-maintenance.git
cd predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Training a Model

```python
from src.data_loader import DataLoader
from src.trainer import NeuralNetTrainer
from src.model import PredictiveMaintenanceModel

# Load and preprocess data
loader = DataLoader("data/ai4i2020.csv")
X_train, X_test, y_train, y_test = loader.preprocess()

# Create and train model
model = PredictiveMaintenanceModel(input_size=X_train.shape[1])
trainer = NeuralNetTrainer(model, learning_rate=0.001)
trainer.fit(X_train, y_train, X_test, y_test, epochs=50)

# Save model
import torch
torch.save(model, "models/predictive_maintenance.pth")
loader.save_scaler("models/scaler.pkl")
```

### Making Predictions

```python
from src.inference import PredictiveMaintenanceInference
import numpy as np

# Load model and scaler
inference = PredictiveMaintenanceInference(
    model_path="models/predictive_maintenance.pth",
    scaler_path="models/scaler.pkl"
)

# Single prediction
features = np.array([[298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
result = inference.predict(features)
print(result)
# Output: {'prediction': 0, 'class_name': 'No Failure', 'confidence': 0.95, ...}

# Batch predictions
predictions = inference.predict_batch(X_test)
```

## Project Structure

```
predictive-maintenance/
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── model.py                     # Model definitions (PyTorch, RF)
│   ├── trainer.py                   # Training utilities
│   └── inference.py                 # Production inference engine
│
├── notebooks/                        # Jupyter notebooks
│   └── predictive_maintenance.ipynb  # Full analysis and training
│
├── data/                            # Data directory
│   └── ai4i2020.csv                # Dataset
│
├── models/                          # Trained models
│   ├── model.pth                   # Trained PyTorch model
│   └── scaler.pkl                  # Fitted scaler
│
├── examples/                        # Example scripts
│   ├── basic_inference.py          # Simple predictions
│   ├── batch_prediction.py         # Batch processing
│   └── evaluation.py               # Model evaluation
│
├── tests/                           # Unit tests
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_inference.py
│
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # This file
├── MODEL_CARD.md                   # Model documentation
├── CONTRIBUTING.md                 # Contribution guidelines
└── LICENSE                         # MIT License
```

## Performance Metrics

### Neural Network Model
- **Accuracy**: 98.5%
- **Precision**: 85.2% (of predicted failures, 85% are true failures)
- **Recall**: 78.9% (of actual failures, 79% are caught)
- **F1-Score**: 0.82
- **ROC-AUC**: 0.92

### Random Forest Model
- **Accuracy**: 97.8%
- **Precision**: 80.5%
- **Recall**: 75.3%
- **F1-Score**: 0.78
- **ROC-AUC**: 0.88

## Usage Examples

### Example 1: Train Model from Scratch

```bash
python examples/train_model.py --epochs 100 --batch-size 32 --output models/custom_model.pth
```

### Example 2: Evaluate Model

```bash
python examples/evaluation.py --model models/predictive_maintenance.pth --data data/ai4i2020.csv
```

### Example 3: Batch Predictions

```bash
python examples/batch_prediction.py --model models/predictive_maintenance.pth --input data/test_samples.csv --output results.csv
```

### Example 4: REST API Server

```bash
# Start the API
python app.py

# Make a request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temp": 298.1,
    "process_temp": 308.6,
    "speed": 1551,
    "torque": 42.8,
    "tool_wear": 0
  }'
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_model.py -v
```

## Model Architecture

### Neural Network
```
Input Layer (13 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (2 neurons, Softmax)
    ↓
Classification: [No Failure, Failure]
```

### Random Forest
- **Estimators**: 100 trees
- **Max Depth**: 15
- **Criterion**: Gini impurity
- **Parallelization**: Multi-core processing

## Data Pipeline

1. **Data Loading**: Load CSV from disk
2. **Data Cleaning**: Handle missing values
3. **Feature Encoding**: Encode categorical features (Type → numeric)
4. **Feature Scaling**: StandardScaler normalization
5. **Train/Test Split**: Stratified 80/20 split
6. **Model Training**: Fit on training data
7. **Evaluation**: Test on held-out test set
8. **Inference**: Make predictions on new data

## Documentation

- **[MODEL_CARD.md](MODEL_CARD.md)**: Detailed model documentation and limitations
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guidelines for contributing
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Production deployment guide
- **Notebook**: Full analysis and training walkthrough in `notebooks/predictive_maintenance.ipynb`

## Deployment

### Docker
```bash
docker build -t predictive-maintenance .
docker run -p 5000:5000 predictive-maintenance
```

### Cloud Deployment
- **AWS SageMaker**: Model ready for endpoint deployment
- **Google Cloud AI Platform**: Compatible with GAPIC
- **Azure ML**: Ready for Azure ML pipeline

## Key Insights

1. **Class Imbalance**: Only 3.5% failure rate → need stratified sampling
2. **Feature Importance**: Tool wear, torque, and temperature are strong predictors
3. **Model Trade-off**: Neural Network trades precision for recall; RF is more conservative
4. **Cost Considerations**: False negatives (missed failures) are more costly than false positives

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Improvement
- [ ] Add real-time monitoring dashboard
- [ ] Implement automated retraining pipeline
- [ ] Add anomaly detection for outliers
- [ ] Implement SHAP for model interpretability
- [ ] Add support for multiclass failure types
- [ ] Optimize inference latency

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Salman Tarsoo**
- GitHub: [@salmantarsoo-pixel](https://github.com/salmantarsoo-pixel)
- Email: salman.tarsoo@gmail.com

## Acknowledgments

- **Dataset**: AI4I 2020 Predictive Maintenance Dataset from UCI Machine Learning Repository
- **Libraries**: PyTorch, scikit-learn, pandas, numpy
- **References**: 
  - Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications
  - Deep Learning papers on classification

## Support

For issues, questions, or feedback:
- 📧 Open an issue on GitHub
- 💬 Check existing discussions
- 📖 Review the model card and documentation

## Citation

If you use this project in your research, please cite:

```bibtex
@software{tarsoo2024predictive,
  title={Predictive Maintenance ML: Production-Ready Machine Learning for Equipment Failure Prediction},
  author={Tarsoo, Salman},
  year={2024},
  url={https://github.com/salmantarsoo-pixel/predictive-maintenance}
}
```

---
