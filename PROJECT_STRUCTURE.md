# Project Structure Overview

## Directory Layout

```
predictive-maintenance/
│
├── src/                                          # Source code package
│   ├── __init__.py                              # Package initialization
│   ├── data_loader.py                           # Data loading & preprocessing
│   ├── model.py                                 # Neural Network & RF models
│   ├── trainer.py                               # Training utilities
│   └── inference.py                             # Production inference engine
│
├── notebooks/                                    # Jupyter notebooks
│   └── Supervised_Machine_Learning_Project_Predictive_Maintenance.ipynb
│                                                # Full analysis & training
│
├── data/                                        # Data directory
│   └── ai4i2020.csv                            # AI4I 2020 dataset (10,000 samples)
│
├── models/                                      # Trained models (to be created)
│   ├── predictive_maintenance.pth              # Trained PyTorch model
│   ├── random_forest_model.pkl                 # Trained Random Forest
│   └── scaler.pkl                              # Fitted StandardScaler
│
├── examples/                                    # Example scripts
│   ├── __init__.py
│   ├── basic_inference.py                      # Single & batch predictions
│   └── batch_prediction.py                     # CSV batch processing
│
├── tests/                                       # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py                     # Data loader tests
│   ├── test_model.py                           # Model tests
│   └── test_inference.py                       # Inference tests
│
├── requirements.txt                             # Python dependencies
├── setup.py                                    # Package setup & metadata
├── .gitignore                                  # Git ignore rules
├── README.md                                   # Project documentation
├── MODEL_CARD.md                               # Model documentation & metrics
├── CONTRIBUTING.md                             # Contribution guidelines
└── LICENSE                                     # MIT License

## Key Files Description

### Core Source Files
- **data_loader.py**: DataLoader class for CSV loading, preprocessing, scaling
- **model.py**: Neural Network and Random Forest model definitions
- **trainer.py**: Training loops for both PyTorch and scikit-learn models
- **inference.py**: Production-ready inference engine with preprocessing

### Documentation
- **README.md**: Complete project overview, quick start, examples
- **MODEL_CARD.md**: Detailed model documentation, performance, limitations
- **CONTRIBUTING.md**: Guidelines for contributors

### Data & Models
- **data/ai4i2020.csv**: Complete dataset with 10,000 samples
- **models/**: Directory for trained model artifacts (will be created during training)

### Testing & Examples
- **tests/**: Unit tests for all modules
- **examples/**: Practical usage examples and demo scripts

## File Statistics

- Total Project Files: 24
- Python Source Files: 8
- Documentation Files: 3
- Test Files: 3
- Example Scripts: 2
- Configuration Files: 3
- Data Files: 1
- Notebook Files: 1

## Installation & Setup

All files are organized and ready for:
1. ✓ Local development (run examples, train models)
2. ✓ Testing (pytest from tests/ directory)
3. ✓ Packaging (pip install -e .)
4. ✓ Production deployment (inference.py)
5. ✓ Git version control (proper .gitignore)

## Next Steps

1. **Initialize Git Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Predictive Maintenance ML Project"
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Tests**
   ```bash
   pytest tests/
   ```

4. **Train Models**
   - Run the Jupyter notebook for full analysis
   - Or use examples/train_model.py script

5. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/yourusername/predictive-maintenance
   git push -u origin main
   ```

## Version Information

- **Project Version**: 1.0.0
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **scikit-learn**: 1.3+
- **Last Updated**: December 2024
