# Credit Card Fraud Detection - CI/CD Practice Project

A lightweight, production-ready Machine Learning project structure designed for practicing Continuous Integration and Continuous Deployment (CI/CD) with GitHub Actions.

---

## 📁 Project Structure

```
ci/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline (lint + unit tests)
├── data/
│   └── sample_creditcard.csv    # Small sample dataset (125 rows) for fast CI runs
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # Data loading & feature scaling
│   ├── model.py                 # GaussianNB model training & evaluation
│   └── predict.py               # Inference helper
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py       # Unit tests for data preprocessing
│   └── test_model.py            # Unit tests for training & predictions
├── Credit Card Fraud Prediction/ # Original notebook and raw data
├── .gitignore                   # Excludes large files (>100MB) & cache
├── requirements.txt             # Project dependencies
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run unit tests:
```bash
pytest tests/ -v
```

### 3. Check code formatting/linting:
```bash
flake8 src tests
```
