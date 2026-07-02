# HTL Bio-Oil Yield Prediction using Machine Learning

## Overview

This project predicts **bio-oil yield (%)** from Hydrothermal Liquefaction (HTL) experiments using machine learning.

The work evaluates multiple regression algorithms, performs hyperparameter optimization, and explains model predictions using SHAP, Partial Dependence Plots (PDP), and Permutation Importance.

The project is built as a modular ML framework instead of a single notebook.

---

# Dataset

Source:

PNNL HTL Database

Dataset contains:

- 2284 experiments
- 29 input features
- Bio-oil Yield (%) prediction target

Features include:

- Biomass composition
- Operating temperature
- Heating profile
- Residence time
- Solvent
- HHV
- Lipids
- Proteins
- Ash
- Cellulose
- Hemicellulose
- etc.

---

# Project Structure

```
HTL_BioOil_Prediction/

├── data/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── *.csv
│
├── src/
│   ├── core/
│   ├── models/
│   ├── visualization/
│   ├── experiments/
│   └── legacy/
│
├── requirements.txt
└── README.md
```

---

# Workflow

```
Dataset
    │
    ▼
Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
    │
    ▼
Hyperparameter Optimization
    │
    ▼
Model Comparison
    │
    ▼
Explainability
    │
    ▼
Final Dashboard
```

---

# Models Evaluated

- Linear Regression
- Gradient Boosting
- Random Forest
- Extra Trees
- XGBoost
- CatBoost

Optimized Models

- Tuned Random Forest
- Tuned XGBoost

---

# Explainability

The project includes:

- SHAP Values
- Partial Dependence Plots
- Permutation Importance
- Feature Importance

---

# Best Model

| Model | Test R² | Test MAE |
|-------|---------|----------|
| Tuned XGBoost | **0.8689** | **4.2589** |

---

# Important Features

Permutation Importance identified:

- Lipids
- Temperature
- HHV
- Proteins
- Fatty Acids

as the strongest predictors of bio-oil yield.

---

# Results

The project automatically generates:

- Parity plots
- Error distribution
- SHAP analysis
- PDP plots
- Violin plots
- Correlation heatmap
- Model comparison dashboard

---

# Installation

```bash
git clone <repo>

cd HTL_BioOil_Prediction

pip install -r requirements.txt
```

---

# Run

Train Random Forest

```bash
python -m src.models.random_forest
```

Train XGBoost

```bash
python -m src.models.xgboost
```

Hyperparameter Search

```bash
python -m src.experiments.hyperparameter_tuning
```

Permutation Importance

```bash
python -m src.visualization.permutation_importance
```

Dashboard

```bash
python -m src.visualization.model_dashboard
```

---

# Technologies

- Python
- Scikit-Learn
- XGBoost
- CatBoost
- Pandas
- NumPy
- SHAP
- Matplotlib

---

# Future Work

- Deep Learning models
- Graph Neural Networks
- Uncertainty Estimation
- Bayesian Optimization
- AutoML
- Streamlit Deployment

---

# Author

Rohan

Indian Institute of Technology Kharagpur