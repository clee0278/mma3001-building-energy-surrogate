# MMA3001: Building Energy Efficiency Surrogate Modeling

Individual computational engineering project for **MMA3001 (Numerical Methods and Machine Learning)**.
This repository implements, verifies, and validates a numerical linear algebra baseline (**Tikhonov-Regularized Singular Value Decomposition**) against a non-linear machine learning surrogate (**Random Forest Regressor**) to predict steady-state building **Heating Load ($Y_1$)** and **Cooling Load ($Y_2$)** ($kWh/m^2$) from eight architectural geometry and glazing parameters (UCI 242 Dataset).

## 1. Repository Structure
```text
mma3001-building-energy-surrogate/
├── README.md                  # Orientation, execution, testing, and results guide
├── LICENSE                    # MIT Open-Source License
├── .gitignore                 # Excludes Python caches, virtualenvs, and OS temp files
├── requirements.txt           # Pinned Python package dependencies
├── data/raw/
│   └── energy_efficiency_uci242.csv  # Immutable UCI 242 dataset (768 samples, 8 inputs, 2 outputs)
├── src/
│   ├── __init__.py            # Package initialization
│   ├── io_validator.py        # Physical domain, unit, and array boundary enforcement
│   ├── numerical_solver.py    # SVD Tikhonov-regularized least squares & FLOPS estimator
│   ├── ml_models.py           # Random Forest surrogate model & complexity estimator
│   └── evaluation.py          # Cross-validation, noise sensitivity, profiling & plotting pipeline
├── tests/
│   └── test_project.py        # Automated pytest suite (MMS verification & physical bounds)
├── docs/                      # Static HTML source documentation auto-generated via pdoc
│   ├── index.html             # Entry point for HTML API documentation
│   ├── src.html               # Top-level module overview
│   └── src/                   # Detailed HTML pages for each Python module
└── results/
    ├── figures/               # High-resolution diagnostic and performance plots (PNG)
    ├── model_comparison_metrics.csv  # Quantitative benchmark summary table
    └── pytest_report.txt      # Saved execution log of automated unit tests
```

## 2. Environment Setup & Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/clee0278/mma3001-building-energy-surrogate.git
cd mma3001-building-energy-surrogate
pip install -r requirements.txt
```

## 3. How to Execute the Project
Run the complete evaluation and profiling pipeline:
```bash
PYTHONPATH=. python src/evaluation.py
```

## 4. How to Run the Automated Tests
Execute the automated verification test suite:
```bash
PYTHONPATH=. pytest -v
```
