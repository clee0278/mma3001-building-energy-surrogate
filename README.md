# MMA3001: Building Energy Efficiency Surrogate Modeling

Computational engineering project for **MMA3001 (Numerical Methods and Machine Learning)**.
This repository implements and validates a numerical linear algebra baseline (**SVD with Tikhonov Regularization**) against a non-linear machine learning surrogate (**Random Forest Regressor**) to predict building **Heating Load ($Y_1$)** and **Cooling Load ($Y_2$)** from eight architectural envelope and glazing parameters (UCI 242 Dataset).

## Repository Structure

```text
mma3001-building-energy-surrogate/
├── README.md                  # Orientation and reproducible workflow instructions
├── LICENSE                    # MIT Open-Source License
├── requirements.txt           # Pinned Python package dependencies
├── data/raw/                  # Immutable UCI 242 Energy Efficiency dataset
├── src/
│   ├── io_validator.py        # Physical domain, unit, and array boundary enforcement
│   ├── numerical_solver.py    # SVD Tikhonov-regularized least squares & FLOPS counter
│   ├── ml_models.py           # Random Forest surrogate model & complexity estimator
│   └── evaluation.py          # Cross-validation, noise sensitivity, and profiling pipeline
├── tests/
│   └── test_project.py        # Pytest suite (MMS verification & physical constraint checks)
├── docs/                      # HTML source documentation auto-generated via pdoc
└── results/
    ├── figures/               # Generated diagnostic and performance plots
    ├── model_comparison_metrics.csv
    └── pytest_report.txt      # Output log of automated unit tests
```
