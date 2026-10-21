"""Automated unit test and numerical verification suite via pytest.

Tests input boundary enforcement, Method of Manufactured Solutions (MMS)
verification for the numerical solver, collinear stability, and physical constraints.
"""

import numpy as np
import pytest
from src.io_validator import validate_building_inputs, validate_thermal_outputs
from src.numerical_solver import SVDRegularizedSolver
from src.ml_models import RandomForestThermalModel


def get_valid_sample() -> np.ndarray:
    """Returns a valid 2-sample building geometry array."""
    return np.array([
        [0.98, 514.5, 294.0, 110.25, 7.0, 2.0, 0.0, 0.0],
        [0.62, 808.5, 367.5, 220.50, 3.5, 5.0, 0.4, 5.0],
    ])


def test_io_validator_accepts_valid_data() -> None:
    """Verifies valid physical inputs pass unchanged."""
    X = get_valid_sample()
    res = validate_building_inputs(X)
    assert res.shape == (2, 8)


def test_io_validator_rejects_nan_and_inf() -> None:
    """Verifies NaN and Inf inputs raise ValueError."""
    X = get_valid_sample()
    X[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_building_inputs(X)


def test_io_validator_rejects_unphysical_bounds() -> None:
    """Verifies negative surface area or invalid glazing ratio raises ValueError."""
    X_neg_area = get_valid_sample()
    X_neg_area[0, 2] = -10.0
    with pytest.raises(ValueError, match="Physical violation"):
        validate_building_inputs(X_neg_area)

    X_bad_glazing = get_valid_sample()
    X_bad_glazing[1, 6] = 1.5
    with pytest.raises(ValueError, match="Physical violation"):
        validate_building_inputs(X_bad_glazing)


def test_manufactured_solution_verification() -> None:
    """Verifies SVD solver accuracy against an exact analytical linear function (MMS)."""
    rng = np.random.default_rng(42)
    X_synth = rng.uniform(1.0, 10.0, size=(100, 8))
    true_weights = np.array([[2.5, -1.5, 0.5, 4.0, -2.0, 1.0, 3.0, -0.5],
                             [1.0,  3.0, -2.0, 0.5,  1.5, -1.0, 2.0,  1.0]]).T
    true_bias = np.array([15.0, 20.0])
    Y_exact = X_synth @ true_weights + true_bias

    solver = SVDRegularizedSolver(reg_lambda=0.0)
    solver.fit(X_synth, Y_exact)
    Y_pred = solver.predict(X_synth)

    max_err = np.max(np.abs(Y_exact - Y_pred))
    assert max_err < 1e-10, f"MMS verification failed with max error {max_err}"


def test_collinear_matrix_conditioning_stabilization() -> None:
    """Verifies Tikhonov regularization reduces condition number when X2 = X3 + 2*X4."""
    rng = np.random.default_rng(0)
    X = get_valid_sample()
    X_large = np.vstack([X] * 30) + rng.normal(0, 0.01, size=(60, 8))
    X_large[:, 1] = X_large[:, 2] + 2.0 * X_large[:, 3]
    Y = rng.uniform(10.0, 40.0, size=(60, 2))

    solver = SVDRegularizedSolver(reg_lambda=1e-2).fit(X_large, Y)
    assert solver.raw_condition_number > 1e12
    assert solver.effective_condition_number < 1e4


def test_random_forest_non_negative_outputs() -> None:
    """Ensures Random Forest predictions obey thermodynamic non-negativity."""
    X = get_valid_sample()
    Y = np.array([[15.5, 21.3], [12.1, 18.0]])
    rf = RandomForestThermalModel(n_estimators=10, max_depth=4).fit(X, Y)
    preds = rf.predict(X)
    assert preds.shape == (2, 2)
    assert np.all(preds >= 0.0)
