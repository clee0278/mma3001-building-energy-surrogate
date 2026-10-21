"""Input and output validation module for Building Energy Efficiency modeling.

Enforces physical domain constraints, data types, and array integrity before
computational solvers are executed.
"""

import numpy as np


def validate_building_inputs(X: np.ndarray) -> np.ndarray:
    """Validates building geometry and glazing input features against physical bounds.

    Args:
        X (np.ndarray): 2D input array of shape (n_samples, 8) containing:
            [X1: Relative Compactness (-), X2: Surface Area (m^2),
             X3: Wall Area (m^2), X4: Roof Area (m^2),
             X5: Overall Height (m), X6: Orientation (index {2,3,4,5}),
             X7: Glazing Area Ratio (-), X8: Glazing Distribution (index {0..5})].

    Returns:
        np.ndarray: Validated 2D float64 numpy array.

    Raises:
        TypeError: If input cannot be coerced to a numeric float64 numpy array.
        ValueError: If array shape is not (n_samples, 8), contains NaN/Inf,
            or violates physical thermodynamic/geometric constraints.
    """
    try:
        X_arr = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise TypeError("Inputs must be numeric and convertible to float64.") from exc

    if X_arr.ndim != 2 or X_arr.shape[1] != 8 or X_arr.shape[0] == 0:
        raise ValueError(f"Expected 2D array of shape (n_samples, 8), got {X_arr.shape}.")

    if not np.all(np.isfinite(X_arr)):
        raise ValueError("Input contains NaN or infinite values.")

    # Enforce physical geometric bounds
    if np.any((X_arr[:, 0] <= 0.0) | (X_arr[:, 0] > 1.0)):
        raise ValueError("Physical violation: Relative Compactness (X1) must be in (0, 1].")

    if np.any(X_arr[:, 1:5] <= 0.0):
        raise ValueError("Physical violation: Areas (X2-X4) and Height (X5) must be > 0.")

    if np.any((X_arr[:, 6] < 0.0) | (X_arr[:, 6] >= 1.0)):
        raise ValueError("Physical violation: Glazing Area Ratio (X7) must be in [0.0, 1.0).")

    return X_arr


def validate_thermal_outputs(Y: np.ndarray) -> np.ndarray:
    """Validates predicted or ground-truth heating and cooling loads.

    Args:
        Y (np.ndarray): Array of shape (n_samples, 2) representing
            [Y1: Heating Load (kWh/m^2), Y2: Cooling Load (kWh/m^2)].

    Returns:
        np.ndarray: Validated 2D float64 array of thermal loads.

    Raises:
        ValueError: If outputs contain NaN/Inf or unphysical negative energy loads.
    """
    Y_arr = np.asarray(Y, dtype=np.float64)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)

    if not np.all(np.isfinite(Y_arr)):
        raise ValueError("Thermal load outputs contain NaN or infinite values.")

    if np.any(Y_arr < 0.0):
        raise ValueError("Thermodynamic violation: Heating/Cooling loads cannot be negative.")

    return Y_arr
