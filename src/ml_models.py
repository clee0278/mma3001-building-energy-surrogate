"""Machine Learning surrogate models for non-linear building thermal load prediction.

Provides a wrapper around Random Forest Regression with physical output clipping
and theoretical FLOPS estimation for comparison against numerical linear solvers.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestThermalModel:
    """Non-linear ensemble surrogate model for simultaneous Heating & Cooling prediction.

    Attributes:
        n_estimators (int): Number of decision trees in the forest.
        max_depth (int | None): Maximum tree depth to control overfitting and memory.
        model (RandomForestRegressor): Underlying scikit-learn ensemble estimator.
    """

    def __init__(
        self, n_estimators: int = 100, max_depth: int | None = 10, random_state: int = 42
    ) -> None:
        """Initializes the Random Forest surrogate model.

        Args:
            n_estimators (int): Number of trees (> 0).
            max_depth (int | None): Max depth of each tree.
            random_state (int): Seed for deterministic reproducibility.
        """
        if n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,  # Single-thread for fair runtime profiling comparison
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RandomForestThermalModel":
        """Fits the Random Forest ensemble to training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, 8).
            Y (np.ndarray): Target matrix of shape (n_samples, 2).

        Returns:
            RandomForestThermalModel: Fitted model instance.
        """
        self.model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts non-negative heating and cooling loads.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, 8).

        Returns:
            np.ndarray: Predicted thermal loads of shape (n_samples, 2).
        """
        preds = self.model.predict(X)
        return np.maximum(preds, 0.0)

    def estimate_inference_flops(self, n_samples: int) -> int:
        """Estimates approximate inference operations (tree node comparisons).

        Args:
            n_samples (int): Number of test samples evaluated.

        Returns:
            int: Estimated comparison and aggregation operations.
        """
        depth = self.max_depth if self.max_depth is not None else 12
        return int(n_samples * self.n_estimators * depth * 2)
