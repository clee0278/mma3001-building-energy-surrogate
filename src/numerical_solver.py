"""Numerical linear algebra solver for baseline thermal load prediction.

Implements Singular Value Decomposition (SVD) with Tikhonov (L2) Regularization
to overcome severe multicollinearity (X2 = X3 + 2*X4) in building geometry matrices.
"""

import numpy as np


class SVDRegularizedSolver:
    """Tikhonov-regularized Linear Least Squares solver using SVD.

    Solves min_W ||X_aug W - Y||_2^2 + lambda * ||W||_2^2 via Singular Value
    Decomposition (X_aug = U S V^T) to guarantee numerical stability even when
    the design matrix is rank-deficient or ill-conditioned.

    Attributes:
        reg_lambda (float): Tikhonov L2 regularization parameter (>= 0).
        weights (np.ndarray | None): Fitted weight matrix of shape (p + 1, n_targets).
        raw_condition_number (float): Condition number kappa(X_aug) before regularization.
        effective_condition_number (float): Stabilized condition number after Tikhonov filtering.
    """

    def __init__(self, reg_lambda: float = 1e-3) -> None:
        """Initializes the SVD solver with a regularization penalty.

        Args:
            reg_lambda (float): L2 penalty parameter. Must be non-negative.

        Raises:
            ValueError: If reg_lambda is negative.
        """
        if reg_lambda < 0.0:
            raise ValueError("Regularization parameter reg_lambda must be >= 0.")
        self.reg_lambda = float(reg_lambda)
        self.weights: np.ndarray | None = None
        self.raw_condition_number: float = np.nan
        self.effective_condition_number: float = np.nan
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _augment_and_scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standardizes features and appends a bias intercept column of ones."""
        if fit:
            self._mean = np.mean(X, axis=0)
            self._std = np.where(np.std(X, axis=0) == 0, 1.0, np.std(X, axis=0))
        X_scaled = (X - self._mean) / self._std
        ones = np.ones((X_scaled.shape[0], 1), dtype=np.float64)
        return np.hstack([ones, X_scaled])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SVDRegularizedSolver":
        """Fits the linear model via economy Singular Value Decomposition.

        Args:
            X (np.ndarray): Training input matrix of shape (n_samples, p).
            Y (np.ndarray): Training target matrix of shape (n_samples, k).

        Returns:
            SVDRegularizedSolver: Fitted solver instance.
        """
        X_aug = self._augment_and_scale(X, fit=True)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Compute economy SVD: X_aug = U * diag(s) * Vt
        U, s, Vt = np.linalg.svd(X_aug, full_matrices=False)

        # Record conditioning metrics for engineering analysis
        min_s = s[-1] if s[-1] > 1e-16 else 1e-16
        self.raw_condition_number = float(s[0] / min_s)
        self.effective_condition_number = float(
            np.sqrt((s[0] ** 2 + self.reg_lambda) / (s[-1] ** 2 + self.reg_lambda))
        )

        # Tikhonov filter factors: d_i = s_i / (s_i^2 + lambda)
        filter_factors = s / (s ** 2 + self.reg_lambda)
        self.weights = (Vt.T * filter_factors) @ (U.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts thermal loads for given building parameters.

        Args:
            X (np.ndarray): Input array of shape (n_samples, p).

        Returns:
            np.ndarray: Predicted thermal loads clipped at physical lower bound 0.

        Raises:
            RuntimeError: If called before fitting the model.
        """
        if self.weights is None:
            raise RuntimeError("Solver must be fitted via .fit() before calling .predict().")
        X_aug = self._augment_and_scale(X, fit=False)
        preds = X_aug @ self.weights
        return np.maximum(preds, 0.0)

    @staticmethod
    def estimate_flops(n_samples: int, n_features: int, n_targets: int = 2) -> int:
        """Estimates theoretical floating-point operations (FLOPS) for Golub-Reinsch SVD fit.

        Args:
            n_samples (int): Number of rows (m).
            n_features (int): Number of columns including intercept (n).
            n_targets (int): Number of output columns (k).

        Returns:
            int: Approximate number of floating-point operations.
        """
        m, n = n_samples, n_features + 1
        svd_flops = 4 * m * (n ** 2) + 8 * (n ** 3)
        backsub_flops = 2 * n * m * n_targets + 2 * (n ** 2) * n_targets
        return int(svd_flops + backsub_flops)
