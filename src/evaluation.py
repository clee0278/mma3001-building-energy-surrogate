"""End-to-end evaluation, validation, sensitivity testing, and performance profiling.

Generates quantitative comparison metrics and publication-quality figures for the report.
"""

import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.io_validator import validate_building_inputs, validate_thermal_outputs
from src.numerical_solver import SVDRegularizedSolver
from src.ml_models import RandomForestThermalModel


def profile_execution(func, *args):
    """Measures execution wall-time (ms) and peak memory allocation (KB)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_ms, peak_bytes / 1024.0


def run_full_analysis() -> None:
    """Executes validation, profiling, noise sensitivity, and figure generation."""
    df = pd.read_csv("data/raw/energy_efficiency_uci242.csv")
    X_raw = validate_building_inputs(df.iloc[:, :8].values)
    Y_raw = validate_thermal_outputs(df.iloc[:, 8:10].values)

    # 80/20 Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_raw, Y_raw, test_size=0.20, random_state=42
    )

    # 1. Fit & Profile Baseline Numerical Solver (SVD + Tikhonov)
    svd_solver = SVDRegularizedSolver(reg_lambda=1e-2)
    _, svd_fit_ms, svd_mem_kb = profile_execution(svd_solver.fit, X_train, Y_train)
    Y_pred_svd, svd_pred_ms, _ = profile_execution(svd_solver.predict, X_test)

    # 2. Fit & Profile Primary ML Model (Random Forest)
    rf_model = RandomForestThermalModel(n_estimators=100, max_depth=10)
    _, rf_fit_ms, rf_mem_kb = profile_execution(rf_model.fit, X_train, Y_train)
    Y_pred_rf, rf_pred_ms, _ = profile_execution(rf_model.predict, X_test)

    # 3. 5-Fold Cross-Validation RMSE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    svd_cv_scores, rf_cv_scores = [], []
    for tr_idx, val_idx in kf.split(X_train):
        s_fold = SVDRegularizedSolver(reg_lambda=1e-2).fit(X_train[tr_idx], Y_train[tr_idx])
        r_fold = RandomForestThermalModel(n_estimators=100, max_depth=10).fit(
            X_train[tr_idx], Y_train[tr_idx]
        )
        svd_cv_scores.append(root_mean_squared_error(Y_train[val_idx], s_fold.predict(X_train[val_idx])))
        rf_cv_scores.append(root_mean_squared_error(Y_train[val_idx], r_fold.predict(X_train[val_idx])))

    # Compile Quantitative Comparison Table
    summary = pd.DataFrame([
        {
            "Method": "SVD Tikhonov Least Squares (Baseline)",
            "Test_RMSE_kWh_m2": root_mean_squared_error(Y_test, Y_pred_svd),
            "Test_MAE_kWh_m2": mean_absolute_error(Y_test, Y_pred_svd),
            "Test_R2": r2_score(Y_test, Y_pred_svd),
            "CV5_Mean_RMSE": np.mean(svd_cv_scores),
            "CV5_Std_RMSE": np.std(svd_cv_scores),
            "Fit_Time_ms": svd_fit_ms,
            "Predict_Time_ms": svd_pred_ms,
            "Peak_Memory_KB": svd_mem_kb,
            "Estimated_FLOPS": svd_solver.estimate_flops(len(X_train), 8),
            "Raw_Cond_Num": svd_solver.raw_condition_number,
            "Stabilized_Cond_Num": svd_solver.effective_condition_number,
        },
        {
            "Method": "Random Forest Regressor (Selected ML)",
            "Test_RMSE_kWh_m2": root_mean_squared_error(Y_test, Y_pred_rf),
            "Test_MAE_kWh_m2": mean_absolute_error(Y_test, Y_pred_rf),
            "Test_R2": r2_score(Y_test, Y_pred_rf),
            "CV5_Mean_RMSE": np.mean(rf_cv_scores),
            "CV5_Std_RMSE": np.std(rf_cv_scores),
            "Fit_Time_ms": rf_fit_ms,
            "Predict_Time_ms": rf_pred_ms,
            "Peak_Memory_KB": rf_mem_kb,
            "Estimated_FLOPS": rf_model.estimate_inference_flops(len(X_test)),
            "Raw_Cond_Num": np.nan,
            "Stabilized_Cond_Num": np.nan,
        },
    ])
    summary.to_csv("results/model_comparison_metrics.csv", index=False)
    print("\n=== QUANTITATIVE MODEL COMPARISON ===")
    print(summary.to_string(index=False))

    # --- FIGURE 1: Parity Plot (Actual vs Predicted Heating & Cooling Loads) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    titles = ["Heating Load $Y_1$ (kWh/m$^2$)", "Cooling Load $Y_2$ (kWh/m$^2$)"]
    for i, ax in enumerate(axes):
        ax.scatter(Y_test[:, i], Y_pred_svd[:, i], alpha=0.6, label="SVD Baseline", marker="o")
        ax.scatter(Y_test[:, i], Y_pred_rf[:, i], alpha=0.7, label="Random Forest", marker="^")
        lims = [min(Y_test[:, i].min(), 5), max(Y_test[:, i].max(), 50)]
        ax.plot(lims, lims, "k--", lw=1.2, label="Ideal 1:1 Parity")
        ax.set_title(titles[i])
        ax.set_xlabel("Ground Truth Load (kWh/m$^2$)")
        ax.set_ylabel("Predicted Load (kWh/m$^2$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/fig1_parity_comparison.png")
    plt.close()

    # --- FIGURE 2: Sensor/Measurement Noise Sensitivity Analysis ---
    noise_levels = [0.0, 0.01, 0.03, 0.05, 0.10]
    svd_noise_rmse, rf_noise_rmse = [], []
    rng = np.random.default_rng(42)
    feat_std = np.std(X_test, axis=0)
    for sigma in noise_levels:
        perturbation = rng.normal(0.0, sigma * feat_std, size=X_test.shape)
        X_noisy = X_test + perturbation
        svd_noise_rmse.append(root_mean_squared_error(Y_test, svd_solver.predict(X_noisy)))
        rf_noise_rmse.append(root_mean_squared_error(Y_test, rf_model.predict(X_noisy)))

    plt.figure(figsize=(6.5, 4.2), dpi=150)
    plt.plot([n * 100 for n in noise_levels], svd_noise_rmse, "o-", label="SVD Baseline")
    plt.plot([n * 100 for n in noise_levels], rf_noise_rmse, "s-", label="Random Forest")
    plt.xlabel("Injected Gaussian Input Noise $\\sigma$ (% of Feature Std)")
    plt.ylabel("Test Set RMSE (kWh/m$^2$)")
    plt.title("Robustness & Sensitivity to Input Measurement Noise")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/fig2_noise_sensitivity.png")
    plt.close()

    # --- FIGURE 3: Hyperparameter Optimization Trade-Off (RMSE vs Runtime) ---
    tree_counts = [5, 10, 25, 50, 100, 150, 200]
    opt_rmse, opt_time = [], []
    for n_trees in tree_counts:
        m = RandomForestThermalModel(n_estimators=n_trees, max_depth=10)
        _, t_fit, _ = profile_execution(m.fit, X_train, Y_train)
        preds = m.predict(X_test)
        opt_rmse.append(root_mean_squared_error(Y_test, preds))
        opt_time.append(t_fit)

    fig, ax1 = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax2 = ax1.twinx()
    ax1.plot(tree_counts, opt_rmse, "b-o", label="Test RMSE (kWh/m$^2$)")
    ax2.plot(tree_counts, opt_time, "r--s", label="Training Runtime (ms)")
    ax1.axvline(100, color="gray", linestyle=":", label="Selected Design Point (T=100)")
    ax1.set_xlabel("Number of Ensemble Trees ($T$)")
    ax1.set_ylabel("Test RMSE (kWh/m$^2$)", color="b")
    ax2.set_ylabel("Training Runtime (ms)", color="r")
    plt.title("Pareto Optimization: Accuracy vs. Computational Runtime")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/fig3_optimization_tradeoff.png")
    plt.close()

    # --- FIGURE 4: Tikhonov Regularization Conditioning Curve ---
    lambdas = np.logspace(-12, 1, 30)
    conds, rmses = [], []
    for lam in lambdas:
        s = SVDRegularizedSolver(reg_lambda=lam).fit(X_train, Y_train)
        conds.append(s.effective_condition_number)
        rmses.append(root_mean_squared_error(Y_test, s.predict(X_test)))

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    ax1.loglog(lambdas, conds, "g-^")
    ax1.set_xlabel("Tikhonov Regularization Parameter $\\lambda$")
    ax1.set_ylabel("Effective Condition Number $\\kappa_{\\mathrm{eff}}(X)$", color="g")
    plt.title("Numerical Stability: Matrix Conditioning vs. Regularization $\\lambda$")
    ax1.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/fig4_numerical_conditioning.png")
    plt.close()
    print("\nAll 4 figures saved to results/figures/!")


if __name__ == "__main__":
    run_full_analysis()
