import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_generator import generate_samples, compute_concentrations, preprocess_X
from neural_model import build_model, predict_with_uncertainty
from evaluate import (
    save_results_to_csv,
    plot_figure_1_learning_curve,
    plot_figure_2_sampling_distribution,
    plot_figure_3_physics_consistency,
    plot_figure_4_spatial_profile,
    plot_figure_5_scatter_comparison,
    plot_figure_6_rmse_r2_bar,
    plot_figure_7_uncertainty_distribution
)
import config


def evaluate_model(model, X_test_p, y_test):
    y_pred = model.predict(X_test_p, verbose=0).flatten()
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    return rmse, r2


def main():
    print("=" * 60)
    print("Air Pollution Diffusion Model - Full Active Learning Experiment")
    print("=" * 60)

    print("\n[Step 1] Generating fixed dataset (seed=42)...")
    total_X = generate_samples(config.DATA_CONFIG["total_samples"])
    total_y = compute_concentrations(total_X)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        total_X, total_y,
        test_size=config.DATA_CONFIG["test_size"],
        random_state=config.DATA_CONFIG["random_state"]
    )
    X_test_p = preprocess_X(X_test)
    pool_X = X_train_full
    pool_y = y_train_full
    pool_X_p = preprocess_X(pool_X)
    print(f"  - Total samples: {config.DATA_CONFIG['total_samples']}")
    print(f"  - Training pool: {len(X_train_full)} samples")
    print(f"  - Fixed test set: {len(X_test)} samples")

    print("\n[Step 2] Running Random Sampling experiments...")
    all_results = []
    for budget in [300, 500, 800, 1000]:
        print(f"  - Random {budget} samples...")
        random_idx = np.random.choice(len(pool_X), size=budget, replace=False)
        random_X_p = pool_X_p[random_idx]
        random_y = pool_y[random_idx]
        m, norm = build_model()
        norm.adapt(random_X_p)
        m.fit(random_X_p, random_y, epochs=config.TRAIN_CONFIG["initial_epochs"],
              batch_size=config.TRAIN_CONFIG["batch_size"], verbose=0, validation_split=0.1)
        rmse, r2 = evaluate_model(m, X_test_p, y_test)
        all_results.append(("Random", budget, rmse, r2))
        print(f"    - RMSE: {rmse:.4f}, R²: {r2:.4f}")

    print("\n[Step 3] Running Active Learning experiments...")
    n_init = config.AL_CONFIG["init_size"]
    pool_size = len(pool_X)
    init_idx = np.random.choice(pool_size, size=n_init, replace=False)
    train_idx = init_idx
    target_budgets = [300, 500, 800, 1000]
    current_target_idx = 0
    al_models_dict = {}
    al_indices_dict = {}

    while len(train_idx) <= max(target_budgets):
        print(f"  - Current AL training size: {len(train_idx)}...")
        model, normalizer = build_model()
        current_X_p = pool_X_p[train_idx]
        current_y = pool_y[train_idx]
        normalizer.adapt(current_X_p)
        model.fit(current_X_p, current_y, epochs=config.TRAIN_CONFIG["initial_epochs"],
                  batch_size=config.TRAIN_CONFIG["batch_size"], verbose=0, validation_split=0.1)

        while current_target_idx < len(target_budgets) and len(train_idx) >= target_budgets[current_target_idx]:
            target = target_budgets[current_target_idx]
            rmse, r2 = evaluate_model(model, X_test_p, y_test)
            all_results.append(("Active Learning", target, rmse, r2))
            al_models_dict[target] = model
            al_indices_dict[target] = train_idx.copy()
            print(f"    - AL {target} samples: RMSE={rmse:.4f}, R²={r2:.4f}")
            current_target_idx += 1

        if current_target_idx >= len(target_budgets):
            break

        unselected_idx = np.setdiff1d(np.arange(pool_size), train_idx)
        unselected_X_p = pool_X_p[unselected_idx]
        _, std = predict_with_uncertainty(model, unselected_X_p)
        std_np = std.numpy().flatten()
        n_select = min(config.AL_CONFIG["step_size"], max(target_budgets) - len(train_idx))
        top_idx = np.argsort(std_np)[-n_select:]
        selected_unselected_idx = unselected_idx[top_idx]
        train_idx = np.hstack([train_idx, selected_unselected_idx])

    print("\n[Step 4] Saving results and models...")
    csv_path = os.path.join(config.RESULTS_DIR, "results.csv")
    save_results_to_csv(all_results, csv_path)

    al_final_model = al_models_dict[1000]
    al_final_path = os.path.join(config.MODELS_DIR, "al_final_model.keras")
    al_final_model.save(al_final_path)
    print(f"[INFO] AL final model (1000 samples) saved to: {al_final_path}")

    al_800_model = al_models_dict[800]
    al_800_path = os.path.join(config.MODELS_DIR, "al_800_model.keras")
    al_800_model.save(al_800_path)
    print(f"[INFO] AL 800 model saved to: {al_800_path}")

    print("\n[Step 5] Generating all figures...")
    plot_figure_1_learning_curve(all_results, os.path.join(config.RESULTS_DIR, "figure_1_learning_curve.png"))

    al_800_idx = al_indices_dict[800]
    random_800_idx = np.random.choice(len(pool_X), size=800, replace=False)
    _, al_800_std = predict_with_uncertainty(al_800_model, pool_X_p[al_800_idx])
    al_800_uncertainty = al_800_std.numpy().flatten()
    plot_figure_2_sampling_distribution(pool_X, random_800_idx, al_800_idx, al_800_uncertainty,
                                        os.path.join(config.RESULTS_DIR, "figure_2_sampling_distribution.png"))

    plot_figure_3_physics_consistency(al_800_model,
                                      os.path.join(config.RESULTS_DIR, "figure_3_physics_consistency.png"))
    plot_figure_4_spatial_profile(al_800_model, pool_X, pool_y,
                                  os.path.join(config.RESULTS_DIR, "figure_4_spatial_profile.png"))
    plot_figure_5_scatter_comparison(al_final_model, X_test, y_test, pool_X, pool_y,
                                     os.path.join(config.RESULTS_DIR, "figure_5_scatter_comparison.png"))
    plot_figure_6_rmse_r2_bar(all_results, os.path.join(config.RESULTS_DIR, "figure_6_rmse_r2_bar.png"))
    plot_figure_7_uncertainty_distribution(pool_X, pool_y, al_800_idx, al_800_model,
                                           os.path.join(config.RESULTS_DIR, "figure_7_uncertainty_distribution.png"))

    print("\n" + "=" * 60)
    print("Experiment Complete! All outputs saved.")
    print("=" * 60)
    print("\nOutput Files:")
    print("  1. results.csv - Full experiment results")
    print("  2. al_final_model.keras - AL final model (1000 samples)")
    print("  3. al_800_model.keras - AL 800 model")
    print("  4. figure_1_learning_curve.png - Learning curve")
    print("  5. figure_2_sampling_distribution.png - Sampling distribution")
    print("  6. figure_3_physics_consistency.png - Physical consistency")
    print("  7. figure_4_spatial_profile.png - Spatial profile")
    print("  8. figure_5_scatter_comparison.png - Scatter plots (vertical)")
    print("  9. figure_6_rmse_r2_bar.png - RMSE/R² bar chart")
    print(" 10. figure_7_uncertainty_distribution.png - Uncertainty distribution")


if __name__ == "__main__":
    main()