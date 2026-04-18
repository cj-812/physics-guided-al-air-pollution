import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import config
from data_generator import compute_concentrations, preprocess_X, N_SOURCES
from neural_model import build_model, predict_with_uncertainty

# Set font to avoid display issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def save_results_to_csv(results, save_path):
    df = pd.DataFrame(results, columns=["Method", "Samples", "RMSE", "R²"])
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\n[INFO] Results saved to: {save_path}")
    print("Results preview:")
    print(df)


def plot_figure_1_learning_curve(results, save_path):
    df = pd.DataFrame(results, columns=["Method", "Samples", "RMSE", "R²"])
    methods = df["Method"].unique()
    colors = {"Random": "tab:blue", "Active Learning": "tab:red"}
    markers = {"Random": "o", "Active Learning": "s"}

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Number of Samples", fontsize=12)
    ax1.set_ylabel("RMSE", fontsize=12, color=colors["Random"])
    for method in methods:
        method_data = df[df["Method"] == method]
        ax1.plot(method_data["Samples"], method_data["RMSE"], label=f"{method} (RMSE)",
                 color=colors[method], marker=markers[method], linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=colors["Random"])
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("R²", fontsize=12, color=colors["Active Learning"])
    for method in methods:
        method_data = df[df["Method"] == method]
        ax2.plot(method_data["Samples"], method_data["R²"], label=f"{method} (R²)",
                 color=colors[method], marker=markers[method], linewidth=2, linestyle="--", markersize=8)
    ax2.tick_params(axis='y', labelcolor=colors["Active Learning"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Figure 1. Learning Curve: Active Learning vs Random Sampling", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figure 1 saved to: {save_path}")


def plot_figure_2_sampling_distribution(pool_X, random_idx, al_idx, al_uncertainty=None, save_path=None):
    x_col = 2
    random_x = pool_X[random_idx, x_col]
    al_x = pool_X[al_idx, x_col]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(random_x, np.random.randn(len(random_x)), alpha=0.5, color="tab:blue", label="Random")
    ax1.set_title("Random Sampling (800 samples) - Downwind Distance x", fontsize=12)
    ax1.set_xlabel("Downwind Distance x (m)", fontsize=10)
    ax1.set_ylabel("Random Jitter (Visualization Only)", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.legend()

    if al_uncertainty is not None:
        sc = ax2.scatter(al_x, np.random.randn(len(al_x)), c=al_uncertainty, cmap="Reds", alpha=0.7,
                         label="Active Learning")
        plt.colorbar(sc, ax=ax2, label="Uncertainty (Std Dev)")
    else:
        ax2.scatter(al_x, np.random.randn(len(al_x)), alpha=0.5, color="tab:red", label="Active Learning")
    ax2.set_title("Active Learning (800 samples) - Downwind Distance x", fontsize=12)
    ax2.set_xlabel("Downwind Distance x (m)", fontsize=10)
    ax2.set_ylabel("Random Jitter (Visualization Only)", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.suptitle("Figure 2. Sampling Distribution Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure 2 saved to: {save_path}")
    plt.close()


def plot_figure_3_physics_consistency(model, save_path):
    base = [25, 2.0, 200, 0, 50] * N_SOURCES
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    qs = np.linspace(10, 100, 40)
    p_q, nn_q = [], []
    for q in qs:
        s = base.copy()
        s[0] = q
        s = np.array([s])
        p_q.append(compute_concentrations(s)[0])
        s_p = preprocess_X(s)
        nn_q.append(model(s_p, training=False).numpy()[0, 0])
    axes[0].plot(qs, p_q, 'b-', linewidth=2.5, label='Physics Ground Truth')
    axes[0].plot(qs, nn_q, 'r--', linewidth=2.5, label='Active Learning Model')
    axes[0].set_xlabel('Source Strength Q (g/s)', fontsize=12)
    axes[0].set_ylabel('Concentration (ppb)', fontsize=12)
    axes[0].set_title('Q vs Concentration', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    us = np.linspace(1, 5, 40)
    p_u, nn_u = [], []
    for u in us:
        s = base.copy()
        s[1] = u
        s = np.array([s])
        p_u.append(compute_concentrations(s)[0])
        s_p = preprocess_X(s)
        nn_u.append(model(s_p, training=False).numpy()[0, 0])
    axes[1].plot(us, p_u, 'b-', linewidth=2.5, label='Physics Ground Truth')
    axes[1].plot(us, nn_u, 'r--', linewidth=2.5, label='Active Learning Model')
    axes[1].set_xlabel('Wind Speed u (m/s)', fontsize=12)
    axes[1].set_ylabel('Concentration (ppb)', fontsize=12)
    axes[1].set_title('u vs Concentration', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    xs = np.linspace(50, 500, 40)
    p_x, nn_x = [], []
    for x in xs:
        s = base.copy()
        s[2] = x
        s = np.array([s])
        p_x.append(compute_concentrations(s)[0])
        s_p = preprocess_X(s)
        nn_x.append(model(s_p, training=False).numpy()[0, 0])
    axes[2].plot(xs, p_x, 'b-', linewidth=2.5, label='Physics Ground Truth')
    axes[2].plot(xs, nn_x, 'r--', linewidth=2.5, label='Active Learning Model')
    axes[2].set_xlabel('Downwind Distance x (m)', fontsize=12)
    axes[2].set_ylabel('Concentration (ppb)', fontsize=12)
    axes[2].set_title('x vs Concentration', fontsize=12)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle('Figure 3. Physical Consistency: Physics vs Active Learning Model', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figure 3 saved to: {save_path}")


def plot_figure_4_spatial_profile(model, pool_X, pool_y, save_path):
    xs = np.linspace(50, 500, 50)
    c_phys = []
    c_al = []
    c_random = []

    base = [25, 2.0, 200, 0, 50] * N_SOURCES

    for x in xs:
        s = base.copy()
        s[2] = x
        s = np.array([s])
        c_phys.append(compute_concentrations(s)[0])
        s_p = preprocess_X(s)
        c_al.append(model(s_p, training=False).numpy()[0, 0])

    # ✅ 修复：使用传入的全局pool，不再重新生成
    random_idx = np.random.choice(len(pool_X), size=800, replace=False)
    random_X_p = preprocess_X(pool_X[random_idx])
    random_y = pool_y[random_idx]
    random_model, random_norm = build_model()
    random_norm.adapt(random_X_p)
    random_model.fit(random_X_p, random_y, epochs=config.TRAIN_CONFIG["initial_epochs"],
                     batch_size=config.TRAIN_CONFIG["batch_size"], verbose=0, validation_split=0.1)
    for x in xs:
        s = base.copy()
        s[2] = x
        s = np.array([s])
        s_p = preprocess_X(s)
        c_random.append(random_model(s_p, training=False).numpy()[0, 0])

    plt.figure(figsize=(10, 6))
    plt.plot(xs, c_phys, 'b-', linewidth=2.5, label='Physics Ground Truth')
    plt.plot(xs, c_al, 'r--', linewidth=2.5, label='Active Learning (800 samples)')
    plt.plot(xs, c_random, 'g:', linewidth=2.5, label='Random (800 samples)')
    plt.xlabel('Downwind Distance x (m)', fontsize=12)
    plt.ylabel('Concentration (ppb)', fontsize=12)
    plt.title('Figure 4. Spatial Concentration Profile (y=0, z=50)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figure 4 saved to: {save_path}")


def plot_figure_5_scatter_comparison(model_al_final, X_test, y_test, pool_X, pool_y, save_path):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    colors = {
        "Random300": "#1f77b4",
        "Random500": "#2ca02c",
        "Random800": "#9467bd",
        "Random1000": "#ff7f0e",
        "AL300": "#d62728",
        "AL500": "#e377c2",
        "AL800": "#8c564b",
        "AL1000": "#17becf"
    }

    methods = [
        "Random300", "AL300",
        "Random500", "AL500",
        "Random800", "AL800",
        "Random1000", "AL1000"
    ]

    X_test_p = preprocess_X(X_test)
    # ✅ 修复：使用传入的全局pool，不再重新生成
    pool_X_p = preprocess_X(pool_X)

    print("[INFO] Generating scatter plots for Random sampling...")
    for i, budget in enumerate([300, 500, 800, 1000]):
        ax_idx = i * 2
        random_idx = np.random.choice(len(pool_X), size=budget, replace=False)
        random_X_p = pool_X_p[random_idx]
        random_y = pool_y[random_idx]
        m, norm = build_model()
        norm.adapt(random_X_p)
        m.fit(random_X_p, random_y, epochs=config.TRAIN_CONFIG["initial_epochs"],
              batch_size=config.TRAIN_CONFIG["batch_size"], verbose=0, validation_split=0.1)
        y_pred = m.predict(X_test_p, verbose=0).flatten()
        axes[ax_idx].scatter(y_test, y_pred, s=8, alpha=0.5, color=colors[methods[ax_idx]])
        axes[ax_idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes[ax_idx].set_title(methods[ax_idx], fontsize=14, fontweight='bold')
        axes[ax_idx].set_xlabel("True Concentration (ppb)", fontsize=12)
        axes[ax_idx].set_ylabel("Predicted Concentration (ppb)", fontsize=12)
        axes[ax_idx].grid(alpha=0.3)
        axes[ax_idx].set_xlim(y_test.min() - 10, y_test.max() + 10)
        axes[ax_idx].set_ylim(y_test.min() - 10, y_test.max() + 10)

    print("[INFO] Generating scatter plots for Active Learning...")
    n_init = config.AL_CONFIG["init_size"]
    pool_size = len(pool_X)
    init_idx = np.random.choice(pool_size, size=n_init, replace=False)
    train_idx = init_idx
    target_budgets = [300, 500, 800, 1000]
    current_target_idx = 0
    al_models = {}

    while len(train_idx) <= max(target_budgets):
        model, normalizer = build_model()
        current_X_p = pool_X_p[train_idx]
        current_y = pool_y[train_idx]
        normalizer.adapt(current_X_p)
        model.fit(current_X_p, current_y, epochs=config.TRAIN_CONFIG["initial_epochs"],
                  batch_size=config.TRAIN_CONFIG["batch_size"], verbose=0, validation_split=0.1)

        while current_target_idx < len(target_budgets) and len(train_idx) >= target_budgets[current_target_idx]:
            target = target_budgets[current_target_idx]
            al_models[target] = model
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

    for i, budget in enumerate([300, 500, 800, 1000]):
        ax_idx = i * 2 + 1
        y_pred = al_models[budget].predict(X_test_p, verbose=0).flatten()
        axes[ax_idx].scatter(y_test, y_pred, s=8, alpha=0.5, color=colors[methods[ax_idx]])
        axes[ax_idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes[ax_idx].set_title(methods[ax_idx], fontsize=14, fontweight='bold')
        axes[ax_idx].set_xlabel("True Concentration (ppb)", fontsize=12)
        axes[ax_idx].set_ylabel("Predicted Concentration (ppb)", fontsize=12)
        axes[ax_idx].grid(alpha=0.3)
        axes[ax_idx].set_xlim(y_test.min() - 10, y_test.max() + 10)
        axes[ax_idx].set_ylim(y_test.min() - 10, y_test.max() + 10)

    plt.suptitle('Figure 5. Predicted vs True Concentration Scatter Plots (Vertical Alignment)', fontsize=18, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figure 5 saved to: {save_path}")


def plot_figure_6_rmse_r2_bar(results, save_path):
    df = pd.DataFrame(results, columns=["Method", "Samples", "RMSE", "R²"])
    samples_list = [300, 500, 800, 1000]
    methods = ["Random", "Active Learning"]
    colors = {"Random": "tab:blue", "Active Learning": "tab:red"}

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(len(samples_list))

    for i, method in enumerate(methods):
        rmse_values = [df[(df["Method"] == method) & (df["Samples"] == s)]["RMSE"].values[0] for s in samples_list]
        ax1.bar(x + i * bar_width, rmse_values, bar_width, label=f"{method} (RMSE)", color=colors[method], alpha=0.6)
    ax1.set_xlabel("Number of Samples", fontsize=12)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_xticks(x + bar_width / 2)
    ax1.set_xticklabels(samples_list)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    for i, method in enumerate(methods):
        r2_values = [df[(df["Method"] == method) & (df["Samples"] == s)]["R²"].values[0] for s in samples_list]
        ax2.plot(x + i * bar_width, r2_values, label=f"{method} (R²)", color=colors[method],
                 marker='o', linewidth=2, markersize=8)
    ax2.set_ylabel("R²", fontsize=12)
    ax2.legend(loc="upper right")

    plt.title("Figure 6. RMSE & R² Comparison by Sample Size", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figure 6 saved to: {save_path}")


def plot_figure_7_uncertainty_distribution(pool_X, pool_y, al_idx, model, save_path):
    pool_X_p = preprocess_X(pool_X)
    _, std = predict_with_uncertainty(model, pool_X_p)
    std_np = std.numpy().flatten()
    al_std = std_np[al_idx]

    plt.figure(figsize=(10, 6))
    plt.hist(std_np, bins=30, alpha=0.5, label="All Pool Samples", color="tab:blue")
    plt.hist(al_std, bins=30, alpha=0.7, label="AL Selected Samples", color="tab:red")
    plt.xlabel("Uncertainty (Standard Deviation)", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Figure 7. Uncertainty Distribution Comparison", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figure 7 saved to: {save_path}")