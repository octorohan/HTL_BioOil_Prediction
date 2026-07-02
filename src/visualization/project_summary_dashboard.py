import matplotlib.pyplot as plt
import pandas as pd

from src.core.io import save_figure


def main():

    print("=" * 60)
    print("PROJECT SUMMARY DASHBOARD")
    print("=" * 60)

    # ----------------------------------------------------
    # Load Data
    # ----------------------------------------------------

    leaderboard = pd.read_csv("outputs/model_summary.csv")

    importance = pd.read_csv("outputs/permutation_importance.csv")

    best = leaderboard.iloc[0]

    top_features = importance.head(5)

    # ----------------------------------------------------
    # Figure
    # ----------------------------------------------------

    fig = plt.figure(figsize=(16, 10))

    fig.suptitle(
        "HTL Bio-Oil Yield Prediction using Machine Learning",
        fontsize=22,
        fontweight="bold",
    )

    # ====================================================
    # Dataset Summary
    # ====================================================

    ax1 = plt.subplot(2, 2, 1)
    ax1.axis("off")

    ax1.set_title(
        "Dataset Summary",
        fontsize=16,
        fontweight="bold",
    )

    dataset_text = (
        "Samples : 2284\n\n"
        "Features : 29\n\n"
        "Target : Bio-Oil Yield (%)\n\n"
        "Algorithms Evaluated : 6\n\n"
        "Explainability :\n"
        "• SHAP\n"
        "• PDP\n"
        "• Permutation Importance\n"
        "• Feature Importance"
    )

    ax1.text(
        0,
        1,
        dataset_text,
        fontsize=13,
        va="top",
    )

    # ====================================================
    # Best Model
    # ====================================================

    ax2 = plt.subplot(2, 2, 2)
    ax2.axis("off")

    ax2.set_title(
        "Best Model",
        fontsize=16,
        fontweight="bold",
    )

    best_text = (
        f"Model : {best['Model']}\n\n"
        f"Test R² : {best['Test_R2']:.4f}\n\n"
        f"Test MAE : {best['Test_MAE']:.4f}\n\n"
        f"Train R² : {best['Train_R2']:.4f}\n\n"
        f"Train MAE : {best['Train_MAE']:.4f}"
    )

    ax2.text(
        0,
        1,
        best_text,
        fontsize=13,
        va="top",
    )

    # ====================================================
    # Model Ranking
    # ====================================================

    ax3 = plt.subplot(2, 2, 3)

    ax3.barh(
        leaderboard["Model"][::-1],
        leaderboard["Test_R2"][::-1],
    )

    ax3.set_title(
        "Model Ranking (Test R²)",
        fontsize=15,
        fontweight="bold",
    )

    ax3.set_xlabel("Test R²")

    # ====================================================
    # Top Features
    # ====================================================

    ax4 = plt.subplot(2, 2, 4)

    ax4.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1],
    )

    ax4.set_title(
        "Top Features",
        fontsize=15,
        fontweight="bold",
    )

    ax4.set_xlabel("Permutation Importance")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(
        plt,
        "project_summary_dashboard.png",
    )

    plt.close()

    print("\nSaved:")
    print("outputs/figures/project_summary_dashboard.png")


if __name__ == "__main__":
    main()