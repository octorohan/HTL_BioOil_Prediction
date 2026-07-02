import matplotlib.pyplot as plt
import pandas as pd

from src.core.io import save_dataframe, save_figure


def main():

    print("=" * 60)
    print("MODEL COMPARISON DASHBOARD")
    print("=" * 60)

    df = pd.read_csv("outputs/experiment_results.csv")

    # ----------------------------------------------------
    # Remove duplicate experiments (keep best Test_R2)
    # ----------------------------------------------------

    df = (
        df.sort_values("Test_R2", ascending=False)
          .drop_duplicates(subset=["Model"], keep="first")
          .reset_index(drop=True)
    )

    # ----------------------------------------------------
    # Leaderboard
    # ----------------------------------------------------

    leaderboard = df.sort_values(
        "Test_R2",
        ascending=False,
    ).reset_index(drop=True)

    leaderboard.index += 1

    print("\nLeaderboard")
    print("-" * 40)
    print(leaderboard)

    save_dataframe(
        leaderboard,
        "model_summary.csv",
    )

    # ----------------------------------------------------
    # Test R2
    # ----------------------------------------------------

    plt.figure(figsize=(10,6))

    plt.barh(
        leaderboard["Model"][::-1],
        leaderboard["Test_R2"][::-1],
    )

    plt.xlabel("Test R²")
    plt.title("Model Comparison (Test R²)")

    save_figure(
        plt,
        "model_r2_comparison.png",
    )

    plt.close()

    # ----------------------------------------------------
    # Test MAE
    # ----------------------------------------------------

    plt.figure(figsize=(10,6))

    plt.barh(
        leaderboard["Model"][::-1],
        leaderboard["Test_MAE"][::-1],
    )

    plt.xlabel("Test MAE")
    plt.title("Model Comparison (Test MAE)")

    save_figure(
        plt,
        "model_mae_comparison.png",
    )

    plt.close()

    # ----------------------------------------------------
    # Best Model
    # ----------------------------------------------------

    best = leaderboard.iloc[0]

    print("\nBest Model")
    print("-" * 40)

    print(best)


if __name__ == "__main__":
    main()