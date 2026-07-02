import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance

from src.core.data_loader import load_dataset
from src.core.io import (
    load_model,
    save_dataframe,
    save_figure,
)


def main():

    print("=" * 60)
    print("PERMUTATION IMPORTANCE")
    print("=" * 60)

    X, y = load_dataset()

    pipeline = load_model("xgboost_tuned.pkl")

    print("\nComputing permutation importance...")

    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="r2",
    )

    feature_names = X.columns

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    )

    save_dataframe(
        importance_df,
        "permutation_importance.csv",
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance_df["Feature"][:20][::-1],
        importance_df["Importance"][:20][::-1],
    )

    plt.xlabel("Permutation Importance")

    plt.title("Top 20 Features (Permutation Importance)")

    save_figure(
        plt,
        "permutation_importance.png",
    )

    plt.close()

    print()
    print("Top 20 Features")
    print("-" * 40)

    print(importance_df.head(20))


if __name__ == "__main__":
    main()