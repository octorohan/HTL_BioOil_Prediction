from src.core.data_loader import load_dataset
from src.core.model_factory import build_catboost
from src.core.trainer import train_model
from src.core.logger import save_metrics


def main():

    X, y = load_dataset()

    model = build_catboost()

    results = train_model(
        model=model,
        X=X,
        y=y,
        model_name="catboost",
    )

    metrics = results["metrics"]

    print()
    print("=" * 40)
    print("CatBoost")
    print("=" * 40)

    for key, value in metrics.items():
        print(f"{key:<12}: {value:.4f}")

    save_metrics(
        "CatBoost",
        metrics,
    )


if __name__ == "__main__":
    main()