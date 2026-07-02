from src.core.data_loader import load_dataset
from src.core.model_factory import build_extra_trees
from src.core.trainer import train_model
from src.core.logger import save_metrics


def main():

    X, y = load_dataset()

    model = build_extra_trees()

    results = train_model(
        model=model,
        X=X,
        y=y,
        model_name="extra_trees",
    )

    metrics = results["metrics"]

    print()
    print("=" * 40)
    print("Extra Trees")
    print("=" * 40)

    for key, value in metrics.items():
        print(f"{key:<12}: {value:.4f}")

    save_metrics(
        "Extra Trees",
        metrics,
    )


if __name__ == "__main__":
    main()