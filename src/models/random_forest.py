from src.core.data_loader import load_dataset
from src.core.model_factory import build_random_forest
from src.core.trainer import train_model
from src.core.logger import save_metrics

def main():

    X, y = load_dataset()

    model = build_random_forest()

    pipeline, metrics = train_model(
        model,
        X,
        y,
    )

    print()

    print("=" * 40)
    print("Random Forest")
    print("=" * 40)

    for k, v in metrics.items():
        print(f"{k:<12}: {v:.4f}")

    save_metrics(
        "Random Forest",
        metrics,
    )

if __name__ == "__main__":
    main()