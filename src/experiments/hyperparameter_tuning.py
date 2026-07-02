from src.core.data_loader import load_dataset
from src.core.hyperparameter_search import tune_model
from src.core.model_factory import build_xgboost


def main():

    X, y = load_dataset()

    param_distributions = {

        "model__n_estimators": [
            300,
            500,
            800,
            1000,
        ],

        "model__learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
        ],

        "model__max_depth": [
            4,
            6,
            8,
            10,
        ],

        "model__subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "model__colsample_bytree": [
            0.6,
            0.8,
            1.0,
        ],

        "model__min_child_weight": [
            1,
            3,
            5,
        ],

        "model__gamma": [
            0,
            0.1,
            0.3,
            0.5,
        ],

    }

    results = tune_model(
        model=build_xgboost(),
        param_distributions=param_distributions,
        X=X,
        y=y,
        model_name="xgboost_tuned",
        n_iter=30,
    )

    print()
    print("=" * 60)
    print("TUNED XGBOOST")
    print("=" * 60)

    for k, v in results["metrics"].items():

        if isinstance(v, float):
            print(f"{k:<20}: {v:.4f}")
        else:
            print(f"{k:<20}: {v}")


if __name__ == "__main__":
    main()