import time

import pandas as pd

from .logger import save_metrics

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    RepeatedKFold,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
)

from .config import RANDOM_STATE
from .config import TEST_SIZE
from .preprocessing import build_preprocessor
from .io import save_model, save_dataframe


def tune_model(
    model,
    param_distributions,
    X,
    y,
    model_name,
    n_iter=30,
):

    print()
    print("=" * 60)
    print(f"Tuning {model_name}")
    print("=" * 60)

    # =====================================================
    # Train Test Split
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # =====================================================
    # Pipeline
    # =====================================================

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    # =====================================================
    # Cross Validation
    # =====================================================

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=RANDOM_STATE,
    )

    # =====================================================
    # Random Search
    # =====================================================

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )

    print("\nStarting RandomizedSearchCV...\n")

    start = time.time()

    search.fit(X_train, y_train)

    elapsed = time.time() - start

    print("\nSearch Completed")
    print(f"Time : {elapsed / 60:.2f} minutes")

    # =====================================================
    # Best Model
    # =====================================================

    best_pipeline = search.best_estimator_

    print("\nBest Parameters")
    print("-" * 40)

    for key, value in search.best_params_.items():
        print(f"{key:<35}: {value}")

    print("\nBest CV R2 :", round(search.best_score_, 4))

    # =====================================================
    # Predictions
    # =====================================================

    train_pred = best_pipeline.predict(X_train)
    test_pred = best_pipeline.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    metrics = {
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Train_MAE": train_mae,
        "Test_MAE": test_mae,
    }

    print("\nFinal Performance")
    print("-" * 40)

    for key, value in metrics.items():
        print(f"{key:<12}: {value:.4f}")

    # =====================================================
    # Prediction Data
    # =====================================================

    prediction_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": test_pred,
        }
    )

    # =====================================================
    # CV Results
    # =====================================================

    cv_results = pd.DataFrame(search.cv_results_)

    # =====================================================
    # Save Outputs
    # =====================================================

    save_model(
        best_pipeline,
        f"{model_name}.pkl",
    )

    save_dataframe(
        prediction_df,
        f"{model_name}_predictions.csv",
    )

    save_dataframe(
        cv_results,
        f"{model_name}_cv_results.csv",
    )

    summary = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Train_R2": train_r2,
                "Test_R2": test_r2,
                "Train_MAE": train_mae,
                "Test_MAE": test_mae,
                "Best_CV_R2": search.best_score_,
                "Time_Minutes": elapsed / 60,
            }
        ]
    )

    save_dataframe(
        summary,
        f"{model_name}_summary.csv",
    )

    # =====================================================
    # Save to experiment_results.csv
    # =====================================================

    save_metrics(
        model_name,
        metrics,
    )

    print("\nSaved Outputs")
    print("-" * 40)
    print(f"Model        : outputs/models/{model_name}.pkl")
    print(f"Predictions  : outputs/{model_name}_predictions.csv")
    print(f"CV Results   : outputs/{model_name}_cv_results.csv")
    print(f"Summary      : outputs/{model_name}_summary.csv")

    return {
        "pipeline": best_pipeline,
        "metrics": metrics,
        "best_params": search.best_params_,
        "cv_results": cv_results,
        "predictions": prediction_df,
    }