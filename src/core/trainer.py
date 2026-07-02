from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE
from .metrics import evaluate_model
from .preprocessing import build_preprocessor
from .results import create_results_dataframe
from .io import save_model, save_dataframe


def train_model(
    model,
    X,
    y,
    model_name=None,
):
    """
    Generic training function for all models.
    """

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    metrics = evaluate_model(
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    prediction_df = create_results_dataframe(
        y_train,
        train_pred,
        y_test,
        test_pred,
    )

    # ------------------------------------------------------
    # Automatically save outputs
    # ------------------------------------------------------
    if model_name is not None:

        save_model(
            pipeline,
            f"{model_name}.pkl",
        )

        save_dataframe(
            prediction_df,
            f"{model_name}_predictions.csv",
        )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "predictions": prediction_df,
    }