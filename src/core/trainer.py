from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE
from .config import TEST_SIZE

from .metrics import evaluate_model
from .preprocessing import build_preprocessor
from .results import create_results_dataframe

def train_model(model, X, y):

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

    return (
        pipeline,
        metrics,
        prediction_df,
    )