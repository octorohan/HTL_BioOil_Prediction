from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE
from .config import TEST_SIZE

from .metrics import evaluate_model
from .preprocessing import build_preprocessor


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

    metrics = evaluate_model(
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    return pipeline, metrics