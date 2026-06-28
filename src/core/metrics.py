from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
)


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    return {
        "Train_R2": r2_score(
            y_train,
            train_pred,
        ),
        "Test_R2": r2_score(
            y_test,
            test_pred,
        ),
        "Train_MAE": mean_absolute_error(
            y_train,
            train_pred,
        ),
        "Test_MAE": mean_absolute_error(
            y_test,
            test_pred,
        ),
    }