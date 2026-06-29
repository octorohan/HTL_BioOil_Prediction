import pandas as pd

from .io import save_dataframe


def create_results_dataframe(
    y_train,
    train_pred,
    y_test,
    test_pred,
):

    train_df = pd.DataFrame({
        "Dataset": "Train",
        "Actual": y_train.values,
        "Prediction": train_pred,
    })

    test_df = pd.DataFrame({
        "Dataset": "Test",
        "Actual": y_test.values,
        "Prediction": test_pred,
    })

    return pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )


def save_results(df):

    save_dataframe(
        df,
        "predictions.csv",
    )