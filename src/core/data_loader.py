import pandas as pd

from .config import DATASET_PATH
from .config import DROP_COLUMNS
from .config import TARGET_COLUMN


def load_dataset():

    df = pd.read_excel(DATASET_PATH)

    df = df[df[TARGET_COLUMN].notna()].copy()

    X = df.drop(columns=DROP_COLUMNS)

    y = df[TARGET_COLUMN]

    return X, y