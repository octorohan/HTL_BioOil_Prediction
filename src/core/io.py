from pathlib import Path
import joblib

from .config import (
    OUTPUT_DIR,
    FIGURE_DIR,
    MODEL_DIR,
)

import pandas as pd


def save_dataframe(df, filename):

    OUTPUT_DIR.mkdir(exist_ok=True)

    path = OUTPUT_DIR / filename

    df.to_csv(path, index=False)

    print(f"Saved:\n{path}")


def save_model(model, filename):

    MODEL_DIR.mkdir(exist_ok=True)

    path = MODEL_DIR / filename

    joblib.dump(model, path)

    print(f"Saved:\n{path}")


def save_figure(plt, filename):

    FIGURE_DIR.mkdir(exist_ok=True)

    path = FIGURE_DIR / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved:\n{path}")

    
def load_model(filename):

    path = MODEL_DIR / filename

    model = joblib.load(path)

    print(f"Loaded:\n{path}")

    return model    