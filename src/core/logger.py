from pathlib import Path
import pandas as pd

from .config import OUTPUT_DIR


def save_metrics(model_name, metrics):

    OUTPUT_DIR.mkdir(exist_ok=True)

    csv_path = OUTPUT_DIR / "experiment_results.csv"

    row = {"Model": model_name}

    row.update(metrics)

    if csv_path.exists():

        df = pd.read_csv(csv_path)

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True,
        )

    else:

        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to:\n{csv_path}")