import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bbb2637-sup-0001-datas1.xlsx"
)

FIG_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "figures"
)

os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(DATA_PATH)

df = df.drop(columns=["Author", "Ref"], errors="ignore")
df = df.dropna(subset=["Oil"])

y = df["Oil"]

X = df.drop(
    columns=[
        "Oil",
        "Biocrude",
        "Char",
        "Gas",
        "WaterPhase"
    ]
)

# ============================================================
# FEATURE TYPES
# ============================================================

cat_cols = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

num_cols = X.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()

print("Categorical:")
print(cat_cols)

print("\nNumerical:")
print(num_cols)

# ============================================================
# PREPROCESSOR
# ============================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# ============================================================
# TRANSFORM
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

feature_names = [
    f.replace("num__", "").replace("cat__", "")
    for f in preprocessor.get_feature_names_out()
]

# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train_processed,
    y_train
)

print("\nRandom Forest trained.")

# ============================================================
# FEATURES TO PLOT
# ============================================================

important_features = [
    "Lipids",
    "Temperature",
    "Proteins",
    "CarboHydrates",
    "HHVResource"
]

# ============================================================
# PDP
# ============================================================

for feat in important_features:

    if feat not in feature_names:
        print(f"{feat} not found")
        continue

    idx = feature_names.index(feat)

    fig, ax = plt.subplots(figsize=(6,5))

    PartialDependenceDisplay.from_estimator(
        rf,
        X_train_processed,
        [idx],
        feature_names=feature_names,
        ax=ax
    )

    plt.tight_layout()

    save_path = os.path.join(
        FIG_DIR,
        f"pdp_{feat}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)

print("\nDone.")