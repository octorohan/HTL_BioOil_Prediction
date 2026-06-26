import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df[df["Oil"].notna()].copy()


# ==================================================
# FEATURES / TARGET
# ==================================================

drop_cols = [
    "Oil",
    "Biocrude",
    "Gas",
    "Char",
    "WaterPhase",
    "Ref",
    "Author"
]

X = df.drop(columns=drop_cols)
y = df["Oil"]


# ==================================================
# COLUMN TYPES
# ==================================================

cat_cols = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

num_cols = X.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()


# ==================================================
# PREPROCESSING
# ==================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


# ==================================================
# RANDOM FOREST
# ==================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])


# ==================================================
# 3-FOLD CROSS VALIDATION
# ==================================================

scores = cross_val_score(
    pipe,
    X,
    y,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

print("\nRANDOM FOREST 5-FOLD CV")
print("=" * 50)

print("Fold Scores:")
print(scores)

print("\nMean CV R2:")
print(scores.mean())

print("\nStd CV R2:")
print(scores.std())