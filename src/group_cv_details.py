import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df[df["Oil"].notna()].copy()

print("Rows:", len(df))
print("Unique Feedstocks (Details):", df["Details"].nunique())


# ==================================================
# TARGET
# ==================================================

y = df["Oil"]


# ==================================================
# FEATURES
# ==================================================

drop_cols = [
    "Oil",
    "Biocrude",
    "Gas",
    "Char",
    "WaterPhase",
    "Ref",
    "Author",

    # IMPORTANT
    # remove feedstock identifier
    "Details"
]

X = df.drop(columns=drop_cols)


# ==================================================
# GROUPS
# ==================================================

groups = df["Details"]


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
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


# ==================================================
# MODEL
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
# GROUP K FOLD
# ==================================================

gkf = GroupKFold(n_splits=5)

scores = []

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    r2 = r2_score(y_test, preds)

    scores.append(r2)

    print(f"Fold {fold}: {r2:.4f}")


print("\n" + "=" * 50)
print("GROUP K-FOLD RESULTS")
print("=" * 50)

print("Scores:", scores)
print("Mean R2:", np.mean(scores))
print("Std  R2:", np.std(scores))