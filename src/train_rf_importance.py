import os

print("Current Working Directory:")
print(os.getcwd())

print("\nFiles in current directory:")
print(os.listdir())

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

# Keep only rows with Oil target available
df = df[df["Oil"].notna()].copy()

print(f"Dataset Shape: {df.shape}")


# ==================================================
# FEATURES / TARGET
# ==================================================

drop_cols = [
    "Oil",          # target
    "Biocrude",     # output
    "Gas",          # output
    "Char",         # output
    "WaterPhase",   # output
    "Ref",
    "Author"
]

X = df.drop(columns=drop_cols)
y = df["Oil"]


# ==================================================
# COLUMN TYPES
# ==================================================

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

print("\nCategorical Columns:")
print(cat_cols)

print("\nNumerical Columns:")
print(num_cols)


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
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# ==================================================
# TRAIN
# ==================================================

pipe.fit(X_train, y_train)


# ==================================================
# PREDICTIONS
# ==================================================

pred_train = pipe.predict(X_train)
pred_test = pipe.predict(X_test)


# ==================================================
# METRICS
# ==================================================

print("\n========================")
print("MODEL PERFORMANCE")
print("========================")

print(f"Train R2 : {r2_score(y_train, pred_train):.4f}")
print(f"Test  R2 : {r2_score(y_test, pred_test):.4f}")

print(f"Train MAE: {mean_absolute_error(y_train, pred_train):.4f}")
print(f"Test  MAE: {mean_absolute_error(y_test, pred_test):.4f}")


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

feature_names = pipe.named_steps[
    "prep"
].get_feature_names_out()

importances = pipe.named_steps[
    "model"
].feature_importances_

imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

imp_df = imp_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========================")
print("TOP 30 FEATURES")
print("========================")

print(imp_df.head(30))


# ==================================================
# SAVE IMPORTANCES
# ==================================================

imp_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")
print("outputs/feature_importance.csv")