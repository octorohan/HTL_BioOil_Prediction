import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bbb2637-sup-0001-datas1.xlsx"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

FIG_DIR = os.path.join(
    OUT_DIR,
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
    columns=["Oil", "Biocrude", "Char", "Gas", "WaterPhase"]
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

print("\nCategorical Columns")
print(cat_cols)

print("\nNumerical Columns")
print(num_cols)

# ============================================================
# PREPROCESSING
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

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()

if hasattr(X_test, "toarray"):
    X_test = X_test.toarray()

feature_names = preprocessor.get_feature_names_out()

print("\nTotal Features:", len(feature_names))

# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("\nRandom Forest trained.")

# ============================================================
# SHAP
# ============================================================

print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(rf)

shap_values = explainer(
    X_test,
    check_additivity=False
)

print("Done.")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = np.abs(
    shap_values.values
).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "MeanAbsSHAP": importance
})

importance_df = importance_df.sort_values(
    "MeanAbsSHAP",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        OUT_DIR,
        "shap_feature_importance.csv"
    ),
    index=False
)

print("\nTop Features")
print(importance_df.head(20))

# ============================================================
# CREATE EXPLANATION WITH FEATURE NAMES
# ============================================================

explanation = shap.Explanation(
    values=shap_values.values,
    base_values=shap_values.base_values,
    data=X_test,
    feature_names=feature_names
)

# ============================================================
# BAR PLOT
# ============================================================

plt.figure(figsize=(10,8))

shap.plots.bar(
    explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "shap_bar.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# BEESWARM
# ============================================================

plt.figure(figsize=(10,8))

shap.plots.beeswarm(
    explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "shap_beeswarm.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved")
print(os.path.join(FIG_DIR, "shap_bar.png"))
print(os.path.join(FIG_DIR, "shap_beeswarm.png"))
print(os.path.join(OUT_DIR, "shap_feature_importance.csv"))