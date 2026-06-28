import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------
# Create output folder
# ----------------------------------------------------------

os.makedirs("outputs/figures", exist_ok=True)

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

# Keep rows with Oil yield
df = df[df["Oil"].notna()].copy()

# Remove columns that leak target information
drop_cols = [
    "Author",
    "Ref",
    "Biocrude",
    "Oil",
    "Char",
    "Gas",
    "WaterPhase",
]

X = df.drop(columns=drop_cols)
y = df["Oil"]

# ----------------------------------------------------------
# Feature types
# ----------------------------------------------------------

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

# ----------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------

numeric_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# ----------------------------------------------------------
# Train/Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    [
        ("prep", preprocessor),
        ("model", model),
    ]
)

pipeline.fit(X_train, y_train)

# ----------------------------------------------------------
# Predictions
# ----------------------------------------------------------

train_pred = pipeline.predict(X_train)
test_pred = pipeline.predict(X_test)

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

from sklearn.metrics import r2_score, mean_absolute_error

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.figure(figsize=(8,8))

plt.scatter(
    y_train,
    train_pred,
    s=20,
    alpha=0.6,
    label="Training",
)

plt.scatter(
    y_test,
    test_pred,
    s=20,
    alpha=0.8,
    label="Test",
)

# Perfect prediction line
plt.plot(
    [0,100],
    [0,100],
    "k--",
    linewidth=2,
    label="Perfect Prediction",
)

plt.xlim(0,100)
plt.ylim(0,100)

plt.xlabel("Experimental Oil Yield (%)", fontsize=13)
plt.ylabel("Predicted Oil Yield (%)", fontsize=13)

plt.title("Random Forest Parity Plot", fontsize=16)

plt.text(
    3,
    95,
    f"Train $R^2$ = {train_r2:.3f}\n"
    f"Test $R^2$ = {test_r2:.3f}\n"
    f"Train MAE = {train_mae:.2f}\n"
    f"Test MAE = {test_mae:.2f}",
    fontsize=11,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.85),
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/figures/parity_plot.png",
    dpi=300,
)

plt.show()

print()
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Train R2 : {train_r2:.4f}")
print(f"Test  R2 : {test_r2:.4f}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test  MAE: {test_mae:.4f}")
print()
print("Figure saved to:")
print("outputs/figures/parity_plot.png")