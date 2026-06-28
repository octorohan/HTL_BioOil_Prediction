import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "bbb2637-sup-0001-datas1.xlsx")

df = pd.read_excel(DATA_PATH)

df = df[df["Oil"].notna()].copy()

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

# ----------------------------------------------------
# Feature Types
# ----------------------------------------------------

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# ----------------------------------------------------
# Train/Test Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# ----------------------------------------------------
# Metrics
# ----------------------------------------------------

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print("\nMODEL PERFORMANCE")
print("=" * 60)

print(f"Train R2 : {train_r2:.4f}")
print(f"Test  R2 : {test_r2:.4f}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test  MAE: {test_mae:.4f}")

# ----------------------------------------------------
# Estimate Sigma
# ----------------------------------------------------

residuals = y_test - test_pred

sigma = np.std(residuals, ddof=1)

print(f"\nEstimated sigma : {sigma:.2f}")

# ----------------------------------------------------
# Plot
# ----------------------------------------------------

plt.figure(figsize=(10,10))

plt.scatter(
    y_train,
    train_pred,
    s=18,
    alpha=0.6,
    label="Training"
)

plt.scatter(
    y_test,
    test_pred,
    s=18,
    alpha=0.7,
    label="Test"
)

xmin = 0
xmax = 100

x = np.linspace(xmin, xmax, 200)

# Perfect prediction
plt.plot(
    x,
    x,
    "k--",
    linewidth=2,
    label="Perfect Prediction"
)

# ±1 sigma
plt.plot(
    x,
    x + sigma,
    "k--",
    linewidth=1.5,
    alpha=0.8,
    label="±1σ"
)

plt.plot(
    x,
    x - sigma,
    "k--",
    linewidth=1.5,
    alpha=0.8,
)

# ±2 sigma
plt.plot(
    x,
    x + 2*sigma,
    "k:",
    linewidth=2,
    alpha=0.8,
    label="±2σ"
)

plt.plot(
    x,
    x - 2*sigma,
    "k:",
    linewidth=2,
    alpha=0.8,
)

text = (
    f"Train $R^2$ = {train_r2:.3f}\n"
    f"Test $R^2$ = {test_r2:.3f}\n"
    f"Train MAE = {train_mae:.2f}\n"
    f"Test MAE = {test_mae:.2f}\n"
    f"$\\sigma$ = {sigma:.2f}"
)

plt.text(
    3,
    95,
    text,
    fontsize=12,
    va="top",
    bbox=dict(facecolor="white", alpha=0.9)
)

plt.xlabel("Experimental Oil Yield (%)", fontsize=14)
plt.ylabel("Predicted Oil Yield (%)", fontsize=14)

plt.title(
    "Random Forest Parity Plot with Confidence Bands",
    fontsize=18
)

plt.xlim(0,100)
plt.ylim(0,100)

plt.legend()

plt.tight_layout()

save_dir = os.path.join(ROOT, "outputs", "figures")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "parity_sigma.png")

plt.savefig(save_path, dpi=300)

print("\nFigure saved to:")
print(save_path)

plt.show()