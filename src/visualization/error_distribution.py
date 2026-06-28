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

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------

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

# -------------------------------------------------------
# Feature types
# -------------------------------------------------------

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

# -------------------------------------------------------
# Train model
# -------------------------------------------------------

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

# -------------------------------------------------------
# Prediction errors
# -------------------------------------------------------

train_error = train_pred - y_train
test_error = test_pred - y_test

# -------------------------------------------------------
# Empirical CDF
# -------------------------------------------------------

def ecdf(values):
    values = np.sort(values)
    p = np.arange(1, len(values)+1) / len(values)
    return values, p

x_train, y_train_cdf = ecdf(train_error)
x_test, y_test_cdf = ecdf(test_error)

sigma = np.std(test_error, ddof=1)

print("\nEstimated sigma:", round(sigma,2))

# -------------------------------------------------------
# Plot
# -------------------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    x_train,
    y_train_cdf,
    linewidth=2,
    label="Training"
)

plt.plot(
    x_test,
    y_test_cdf,
    linewidth=2,
    label="Test"
)

# sigma lines

for s, style, label in [
    (sigma,"--","±1σ"),
    (2*sigma,":","±2σ")
]:

    plt.axvline(s,color="black",linestyle=style,label=label)
    plt.axvline(-s,color="black",linestyle=style)

plt.axhline(
    0.5,
    color="black",
    linewidth=1
)

plt.xlabel("Prediction Error (%)",fontsize=14)
plt.ylabel("Cumulative Proportion",fontsize=14)

plt.title(
    "Prediction Error Distribution",
    fontsize=18
)

plt.xlim(-30,30)

plt.legend()

plt.grid(alpha=0.25)

plt.tight_layout()

save_dir = os.path.join(ROOT,"outputs","figures")
os.makedirs(save_dir,exist_ok=True)

save_path = os.path.join(
    save_dir,
    "error_distribution.png"
)

plt.savefig(save_path,dpi=300)

print("\nSaved to:")
print(save_path)

plt.show()