import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# =========================
# Load Data
# =========================

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

# Keep only rows where Oil yield exists
df = df[df["Oil"].notna()].copy()


# =========================
# Drop columns
# =========================

drop_cols = [
    "Oil",          # Target
    "Biocrude",     # Output
    "Gas",          # Output
    "Char",         # Output
    "WaterPhase",   # Output
    "Ref",
    "Author",
    "Details",
    "Origin",
    "Additive"
]

X = df.drop(columns=drop_cols)
y = df["Oil"]


# =========================
# Identify column types
# =========================

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

print("Categorical Columns:")
print(cat_cols)

print("\nNumerical Columns:")
print(num_cols)


# =========================
# Preprocessing
# =========================

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


# =========================
# Model
# =========================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# =========================
# Train
# =========================

pipe.fit(X_train, y_train)


# =========================
# Predict
# =========================

pred_train = pipe.predict(X_train)
pred_test = pipe.predict(X_test)


# =========================
# Metrics
# =========================

print("\n====================")
print("RESULTS")
print("====================")

print(f"Train R2 : {r2_score(y_train, pred_train):.4f}")
print(f"Test  R2 : {r2_score(y_test, pred_test):.4f}")

print(f"Train MAE: {mean_absolute_error(y_train, pred_train):.4f}")
print(f"Test  MAE: {mean_absolute_error(y_test, pred_test):.4f}")