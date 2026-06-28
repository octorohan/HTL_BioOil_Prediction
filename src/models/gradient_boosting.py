import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error


df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df[df["Oil"].notna()].copy()

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

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train_processed, y_train)

pred_train = model.predict(X_train_processed)
pred_test = model.predict(X_test_processed)

print("\nGRADIENT BOOSTING")
print("=" * 50)

print(f"Train R2 : {r2_score(y_train, pred_train):.4f}")
print(f"Test  R2 : {r2_score(y_test, pred_test):.4f}")

print(f"Train MAE: {mean_absolute_error(y_train, pred_train):.4f}")
print(f"Test  MAE: {mean_absolute_error(y_test, pred_test):.4f}")