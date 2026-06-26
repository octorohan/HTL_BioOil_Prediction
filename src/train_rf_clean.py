import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
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
    "Author",

    # remove feedstock identity leakage
    "Details",
    "Origin"
]

X = df.drop(columns=drop_cols)
y = df["Oil"]


cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()


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


model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


pipe.fit(X_train, y_train)

pred_train = pipe.predict(X_train)
pred_test = pipe.predict(X_test)


print("\nCLEAN RANDOM FOREST")
print("=" * 50)

print(f"Train R2 : {r2_score(y_train, pred_train):.4f}")
print(f"Test  R2 : {r2_score(y_test, pred_test):.4f}")

print(f"Train MAE: {mean_absolute_error(y_train, pred_train):.4f}")
print(f"Test  MAE: {mean_absolute_error(y_test, pred_test):.4f}")