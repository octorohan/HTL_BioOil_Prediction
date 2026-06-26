import pandas as pd

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df[df["Oil"].notna()].copy()

print("\nTOP RESOURCE TYPES")
print(df["Resource"].value_counts().head(20))

print("\nNUMBER OF UNIQUE DETAILS")
print(df["Details"].nunique())

print("\nTOP SOLVENTS")
print(df["Solvent"].value_counts())

print("\nTOP ADDITIVES")
print(df["Additive"].value_counts().head(20))

print("\nCORRELATION WITH OIL")

num_cols = df.select_dtypes(include=["float64","int64"]).columns

corr = (
    df[num_cols]
    .corr(numeric_only=True)["Oil"]
    .sort_values(ascending=False)
)

print(corr)