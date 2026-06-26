import pandas as pd

df = pd.read_excel(r"data\bbb2637-sup-0001-datas1.xlsx")

print("Shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(15))

print("\nOil rows available:")
print(df["Oil"].notna().sum())

print("\nUnique Solvents:")
print(df["Solvent"].value_counts())

print("\nUnique Heating Profiles:")
print(df["HeatingProfile"].value_counts())

print("\nUnique Additives:")
print(df["Additive"].value_counts().head(30))