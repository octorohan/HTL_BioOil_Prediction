import pandas as pd

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df[df["Oil"].notna()].copy()

print("Rows:", len(df))
print("Unique Resources:", df["Resource"].nunique())
print()

print(df["Resource"].value_counts())