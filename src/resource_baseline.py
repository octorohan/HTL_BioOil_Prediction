import pandas as pd

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df[df["Oil"].notna()].copy()

stats = (
    df.groupby("Resource")["Oil"]
      .agg(["count", "mean", "std"])
)

print(stats)

print("\nOverall Mean:")
print(df["Oil"].mean())