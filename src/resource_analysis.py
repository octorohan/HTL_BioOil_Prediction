import pandas as pd

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df[df["Oil"].notna()].copy()

oil_by_resource = (
    df.groupby("Resource")["Oil"]
      .agg(["count", "mean", "std", "min", "max"])
      .sort_values("mean", ascending=False)
)

print("\nRESOURCE STATISTICS")
print("=" * 80)
print(oil_by_resource)

oil_by_resource.to_csv(
    "outputs/resource_statistics.csv"
)

print("\nSaved to outputs/resource_statistics.csv")