import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df[df["Oil"].notna()].copy()

# Create output folder if it doesn't exist
import os
os.makedirs("outputs/figures", exist_ok=True)

# --------------------------------------------------
# 1. Oil Yield Distribution
# --------------------------------------------------

plt.figure(figsize=(8,5))
plt.hist(df["Oil"], bins=40)
plt.title("Distribution of Oil Yield")
plt.xlabel("Oil Yield (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/oil_distribution.png")
plt.close()

# --------------------------------------------------
# 2. Oil vs Lipids
# --------------------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(df["Lipids"], df["Oil"], alpha=0.4)
plt.title("Oil Yield vs Lipids")
plt.xlabel("Lipids")
plt.ylabel("Oil Yield")
plt.tight_layout()
plt.savefig("outputs/figures/oil_vs_lipids.png")
plt.close()

# --------------------------------------------------
# 3. Oil vs Temperature
# --------------------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(df["Temperature"], df["Oil"], alpha=0.4)
plt.title("Oil Yield vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Oil Yield")
plt.tight_layout()
plt.savefig("outputs/figures/oil_vs_temperature.png")
plt.close()

# --------------------------------------------------
# 4. Oil vs HHVResource
# --------------------------------------------------

hhv_df = df[df["HHVResource"].notna()]

plt.figure(figsize=(8,5))
plt.scatter(hhv_df["HHVResource"], hhv_df["Oil"], alpha=0.4)
plt.title("Oil Yield vs HHVResource")
plt.xlabel("HHVResource")
plt.ylabel("Oil Yield")
plt.tight_layout()
plt.savefig("outputs/figures/oil_vs_hhv.png")
plt.close()

# --------------------------------------------------
# 5. Average Oil Yield by Resource
# --------------------------------------------------

resource_mean = (
    df.groupby("Resource")["Oil"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,5))
resource_mean.plot(kind="bar")
plt.title("Average Oil Yield by Resource")
plt.ylabel("Average Oil Yield")
plt.tight_layout()
plt.savefig("outputs/figures/resource_vs_oil.png")
plt.close()

# --------------------------------------------------
# 6. Correlation Heatmap
# --------------------------------------------------

num_df = df.select_dtypes(include=["float64", "int64"])

corr = num_df.corr(numeric_only=True)

plt.figure(figsize=(12,10))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/figures/correlation_heatmap.png")
plt.close()

print("EDA plots saved successfully.")