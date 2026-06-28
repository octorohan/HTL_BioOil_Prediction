from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Load Dataset
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bbb2637-sup-0001-datas1.xlsx"

df = pd.read_excel(DATA_PATH)

print("=" * 60)
print("DATASET")
print("=" * 60)
print("Rows :", len(df))
print("Columns :", len(df.columns))


# ==========================================================
# Composition Columns
# ==========================================================

composition_cols = [
    "CarboHydrates",
    "Lignin",
    "Cellulose",
    "HemiCellulose",
    "Sugar",
    "Proteins",
    "Lipids",
    "Ash",
    "Guaiacol",
    "FattyAcids",
    "Glycerol",
    "CarboxylicAcids",
    "AminoAcids",
]

# Missing values treated as zero only for composition summation
composition = df[composition_cols].fillna(0)

df["CompositionSum"] = composition.sum(axis=1)


# ==========================================================
# Statistics
# ==========================================================

print("\n" + "=" * 60)
print("COMPOSITION SUM STATISTICS")
print("=" * 60)

print(df["CompositionSum"].describe())


# ==========================================================
# Quality Analysis
# ==========================================================

good = ((df["CompositionSum"] >= 95) &
        (df["CompositionSum"] <= 105)).sum()

low = (df["CompositionSum"] < 90).sum()

high = (df["CompositionSum"] > 110).sum()

print("\n" + "=" * 60)
print("QUALITY CHECK")
print("=" * 60)

print(f"95-105% : {good}")
print(f"<90%    : {low}")
print(f">110%   : {high}")

print(f"\nPercentage within 95-105% : {100*good/len(df):.2f}%")



# ==========================================================
# Largest Deviations
# ==========================================================

print("\n" + "=" * 60)
print("MOST ABNORMAL COMPOSITIONS")
print("=" * 60)

abnormal = df[
    ["Resource", "Details", "CompositionSum"]
].copy()

abnormal["Deviation"] = abs(abnormal["CompositionSum"] - 100)

abnormal = abnormal.sort_values(
    "Deviation",
    ascending=False
)

print(abnormal.head(20))



# ==========================================================
# Histogram
# ==========================================================

plt.figure(figsize=(9,6))

plt.hist(
    df["CompositionSum"],
    bins=30,
    edgecolor="black"
)

plt.axvline(
    100,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Ideal = 100%"
)

plt.xlabel("Sum of Biomass Composition (%)")
plt.ylabel("Number of Samples")
plt.title("Distribution of Biomass Composition Sum")
plt.legend()

plt.tight_layout()

output_dir = ROOT / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "composition_sum_histogram.png",
    dpi=300
)

plt.close()



# ==========================================================
# Save Statistics
# ==========================================================

stats = pd.DataFrame({
    "Statistic": [
        "Mean",
        "Median",
        "Minimum",
        "Maximum",
        "Std",
        "Within95to105",
        "Below90",
        "Above110"
    ],
    "Value": [
        df["CompositionSum"].mean(),
        df["CompositionSum"].median(),
        df["CompositionSum"].min(),
        df["CompositionSum"].max(),
        df["CompositionSum"].std(),
        good,
        low,
        high
    ]
})

stats.to_csv(
    ROOT / "outputs" / "composition_statistics.csv",
    index=False
)



# ==========================================================
# Save abnormal samples
# ==========================================================

abnormal.to_csv(
    ROOT / "outputs" / "composition_outliers.csv",
    index=False
)


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("Figure : outputs/figures/composition_sum_histogram.png")
print("Stats  : outputs/composition_statistics.csv")
print("Outliers : outputs/composition_outliers.csv")