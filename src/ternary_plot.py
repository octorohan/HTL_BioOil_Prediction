import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ternary

os.makedirs("outputs/figures", exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df.dropna(how="all")

# --------------------------------------------------
# Build grouped composition
# --------------------------------------------------

df["Carb"] = (
    df["CarboHydrates"].fillna(0)
    + df["Cellulose"].fillna(0)
    + df["HemiCellulose"].fillna(0)
    + df["Lignin"].fillna(0)
    + df["Sugar"].fillna(0)
    + df["Guaiacol"].fillna(0)
    + df["CarboxylicAcids"].fillna(0)
    + df["Glycerol"].fillna(0)
)

df["Protein"] = (
    df["Proteins"].fillna(0)
    + df["AminoAcids"].fillna(0)
)

df["Lipid"] = (
    df["Lipids"].fillna(0)
    + df["FattyAcids"].fillna(0)
)

# --------------------------------------------------
# Keep only rows with meaningful composition
# --------------------------------------------------

total = df["Carb"] + df["Protein"] + df["Lipid"]

df = df[total > 0].copy()

total = (
    df["Carb"]
    + df["Protein"]
    + df["Lipid"]
)

# --------------------------------------------------
# Normalize to 100%
# --------------------------------------------------

df["Carb"] = df["Carb"] / total * 100
df["Protein"] = df["Protein"] / total * 100
df["Lipid"] = df["Lipid"] / total * 100

# --------------------------------------------------
# Plot
# --------------------------------------------------

figure, tax = ternary.figure(scale=100)

figure.set_size_inches(10, 8)

tax.boundary(linewidth=2)

tax.gridlines(
    multiple=10,
    color="gray",
    linewidth=0.4
)

points = list(
    zip(
        df["Carb"],
        df["Protein"],
        df["Lipid"]
    )
)

tax.scatter(
    points,
    marker="o",
    s=8,
    color="tab:green",
    alpha=0.5
)

tax.left_axis_label(
    "Lipids (%)",
    fontsize=14
)

tax.right_axis_label(
    "Proteins (%)",
    fontsize=14
)

tax.bottom_axis_label(
    "Carbohydrates (%)",
    fontsize=14
)

tax.ticks(
    axis="lbr",
    multiple=10,
    linewidth=1
)

tax.clear_matplotlib_ticks()

plt.title(
    "Biomass Composition (Normalized)",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/ternary_composition.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()

print("Saved to")

print("outputs/figures/ternary_composition.png")