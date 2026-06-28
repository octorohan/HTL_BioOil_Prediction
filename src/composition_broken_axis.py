import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

df = df.dropna(how="all")

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

df["CompositionSum"] = df[composition_cols].fillna(0).sum(axis=1)

# -----------------------------
# Broken axis histogram
# -----------------------------
fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    figsize=(10,6),
    sharex=True,
    gridspec_kw={"height_ratios":[1,3]}
)

bins = 30

ax_top.hist(
    df["CompositionSum"],
    bins=bins,
    edgecolor="black"
)

ax_bottom.hist(
    df["CompositionSum"],
    bins=bins,
    edgecolor="black"
)

# Set y-limits to create the broken axis
ax_top.set_ylim(1300, 1800)
ax_bottom.set_ylim(0, 200)

# Hide spines
ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)

ax_top.tick_params(labeltop=False)
ax_bottom.xaxis.tick_bottom()

# diagonal break marks
d = .008

kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)

ax_top.plot((-d,+d),(-d,+d), **kwargs)
ax_top.plot((1-d,1+d),(-d,+d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)

ax_bottom.plot((-d,+d),(1-d,1+d), **kwargs)
ax_bottom.plot((1-d,1+d),(1-d,1+d), **kwargs)

# ideal composition line
ax_top.axvline(100,color="red",linestyle="--",linewidth=2)
ax_bottom.axvline(100,color="red",linestyle="--",linewidth=2)

ax_bottom.set_xlabel("Sum composition (%)", fontsize=12)
ax_bottom.set_ylabel("Count", fontsize=12)

plt.suptitle("Distribution of Biomass Composition Sum", fontsize=16)

plt.tight_layout()

plt.savefig(
    "outputs/figures/composition_broken_axis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:")
print("outputs/figures/composition_broken_axis.png")