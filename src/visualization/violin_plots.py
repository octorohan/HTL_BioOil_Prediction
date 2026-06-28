import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_excel("data/bbb2637-sup-0001-datas1.xlsx")

os.makedirs("outputs/figures", exist_ok=True)

# --------------------------------------------------
# Columns
# --------------------------------------------------

resource_col = "Resource"
oil_col = "Oil"
char_col = "Char"

# --------------------------------------------------
# Basic Cleaning
# --------------------------------------------------

df = df[[resource_col, oil_col, char_col]].copy()

df = df.dropna(subset=[resource_col])

# Resource order (same order every time)

order = sorted(df[resource_col].unique())

# --------------------------------------------------
# Oil Yield Violin Plot
# --------------------------------------------------

plt.figure(figsize=(11,6))

sns.violinplot(
    data=df,
    x=resource_col,
    y=oil_col,
    order=order,
    inner="quartile",
    cut=0
)

plt.title("Distribution of Bio-oil Yield by Resource", fontsize=15)
plt.xlabel("Resource")
plt.ylabel("Oil Yield (%)")
plt.xticks(rotation=35)

plt.tight_layout()

oil_path = "outputs/figures/violin_oil.png"
plt.savefig(oil_path, dpi=300)
plt.close()

print("Saved:", os.path.abspath(oil_path))

# --------------------------------------------------
# Char Yield Violin Plot
# --------------------------------------------------

char_df = df.dropna(subset=[char_col])

plt.figure(figsize=(11,6))

sns.violinplot(
    data=char_df,
    x=resource_col,
    y=char_col,
    order=order,
    inner="quartile",
    cut=0
)

plt.title("Distribution of Char Yield by Resource", fontsize=15)
plt.xlabel("Resource")
plt.ylabel("Char Yield (%)")
plt.xticks(rotation=35)

plt.tight_layout()

char_path = "outputs/figures/violin_char.png"
plt.savefig(char_path, dpi=300)
plt.close()

print("Saved:", os.path.abspath(char_path))

print("\nDone.")