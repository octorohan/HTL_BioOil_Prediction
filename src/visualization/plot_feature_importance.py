import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/feature_importance.csv")

top20 = df.head(20)

plt.figure(figsize=(10, 7))

plt.barh(
    top20["Feature"][::-1],
    top20["Importance"][::-1]
)

plt.title("Top 20 Random Forest Features")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "outputs/figures/feature_importance.png",
    dpi=300
)

plt.show()