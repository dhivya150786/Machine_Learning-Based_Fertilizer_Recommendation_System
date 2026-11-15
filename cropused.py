import pandas as pd

# Load dataset
df = pd.read_csv("fertilizer_cleaned.csv")

print("\n===== Crop Names =====")
print(df["Crop_Type"].unique())

print("\n===== Number of Samples per Crop =====")
print(df["Crop_Type"].value_counts())
