# ==========================================
# CROP SEARCH AND VISUALIZATION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# GET CROP NAME FROM USER
# ==========================================

crop_name = input("Enter Crop Name: ")

# ==========================================
# FILTER DATASET
# ==========================================

crop_data = df[
    df["Crop_Type"].str.lower() ==
    crop_name.lower()
]

# ==========================================
# DISPLAY TABLE
# ==========================================

if len(crop_data) == 0:
    print("\nCrop not found in dataset.")

else:

    print("\n===================================")
    print("CROP DETAILS")
    print("===================================\n")

    print(crop_data)

    # ==========================================
    # BAR GRAPH
    # ==========================================

    fertilizer_counts = (
        crop_data["Recommended_Fertilizer"]
        .value_counts()
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        fertilizer_counts.index,
        fertilizer_counts.values
    )

    plt.title(
        f"Fertilizer Distribution for {crop_name}"
    )

    plt.xlabel("Recommended Fertilizer")
    plt.ylabel("Count")

    plt.xticks(rotation=45)

    plt.show()
