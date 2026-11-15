# ==========================================
# MODEL COMPARISON
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# ENTER YOUR MODEL RESULTS
# ==========================================

results = {

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM",
        "Gradient Boosting",
        "XGBoost"
    ],

    "Accuracy": [
        0.7930,
        0.8735,
        0.8730,
        0.5955,
        0.7850,
        0.8800,
        0.8725
    ],

    "Precision": [
        0.7766,
        0.8764,
        0.8700,
        0.5857,
        0.7700,
        0.8789,
        0.8700
    ],

    "Recall": [
        0.7930,
        0.8735,
        0.8730,
        0.5955,
        0.7850,
        0.8800,
        0.8725
    ],

    "F1 Score": [
        0.7814,
        0.8748,
        0.8700,
        0.5842,
        0.7700,
        0.8791,
        0.8700
    ]
}

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(results)

# ==========================================
# CALCULATE OVERALL SCORE
# ==========================================

df["Overall Score"] = (
    df["Accuracy"] +
    df["Precision"] +
    df["Recall"] +
    df["F1 Score"]
) / 4

# ==========================================
# SORT MODELS
# ==========================================

df = df.sort_values(
    by="Overall Score",
    ascending=False
)

df.reset_index(
    drop=True,
    inplace=True
)

# ==========================================
# DISPLAY COMPARISON TABLE
# ==========================================

pd.set_option(
    'display.float_format',
    '{:.4f}'.format
)

print("\n")
print("=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)

print(df)

# ==========================================
# BEST MODEL
# ==========================================

best_model = df.iloc[0]

print("\n")
print("=" * 80)
print("BEST MODEL")
print("=" * 80)

print("Model         :", best_model["Model"])
print("Accuracy      :", best_model["Accuracy"])
print("Precision     :", best_model["Precision"])
print("Recall        :", best_model["Recall"])
print("F1 Score      :", best_model["F1 Score"])
print("Overall Score :", best_model["Overall Score"])

print("\n")
print("=" * 80)
print("MODEL COMPARISON COMPLETED")
print("=" * 80)

# ==========================================
# GRAPH 1 : ACCURACY COMPARISON
# ==========================================

plt.figure(figsize=(12,6))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.title("Accuracy Comparison of Models")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.xticks(
    rotation=45
)

plt.ylim(0,1)

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.6
)

plt.tight_layout()
plt.show()

# ==========================================
# GRAPH 2 : PRECISION, RECALL, F1 COMPARISON
# ==========================================

x = np.arange(len(df["Model"]))
width = 0.25

plt.figure(figsize=(14,7))

plt.bar(
    x - width,
    df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x,
    df["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + width,
    df["F1 Score"],
    width,
    label="F1 Score"
)

plt.title("Precision, Recall and F1 Score Comparison")
plt.xlabel("Models")
plt.ylabel("Scores")

plt.xticks(
    x,
    df["Model"],
    rotation=45
)

plt.ylim(0,1)

plt.legend()

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.6
)

plt.tight_layout()
plt.show()
