# ==========================================
# CONFUSION MATRIX USING XGBOOST
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

from xgboost import XGBClassifier

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

input_encoders = {}

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for col in categorical_columns:

    if col != "Recommended_Fertilizer":

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        input_encoders[col] = encoder

# ==========================================
# ENCODE TARGET
# ==========================================

target_encoder = LabelEncoder()

df["Recommended_Fertilizer"] = target_encoder.fit_transform(
    df["Recommended_Fertilizer"]
)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Recommended_Fertilizer", axis=1)

y = df["Recommended_Fertilizer"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# ACCURACY (Console Only)
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy : {accuracy*100:.2f}%")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

labels = target_encoder.classes_

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor="white",
    cbar=True
)

# ==========================================
# REMOVE ACCURACY FROM TITLE
# ==========================================

plt.title(
    "Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Predicted Fertilizer",
    fontsize=12
)

plt.ylabel(
    "Actual Fertilizer",
    fontsize=12
)

plt.xticks(rotation=45, ha="right")

plt.yticks(rotation=0)

plt.tight_layout()

plt.show()
