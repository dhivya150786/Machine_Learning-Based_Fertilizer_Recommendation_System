# ==========================================
# FEATURE IMPORTANCE ANALYSIS USING XGBOOST
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# INPUT FEATURES AND TARGET
# ==========================================

X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

for col in categorical_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN FINAL XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(
    X_scaled,
    y
)

# ==========================================
# EXTRACT FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# ==========================================
# SORT FEATURES BY IMPORTANCE
# ==========================================

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

feature_importance_df.reset_index(
    drop=True,
    inplace=True
)

# ==========================================
# DISPLAY FEATURE IMPORTANCE TABLE
# ==========================================

print("\n")
print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print(feature_importance_df)

# ==========================================
# MOST IMPORTANT FEATURE
# ==========================================

best_feature = feature_importance_df.iloc[0]

print("\n")
print("=" * 80)
print("MOST IMPORTANT FEATURE")
print("=" * 80)

print(
    "Feature Name :",
    best_feature["Feature"]
)

print(
    "Importance Score :",
    best_feature["Importance"]
)

# ==========================================
# FEATURE IMPORTANCE GRAPH
# ==========================================

plt.figure(figsize=(18,8))

plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"],
    color='steelblue'
)

plt.title(
    "Feature Importance using XGBoost",
    fontsize=16
)

plt.xlabel(
    "Features",
    fontsize=12
)

plt.ylabel(
    "Importance Score",
    fontsize=12
)

# Proper alignment of X-axis labels
plt.xticks(
    rotation=90,
    ha='center',
    fontsize=10
)

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.7
)

# Extra space for long labels
plt.subplots_adjust(bottom=0.35)

plt.tight_layout()

plt.show()

# ==========================================
# COMPLETED
# ==========================================

print("\n")
print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 80)
