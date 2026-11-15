# ==========================================
# FINAL BEST MODEL - XGBOOST
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# INPUTS AND OUTPUT
# ==========================================

X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

label_encoders = {}

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

for col in categorical_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    label_encoders[col] = encoder

# Encode target variable

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# FINAL XGBOOST MODEL
# ==========================================

final_model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'
)

# ==========================================
# TRAIN FINAL MODEL
# ==========================================

print("\n===================================")
print("TRAINING FINAL XGBOOST MODEL")
print("===================================\n")

final_model.fit(
    X_scaled,
    y
)

print("Final model trained successfully.")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    final_model,
    "final_xgboost_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    label_encoders,
    "label_encoders.pkl"
)

joblib.dump(
    target_encoder,
    "target_encoder.pkl"
)

# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\n===================================")
print("FINAL MODEL SAVED SUCCESSFULLY")
print("===================================\n")

print("Saved Files:")
print("1. final_xgboost_model.pkl")
print("2. scaler.pkl")
print("3. label_encoders.pkl")
print("4. target_encoder.pkl")

print("\nYour final fertilizer recommendation model is ready.")
