# ==========================================
# SAVE FINAL XGBOOST MODEL
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
# ENCODE CATEGORICAL INPUT COLUMNS
# ==========================================

encoders = {}

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

for col in categorical_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])

    encoders[col] = encoder

# ==========================================
# ENCODE TARGET COLUMN
# ==========================================

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
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "fertilizer_model.pkl"
)

# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    "scaler.pkl"
)

# ==========================================
# SAVE TARGET ENCODER
# ==========================================

joblib.dump(
    target_encoder,
    "target_encoder.pkl"
)

# ==========================================
# SAVE INPUT ENCODERS
# ==========================================

joblib.dump(
    encoders,
    "input_encoders.pkl"
)

# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("\n")
print("=" * 70)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print("Saved Files:")
print("1. fertilizer_model.pkl")
print("2. scaler.pkl")
print("3. target_encoder.pkl")
print("4. input_encoders.pkl")

print("\nYour model is now ready for prediction and deployment.")
