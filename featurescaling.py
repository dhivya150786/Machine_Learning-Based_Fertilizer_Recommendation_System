# ==========================================
# FEATURE SCALING
# ==========================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# SEPARATE FEATURES AND TARGET
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
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(scaler, "scaler.pkl")

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

print("\nFeature scaling completed successfully!")
