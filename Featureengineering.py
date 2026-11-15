# ==========================================
# FEATURE ENGINEERING AND PREPROCESSING
# ==========================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# SEPARATE INPUTS AND TARGET
# ==========================================

X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ==========================================
# IDENTIFY CATEGORICAL COLUMNS
# ==========================================

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

print("Categorical Columns:")
print(categorical_columns)

# ==========================================
# LABEL ENCODING INPUT FEATURES
# ==========================================

label_encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col])

    label_encoders[col] = le

# ==========================================
# LABEL ENCODING TARGET VARIABLE
# ==========================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

# Save target encoder
label_encoders["Recommended_Fertilizer"] = target_encoder

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# SAVE ENCODERS AND SCALER
# ==========================================

joblib.dump(label_encoders, "label_encoders.pkl")

joblib.dump(scaler, "scaler.pkl")

print("\nPreprocessing completed successfully.")
